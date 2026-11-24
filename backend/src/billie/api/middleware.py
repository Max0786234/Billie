"""HTTP middleware written as pure ASGI.

Why not ``@app.middleware("http")``? Starlette's ``BaseHTTPMiddleware`` relays the response
body through its own internal stream, and the client's ``http.disconnect`` then never
reaches the ``StreamingResponse``. Measured: with one such middleware, closing the browser
mid-answer did not stop the generator at all; it kept the local model busy for 20-35 s
and blocked the next question. Pure ASGI middleware only wraps ``send`` and passes
``receive`` through untouched, so disconnects propagate and generation stops within ~0.1 s.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from billie.api.logging_setup import request_id_var
from billie.api.metrics import Metrics
from billie.api.security import LIMITED_PREFIXES, SECURITY_HEADERS, TokenBucketLimiter

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SECURITY_HEADER_BYTES = [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key == name:
            return str(value.decode("latin-1"))
    return ""


class MetricsMiddleware:
    """Prometheus request counter and latency histogram, labelled by route template."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.http_requests.labels(scope["method"], route, str(status["code"])).inc()
            self.metrics.http_latency.labels(route).observe(time.perf_counter() - started)


class SecurityMiddleware:
    """Rate-limits expensive POST routes per client and adds browser security headers."""

    def __init__(self, app: ASGIApp, limiter: TokenBucketLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self.limiter.enabled and scope["method"] == "POST" and scope["path"].startswith(LIMITED_PREFIXES):
            client = scope.get("client") or ("unknown", 0)
            allowed, retry_after = self.limiter.allow(client[0])
            if not allowed:
                response = JSONResponse(
                    {"detail": "Too many requests. Please wait a moment."},
                    status_code=429,
                    headers={"Retry-After": str(max(1, round(retry_after)))},
                )
                await self._with_headers(response, scope, receive, send)
                return
        await self.app(scope, receive, self._header_sender(send))

    def _header_sender(self, send: Send) -> Send:
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                present = {key.lower() for key, _ in headers}
                headers.extend((k, v) for k, v in _SECURITY_HEADER_BYTES if k not in present)
                message = {**message, "headers": headers}
            await send(message)

        return send_wrapper

    async def _with_headers(self, response: JSONResponse, scope: Scope, receive: Receive, send: Send) -> None:
        await response(scope, receive, self._header_sender(send))


class RequestIdMiddleware:
    """Assigns an X-Request-ID, exposes it to log records, and logs one line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = logging.getLogger("billie.http")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        request_id = incoming if _VALID_ID.match(incoming) else secrets.token_hex(8)
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)
            if not scope["path"].startswith(("/assets", "/metrics")):
                self.log.info(
                    "request",
                    extra={
                        "fields": {
                            "request_id": request_id,
                            "method": scope["method"],
                            "path": scope["path"],
                            "status": status["code"],
                            "ms": round((time.perf_counter() - started) * 1000, 1),
                        }
                    },
                )
