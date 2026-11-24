"""HTTP hardening: per-client rate limiting for expensive routes and standard security headers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

# Routes that run the LLM, embeddings or indexing - the ones worth protecting from floods.
LIMITED_PREFIXES = ("/api/chat", "/api/uploads", "/api/documents", "/api/llm/test")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
    ),
}


class TokenBucketLimiter:
    """In-memory token bucket per client key. ``rate_per_minute`` <= 0 disables limiting.

    Single-process by design (Billie runs as one process); a multi-instance deployment
    would move this to Redis or the reverse proxy.
    """

    def __init__(self, rate_per_minute: int, burst: int | None = None, clock: Callable[[], float] = time.monotonic):
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(rate_per_minute // 3, 1))
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def allow(self, key: str) -> tuple[bool, float]:
        """(allowed, seconds until the next token if refused)."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1 - tokens) / self.rate
