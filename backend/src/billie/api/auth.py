"""API-key authentication.

* ``X-API-Key`` protects chat/data endpoints, ``X-Admin-Key`` protects settings and
  knowledge-base changes.
* In development with no keys configured, the API is open (it binds to localhost).
* In production, startup refuses to run without both keys (see ``runtime.load_settings``),
  so a misconfigured deployment fails closed instead of silently going public.
* Keys are compared in constant time.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


def _check(expected: str | None, provided: str | None, label: str) -> None:
    if not expected:
        return
    if not provided or not hmac.compare_digest(expected.encode(), provided.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid or missing {label}.")


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.services.settings
    _check(settings.api_key, x_api_key, "API key")


def require_admin_key(request: Request, x_admin_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.services.settings
    _check(settings.admin_key, x_admin_key, "admin key")
