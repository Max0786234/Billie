"""Public demo mode: visitor identity and daily question quotas.

A public deployment (e.g. a Hugging Face Space) runs on a free cloud-model quota
(Groq: ~1k requests and 200k tokens per day). Without limits, one visitor or a bot
could use it up in minutes and the demo would be broken for everyone else, including
the interviewer it is meant for. So in demo mode:

* each browser sends a random ``X-Visitor-Id``; conversations are stored under a hash
  of it, so visitors only ever see their own chats;
* questions are counted per client IP *and* in total per UTC day. The IP (not the
  visitor id) is used for the personal cap because clearing browser storage would
  reset a visitor id;
* answers served from the cache are refunded, since they cost no model call.

Counters live in memory: a restart resets them, which only ever errs on the side of
availability. One process serves the Space, so no shared store is needed.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable

from fastapi import Request

_VISITOR = re.compile(r"^[A-Za-z0-9-]{16,64}$")


def _utc_day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


class DailyQuota:
    def __init__(self, total: int, per_client: int, clock: Callable[[], float] = time.time) -> None:
        self.total = total
        self.per_client = per_client
        self.clock = clock
        self._day = _utc_day(clock())
        self._used = 0
        self._by_client: dict[str, int] = {}
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = _utc_day(self.clock())
        if day != self._day:
            self._day, self._used, self._by_client = day, 0, {}

    def take(self, client: str) -> str | None:
        """Count one question. Returns None if allowed, else a message for the visitor."""
        with self._lock:
            self._roll()
            if self._used >= self.total:
                return (
                    "The public demo has reached its daily limit (it runs on a free model quota). "
                    "Please come back tomorrow, or run Billie locally - see the README."
                )
            if self._by_client.get(client, 0) >= self.per_client:
                return f"You have used today's {self.per_client} demo questions. Please come back tomorrow."
            self._used += 1
            self._by_client[client] = self._by_client.get(client, 0) + 1
            return None

    def refund(self, client: str) -> None:
        with self._lock:
            self._roll()
            self._used = max(0, self._used - 1)
            if self._by_client.get(client):
                self._by_client[client] -= 1

    def stats(self, client: str | None = None) -> dict[str, int]:
        with self._lock:
            self._roll()
            out = {"daily_limit": self.total, "used_today": self._used, "per_visitor_limit": self.per_client}
            if client is not None:
                out["remaining_for_you"] = max(
                    0, min(self.per_client - self._by_client.get(client, 0), self.total - self._used)
                )
            return out


def client_ip(request: Request) -> str:
    # Behind the Space proxy, uvicorn's --proxy-headers puts the real visitor IP here.
    return request.client.host if request.client else "unknown"


def visitor_owner(request: Request) -> str | None:
    """Conversation owner for this request: None outside demo mode (one local user)."""
    if not request.app.state.services.settings.demo:
        return None
    raw = request.headers.get("x-visitor-id", "")
    # No (valid) id: fall back to the IP so the visitor still gets a private, if less stable, space.
    seed = raw if _VISITOR.match(raw) else f"ip:{client_ip(request)}"
    return "v_" + hashlib.sha256(seed.encode()).hexdigest()[:32]
