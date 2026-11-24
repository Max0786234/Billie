"""Cloud-first generation with automatic fallback to the local model.

A free cloud tier (e.g. Groq: 8k tokens/minute, 200k/day) is ~10x faster than a CPU-only
laptop, but it can hit rate limits or lose the network. ``FallbackProvider`` tries the
cloud provider first and, if it fails *before producing any text*, answers with the local
Ollama model instead. After a rate limit (HTTP 429) the cloud is skipped for a cool-down,
so every following question doesn't pay for a failed request first.

``name`` / ``model`` always describe the provider that actually answered the last request,
and ``fallback_reason`` explains a switch - the pipeline shows it in the answer details.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Sequence

from billie.llm.base import GenerationOptions, LLMError, LLMProvider, Message, ProviderHealth

log = logging.getLogger(__name__)


class FallbackProvider:
    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        *,
        cooldown_seconds: float = 60.0,
        clock=time.monotonic,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._skip_primary_until = 0.0
        self._lock = threading.Lock()
        self.active: LLMProvider = primary
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def model(self) -> str:
        return self.active.model

    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        with self._lock:
            cooling = self._clock() < self._skip_primary_until
        if not cooling:
            produced = False
            try:
                self.active, self.fallback_reason = self.primary, None
                for token in self.primary.stream(messages, options):
                    produced = True
                    yield token
                return
            except LLMError as exc:
                if produced:  # a half-written answer can't be silently restarted on another model
                    raise
                reason = str(exc)
                if "429" in reason or "rate" in reason.lower():
                    with self._lock:
                        self._skip_primary_until = self._clock() + self.cooldown_seconds
                log.warning("cloud model failed, answering with the local model: %s", reason)
                self.fallback_reason = f"{self.primary.name} unavailable ({reason[:120]}); answered locally"
        else:
            self.fallback_reason = f"{self.primary.name} rate-limited; answered locally during the cool-down"
        self.active = self.fallback
        yield from self.fallback.stream(messages, options)

    def list_models(self) -> list[str]:
        return self.primary.list_models()

    def supports_vision(self) -> bool:
        return self.primary.supports_vision()

    def health(self) -> ProviderHealth:
        health = self.primary.health()
        if not health.ok:
            local = self.fallback.health()
            health.detail = f"{health.detail} - local fallback {'ready' if local.ok else 'unavailable'}"
            health.ok = local.ok  # Billie can still answer
        return health
