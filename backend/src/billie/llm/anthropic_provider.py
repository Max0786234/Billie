"""Anthropic (Claude) provider, built on the official ``anthropic`` SDK.

Optional: install with ``pip install -e ".[anthropic]"``. The SDK is imported
lazily so Billie runs fully offline without it.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from typing import Any

from billie.llm.base import GenerationOptions, LLMError, Message, ProviderHealth

DEFAULT_MODEL = "claude-opus-5-5"
SUGGESTED_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
# Server-side refusal fallback: on a policy decline the API re-runs the request on a
# suitable model inside the same call. Supported on the Opus 5.5 / Sonnet 5.5 line.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: str | None, effort: str = "low") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise LLMError('Anthropic provider needs the SDK: pip install -e ".[anthropic]"') from exc
        if not api_key:
            raise LLMError("Anthropic provider needs an API key.")
        self._sdk = anthropic
        self.model = model
        # Chat answers over retrieved context do not need deep reasoning; `low` effort
        # keeps latency and cost down. Configurable from settings.
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=120.0)

    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        options = options or GenerationOptions()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        conversation = [_to_anthropic(m) for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(options.max_tokens, 4096),
            "messages": conversation,
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        if self.model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return self._stream(kwargs)

    def _stream(self, kwargs: dict[str, Any]) -> Iterator[str]:
        sdk = self._sdk
        try:
            with self._client.beta.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                raise LLMError("Claude declined to answer this request.")
        except sdk.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key.") from exc
        except sdk.NotFoundError as exc:
            raise LLMError(f"Unknown Anthropic model '{self.model}'.") from exc
        except sdk.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached; retry shortly.") from exc
        except sdk.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMError(f"Cannot reach Anthropic: {exc}") from exc

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self._client.models.list())
        except self._sdk.APIError as exc:
            raise LLMError(f"Anthropic models list failed: {exc}") from exc

    def supports_vision(self) -> bool:
        return True  # all current Claude models accept images

    def health(self) -> ProviderHealth:
        try:
            models = self.list_models()
        except LLMError as exc:
            return ProviderHealth(ok=False, provider=self.name, model=self.model, detail=str(exc))
        return ProviderHealth(ok=True, provider=self.name, model=self.model, detail="ready", models=models)


def _to_anthropic(message: Message) -> dict[str, Any]:
    if not message.images:
        return {"role": message.role, "content": message.content}
    blocks: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64.standard_b64encode(image.data).decode("ascii"),
            },
        }
        for image in message.images
    ]
    blocks.append({"type": "text", "text": message.content})
    return {"role": message.role, "content": blocks}
