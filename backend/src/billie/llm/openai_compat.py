"""Provider for any OpenAI-compatible Chat Completions API.

Covers OpenAI, Groq, OpenRouter, Google Gemini (OpenAI endpoint), Together,
LM Studio, vLLM, etc. Only the base URL, key and model differ.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Sequence

import httpx

from billie.llm.base import GenerationOptions, LLMError, Message, ProviderHealth
from billie.llm.think_filter import strip_think

# Heuristic: names of model families known to accept image input.
_VISION_HINTS = ("gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4", "gemini", "vision", "-vl", "llava", "pixtral", "llama-4")


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        vision: bool | None = None,
    ) -> None:
        self._vision = vision  # explicit override; otherwise guessed from the model name
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        options = options or GenerationOptions()
        payload = {
            "model": self.model,
            "messages": [_to_openai(message) for message in messages],
            "stream": True,
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
        }
        return strip_think(self._stream_tokens(payload))

    def _stream_tokens(self, payload: dict) -> Iterator[str]:
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(f"{self.name} error {response.status_code}: {response.text[:300]}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    token = (choices[0].get("delta") or {}).get("content") or ""
                    if token:
                        yield token
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach {self.name} at {self.base_url}: {exc}") from exc

    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/models", timeout=10.0)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach {self.name}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"{self.name} error {response.status_code}: {response.text[:200]}")
        return sorted(item["id"] for item in response.json().get("data", []))

    def supports_vision(self) -> bool:
        if self._vision is not None:
            return self._vision
        return any(hint in self.model.lower() for hint in _VISION_HINTS)

    def health(self) -> ProviderHealth:
        try:
            models = self.list_models()
        except LLMError as exc:
            return ProviderHealth(ok=False, provider=self.name, model=self.model, detail=str(exc))
        return ProviderHealth(ok=True, provider=self.name, model=self.model, detail="ready", models=models)


def _to_openai(message: Message) -> dict:
    if not message.images:
        return {"role": message.role, "content": message.content}
    parts: list[dict] = [{"type": "text", "text": message.content}]
    for image in message.images:
        encoded = base64.b64encode(image.data).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{encoded}"}})
    return {"role": message.role, "content": parts}
