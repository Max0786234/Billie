"""Ollama provider (local, free, offline). https://github.com/ollama/ollama/blob/main/docs/api.md"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Sequence
from functools import lru_cache

import httpx

from billie.llm.base import GenerationOptions, LLMError, Message, ProviderHealth
from billie.llm.think_filter import strip_think

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 300.0,
        keep_alive: str = "30m",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = normalize_base_url(base_url)
        self.keep_alive = keep_alive
        # Generous read timeout: first token on CPU can take a while for long prompts.
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    # ------------------------------------------------------------------ chat
    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        options = options or GenerationOptions()
        payload = {
            "model": self.model,
            "messages": [_to_ollama(message) for message in messages],
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": options.temperature,
                "num_predict": options.max_tokens,
                "num_ctx": options.num_ctx,
                "repeat_penalty": options.repeat_penalty,
            },
        }
        return strip_think(self._stream_tokens(payload))

    def _stream_tokens(self, payload: dict) -> Iterator[str]:
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(f"Ollama error {response.status_code}: {_error_text(response)}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise LLMError(f"Ollama error: {event['error']}")
                    token = event.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------ embeddings
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        try:
            response = self._client.post("/api/embed", json={"model": model, "input": list(texts)})
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"Ollama embed error {response.status_code}: {_error_text(response)}")
        return response.json()["embeddings"]

    # ------------------------------------------------------------- discovery
    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        return sorted(model["name"] for model in response.json().get("models", []))

    def capabilities(self, model: str | None = None) -> frozenset[str]:
        return _capabilities(self._client, model or self.model)

    def supports_vision(self) -> bool:
        return "vision" in self.capabilities()

    def health(self) -> ProviderHealth:
        try:
            models = self.list_models()
        except LLMError as exc:
            return ProviderHealth(ok=False, provider=self.name, model=self.model, detail=str(exc))
        if self.model not in models:
            return ProviderHealth(
                ok=False,
                provider=self.name,
                model=self.model,
                detail=f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}",
                models=models,
            )
        return ProviderHealth(ok=True, provider=self.name, model=self.model, detail="ready", models=models)


def normalize_base_url(url: str) -> str:
    """Use 127.0.0.1 instead of localhost.

    On Windows "localhost" resolves to IPv6 ::1 first, while Ollama listens on IPv4 only,
    so every new connection waits ~2.3 s for the IPv6 attempt to fail (measured).
    """
    return url.rstrip("/").replace("://localhost", "://127.0.0.1")


def _capabilities(client: httpx.Client, model: str) -> frozenset[str]:
    # Cache per (base_url, model): capabilities never change for a pulled model.
    return _cached_capabilities(str(client.base_url), model, client)


@lru_cache(maxsize=64)
def _cached_capabilities(_base: str, model: str, client: httpx.Client) -> frozenset[str]:
    try:
        response = client.post("/api/show", json={"model": model}, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return frozenset()
    return frozenset(response.json().get("capabilities", []))


def _to_ollama(message: Message) -> dict:
    item: dict = {"role": message.role, "content": message.content}
    if message.images:
        item["images"] = [base64.b64encode(image.data).decode("ascii") for image in message.images]
    return item


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text[:300]
