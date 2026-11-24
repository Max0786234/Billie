"""Provider-neutral LLM interface.

Every backend (Ollama, OpenAI-compatible HTTP APIs, Anthropic) implements
:class:`LLMProvider`. The rest of Billie only ever talks to this protocol, so a
provider can be swapped at runtime from the settings panel without touching the
chat pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ImagePart:
    """A raw image attached to a message. Providers encode it in their own format."""

    data: bytes
    media_type: str = "image/png"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    images: tuple[ImagePart, ...] = ()


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.2
    max_tokens: int = 1024
    # Ollama only: context window to allocate. Bigger = slower prompt processing on CPU.
    num_ctx: int = 8192
    # Discourages small local models from looping on a phrase (Ollama only).
    repeat_penalty: float = 1.15


@dataclass
class ProviderHealth:
    ok: bool
    provider: str
    model: str
    detail: str = ""
    models: list[str] = field(default_factory=list)


class LLMError(RuntimeError):
    """Raised for any provider failure (network, auth, bad model, refusal)."""


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        """Yield the answer text incrementally. Raises :class:`LLMError` on failure."""
        ...

    def list_models(self) -> list[str]: ...

    def health(self) -> ProviderHealth: ...

    def supports_vision(self) -> bool: ...


def complete(provider: LLMProvider, messages: Sequence[Message], options: GenerationOptions | None = None) -> str:
    """Convenience: run a streaming call to completion and return the full text."""
    return "".join(provider.stream(messages, options)).strip()
