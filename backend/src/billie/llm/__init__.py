"""Provider-neutral LLM layer: Ollama, OpenAI-compatible APIs and Anthropic."""

from billie.llm.base import GenerationOptions, ImagePart, LLMError, LLMProvider, Message, ProviderHealth, complete
from billie.llm.manager import LLMManager, LLMSettings, LLMSettingsUpdate

__all__ = [
    "GenerationOptions",
    "ImagePart",
    "LLMError",
    "LLMManager",
    "LLMProvider",
    "LLMSettings",
    "LLMSettingsUpdate",
    "Message",
    "ProviderHealth",
    "complete",
]
