"""Runtime-switchable LLM configuration.

The UI's settings panel reads and writes :class:`LLMSettings` through this
manager. Settings persist to ``data/llm_settings.json`` (git-ignored). API keys
are stored per provider so switching back and forth keeps them, and they are
never returned in full by :meth:`LLMManager.public_settings`.

Security note: this is a single-user, local-first app; keys sit in a local file
with the same trust level as a ``.env`` file. A multi-user deployment should use
a secrets manager instead.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from billie.fileio import atomic_write_json
from billie.llm.anthropic_provider import SUGGESTED_MODELS as ANTHROPIC_MODELS
from billie.llm.anthropic_provider import AnthropicProvider
from billie.llm.base import LLMError, LLMProvider
from billie.llm.fallback import FallbackProvider
from billie.llm.ollama import DEFAULT_BASE_URL as OLLAMA_URL
from billie.llm.ollama import OllamaProvider
from billie.llm.openai_compat import OpenAICompatibleProvider

ProviderId = Literal["ollama", "openai", "groq", "openrouter", "gemini", "anthropic", "custom"]

DEFAULT_OLLAMA_CHAT_MODEL = "qwen3:4b-instruct"
DEFAULT_OLLAMA_VISION_MODEL = "gemma3:4b"


@dataclass(frozen=True)
class ProviderPreset:
    id: ProviderId
    label: str
    kind: Literal["ollama", "openai", "anthropic"]
    base_url: str
    default_model: str
    suggested_models: tuple[str, ...]
    needs_key: bool
    key_env: str | None
    key_url: str | None = None


PRESETS: dict[str, ProviderPreset] = {
    p.id: p
    for p in [
        ProviderPreset(
            "ollama",
            "Ollama (local, free)",
            "ollama",
            OLLAMA_URL,
            DEFAULT_OLLAMA_CHAT_MODEL,
            (DEFAULT_OLLAMA_CHAT_MODEL, DEFAULT_OLLAMA_VISION_MODEL, "llama3.2", "qwen3:8b"),
            needs_key=False,
            key_env=None,
            key_url="https://ollama.com/library",
        ),
        ProviderPreset(
            "anthropic",
            "Anthropic Claude",
            "anthropic",
            "https://api.anthropic.com",
            ANTHROPIC_MODELS[0],
            tuple(ANTHROPIC_MODELS),
            needs_key=True,
            key_env="ANTHROPIC_API_KEY",
            key_url="https://console.anthropic.com/settings/keys",
        ),
        ProviderPreset(
            "openai",
            "OpenAI",
            "openai",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"),
            needs_key=True,
            key_env="OPENAI_API_KEY",
            key_url="https://platform.openai.com/api-keys",
        ),
        ProviderPreset(
            "groq",
            "Groq (fast, free tier)",
            "openai",
            "https://api.groq.com/openai/v1",
            "qwen/qwen3.8-27b",
            ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
            needs_key=True,
            key_env="GROQ_API_KEY",
            key_url="https://console.groq.com/keys",
        ),
        ProviderPreset(
            "gemini",
            "Google Gemini",
            "openai",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.5-flash-lite",
            ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-flash-lite"),
            needs_key=True,
            key_env="GEMINI_API_KEY",
            key_url="https://aistudio.google.com/apikey",
        ),
        ProviderPreset(
            "openrouter",
            "OpenRouter",
            "openai",
            "https://openrouter.ai/api/v1",
            "meta-llama/llama-3.3-70b-instruct",
            ("meta-llama/llama-3.3-70b-instruct",),
            needs_key=True,
            key_env="OPENROUTER_API_KEY",
            key_url="https://openrouter.ai/keys",
        ),
        ProviderPreset(
            "custom",
            "Custom OpenAI-compatible",
            "openai",
            "http://localhost:1234/v1",
            "local-model",
            (),
            needs_key=False,
            key_env=None,
        ),
    ]
}


class LLMSettings(BaseModel):
    provider: ProviderId = "ollama"
    model: str = DEFAULT_OLLAMA_CHAT_MODEL
    base_url: str = OLLAMA_URL
    # Local Ollama vision model; reads images whenever the chat model is text-only (cloud or local).
    vision_model: str = DEFAULT_OLLAMA_VISION_MODEL
    # Cloud providers fall back to this local model on errors, rate limits or no internet.
    local_fallback: bool = True
    fallback_model: str = DEFAULT_OLLAMA_CHAT_MODEL
    # Optional cloud vision (e.g. Gemini: ~11 s vs ~51 s locally), falling back to the local vision model.
    vision_cloud: ProviderId | None = None
    vision_cloud_model: str = "gemini-3.5-flash-lite"
    ollama_url: str = OLLAMA_URL
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=700, ge=64, le=8192)
    api_keys: dict[str, str] = Field(default_factory=dict)


class LLMSettingsUpdate(BaseModel):
    provider: ProviderId | None = None
    model: str | None = None
    base_url: str | None = None
    vision_model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)
    local_fallback: bool | None = None
    vision_cloud: ProviderId | Literal["off"] | None = None  # "off" disables cloud vision
    vision_cloud_model: str | None = None
    # Stores the key for this provider (defaults to ``provider``); lets a vision key be saved separately.
    key_provider: ProviderId | None = None
    # None = keep the stored key; "" = delete it.
    api_key: str | None = None


def mask_key(key: str | None) -> str | None:
    if not key:
        return None
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "••••"


class LLMManager:
    def __init__(self, settings_path: Path, *, env: dict[str, str] | None = None) -> None:
        self.settings_path = settings_path
        self._env = dict(os.environ if env is None else env)
        self._lock = threading.RLock()
        self._settings = self._load()
        self._provider: LLMProvider | None = None
        self._vision: LLMProvider | None = None

    # ------------------------------------------------------------ settings
    def _load(self) -> LLMSettings:
        if self.settings_path.exists():
            try:
                return LLMSettings.model_validate(json.loads(self.settings_path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass  # corrupt file -> fall back to env defaults
        provider = self._env.get("BILLIE_LLM_PROVIDER", "ollama")
        vision_cloud = self._env.get("BILLIE_VISION_CLOUD", "").strip() or None
        preset = PRESETS.get(provider, PRESETS["ollama"])
        return LLMSettings(
            provider=preset.id,
            model=self._env.get("BILLIE_LLM_MODEL", preset.default_model),
            base_url=self._env.get("OLLAMA_BASE_URL", preset.base_url) if preset.kind == "ollama" else preset.base_url,
            vision_model=self._env.get("BILLIE_VISION_MODEL", DEFAULT_OLLAMA_VISION_MODEL),
            # Deployments without local models (e.g. a Space) turn the local fallback off.
            local_fallback=self._env.get("BILLIE_LOCAL_FALLBACK", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            vision_cloud=vision_cloud if vision_cloud in PRESETS else None,  # type: ignore[arg-type]
            vision_cloud_model=self._env.get("BILLIE_VISION_CLOUD_MODEL", "gemini-3.5-flash-lite"),
        )

    @property
    def settings(self) -> LLMSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def api_key_for(self, provider: str) -> str | None:
        preset = PRESETS[provider]
        stored = self._settings.api_keys.get(provider)
        return stored or (self._env.get(preset.key_env) if preset.key_env else None)

    def public_settings(self) -> dict:
        with self._lock:
            s = self._settings
            return {
                "provider": s.provider,
                "model": s.model,
                "base_url": s.base_url,
                "vision_model": s.vision_model,
                "temperature": s.temperature,
                "max_tokens": s.max_tokens,
                "local_fallback": s.local_fallback,
                "fallback_model": s.fallback_model,
                "vision_cloud": s.vision_cloud,
                "vision_cloud_model": s.vision_cloud_model,
                "keys": {pid: mask_key(self.api_key_for(pid)) for pid, p in PRESETS.items() if p.needs_key},
            }

    def update(self, patch: LLMSettingsUpdate) -> dict:
        with self._lock:
            data = self._settings.model_dump()
            provider_changed = patch.provider is not None and patch.provider != self._settings.provider
            if patch.provider is not None:
                data["provider"] = patch.provider
                if provider_changed:
                    # Switching provider resets model/url to that provider's defaults unless given.
                    preset = PRESETS[patch.provider]
                    data["model"] = preset.default_model
                    data["base_url"] = preset.base_url
            if patch.vision_cloud is not None:
                data["vision_cloud"] = None if patch.vision_cloud == "off" else patch.vision_cloud
            for field in (
                "model",
                "base_url",
                "vision_model",
                "temperature",
                "max_tokens",
                "local_fallback",
                "vision_cloud_model",
            ):
                value = getattr(patch, field)
                if value is not None:
                    data[field] = value.strip() if isinstance(value, str) else value
            if patch.api_key is not None:
                keys = dict(data["api_keys"])
                target = patch.key_provider or data["provider"]
                if patch.api_key.strip():
                    keys[target] = patch.api_key.strip()
                else:
                    keys.pop(target, None)
                data["api_keys"] = keys
            candidate = LLMSettings.model_validate(data)
            # Validate by constructing (not calling) the provider: catches missing keys/SDK early.
            self._build(candidate)
            self._settings = candidate
            self._provider = None
            self._vision = None
            atomic_write_json(self.settings_path, candidate.model_dump())
            return self.public_settings()

    # ----------------------------------------------------------- providers
    def provider(self) -> LLMProvider:
        with self._lock:
            if self._provider is None:
                self._provider = self._build(self._settings)
            return self._provider

    def vision_provider(self) -> LLMProvider | None:
        """Provider able to read images, or None if none is configured/available."""
        with self._lock:
            chat = self.provider()
            try:
                if chat.supports_vision():
                    return chat
            except LLMError:
                return None
            if self._vision is None:
                self._vision = self._build_vision(self._settings)
            return self._vision

    def _build_vision(self, settings: LLMSettings) -> LLMProvider | None:
        """Cloud vision (if configured and keyed) with the local vision model as fallback; else local only."""
        url = settings.base_url if settings.provider == "ollama" else settings.ollama_url
        local = OllamaProvider(settings.vision_model, url) if settings.vision_model else None
        cloud_id = settings.vision_cloud
        key = self.api_key_for(cloud_id) if cloud_id else None
        if cloud_id and key and PRESETS[cloud_id].kind == "openai":
            cloud = OpenAICompatibleProvider(
                name=cloud_id,
                model=settings.vision_cloud_model,
                base_url=PRESETS[cloud_id].base_url,
                api_key=key,
                timeout=30.0,  # free tiers can hang; fall back quickly instead of waiting
                vision=True,
            )
            return FallbackProvider(cloud, local) if local else cloud
        return local

    def preview(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> LLMProvider:
        """Build (but do not save or activate) a provider from the current settings plus overrides."""
        with self._lock:
            data = self._settings.model_dump()
        if provider and provider != data["provider"]:
            if provider not in PRESETS:
                raise LLMError(f"Unknown provider '{provider}'.")
            data.update(provider=provider, model=PRESETS[provider].default_model, base_url=PRESETS[provider].base_url)
        if model:
            data["model"] = model
        if base_url:
            data["base_url"] = base_url
        if api_key:
            data["api_keys"] = {**data["api_keys"], data["provider"]: api_key}
        return self._build(LLMSettings.model_validate(data))

    def _build(self, settings: LLMSettings) -> LLMProvider:
        preset = PRESETS[settings.provider]
        key = settings.api_keys.get(settings.provider) or (self._env.get(preset.key_env) if preset.key_env else None)
        if preset.needs_key and not key:
            raise LLMError(f"{preset.label} needs an API key. Add one in Settings or set {preset.key_env}.")
        if preset.kind == "ollama":
            return OllamaProvider(settings.model, settings.base_url)
        if preset.kind == "anthropic":
            return self._with_fallback(AnthropicProvider(model=settings.model, api_key=key), settings)
        cloud: LLMProvider = OpenAICompatibleProvider(
            name=preset.id, model=settings.model, base_url=settings.base_url, api_key=key
        )
        return self._with_fallback(cloud, settings)

    @staticmethod
    def _with_fallback(cloud: LLMProvider, settings: LLMSettings) -> LLMProvider:
        if not settings.local_fallback:
            return cloud
        return FallbackProvider(cloud, OllamaProvider(settings.fallback_model, settings.ollama_url))

    @staticmethod
    def presets() -> list[dict]:
        return [
            {
                "id": p.id,
                "label": p.label,
                "base_url": p.base_url,
                "default_model": p.default_model,
                "suggested_models": list(p.suggested_models),
                "needs_key": p.needs_key,
                "key_url": p.key_url,
            }
            for p in PRESETS.values()
        ]
