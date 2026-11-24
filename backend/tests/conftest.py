from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from billie.llm.base import GenerationOptions, LLMError, Message, ProviderHealth
from billie.llm.manager import LLMSettings
from billie.runtime import load_settings


class FakeProvider:
    """Scripted LLM: records the prompts it receives and replays a canned answer."""

    def __init__(self, answer: str = "Answer from the sources [1].", *, fail: bool = False, vision: bool = False):
        self.name = "fake"
        self.model = "fake-model"
        self.answer = answer
        self.fail = fail
        self.vision = vision
        self.calls: list[list[Message]] = []

    def stream(self, messages: Sequence[Message], options: GenerationOptions | None = None) -> Iterator[str]:
        self.calls.append(list(messages))
        if self.fail:
            raise LLMError("model offline")
        words = self.answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "

    def list_models(self) -> list[str]:
        return [self.model]

    def health(self) -> ProviderHealth:
        return ProviderHealth(ok=not self.fail, provider=self.name, model=self.model, detail="fake")

    def supports_vision(self) -> bool:
        return self.vision


class FakeLLMManager:
    def __init__(self, chat: FakeProvider | None = None, vision: FakeProvider | None = None):
        self.chat = chat or FakeProvider()
        self.vision = vision
        self.settings = LLMSettings()

    def provider(self):
        return self.chat

    def vision_provider(self):
        if self.chat.supports_vision():
            return self.chat
        return self.vision

    def public_settings(self):
        return {"provider": "fake", "model": self.chat.model, "keys": {}}


KNOWLEDGE = {
    "billie.md": "# Billie\n\nBillie answers questions from documents and cites them.\n\n"
    "## Refresh\n\nThe knowledge index is rebuilt automatically when a document changes.",
    "security.md": "# Security\n\nAdministrative actions require the X-Admin-Key header. "
    "Phishing emails should be reported to security@example.com.",
}


@pytest.fixture()
def settings(tmp_path: Path):
    env = {"BILLIE_EMBED_MODEL": "off"}
    s = load_settings(env, data_dir=tmp_path / "data")
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    for name, text in KNOWLEDGE.items():
        (knowledge / name).write_text(text, encoding="utf-8")
    object.__setattr__(s.paths, "knowledge_dir", knowledge)
    object.__setattr__(s.paths, "repo_root", tmp_path)
    return s


@pytest.fixture()
def fake_llm() -> FakeLLMManager:
    return FakeLLMManager()
