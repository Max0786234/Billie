from __future__ import annotations

from billie.llm.base import LLMError, Message, complete
from billie.llm.fallback import FallbackProvider
from billie.llm.manager import LLMManager, LLMSettingsUpdate
from tests.conftest import FakeProvider


class FailingProvider(FakeProvider):
    def __init__(self, error: str = "groq error 429: rate limit reached", after_tokens: int = 0):
        super().__init__("cloud answer here")
        self.name, self.model = "groq", "qwen/qwen3.8-27b"
        self.error, self.after_tokens = error, after_tokens

    def stream(self, messages, options=None):
        self.calls.append(list(messages))
        for i in range(self.after_tokens):
            yield f"t{i} "
        raise LLMError(self.error)


def local():
    p = FakeProvider("local answer")
    p.name, p.model = "ollama", "qwen3:4b-instruct"
    return p


def test_cloud_success_is_used_and_reported():
    cloud = FakeProvider("cloud answer")
    cloud.name, cloud.model = "groq", "qwen/qwen3.8-27b"
    fb = FallbackProvider(cloud, local())
    assert complete(fb, [Message("user", "hi")]) == "cloud answer"
    assert (fb.name, fb.model, fb.fallback_reason) == ("groq", "qwen/qwen3.8-27b", None)


def test_rate_limit_falls_back_and_cools_down():
    now = [0.0]
    cloud, loc = FailingProvider(), local()
    fb = FallbackProvider(cloud, loc, cooldown_seconds=60, clock=lambda: now[0])
    assert complete(fb, [Message("user", "q1")]) == "local answer"
    assert fb.name == "ollama" and "429" in fb.fallback_reason
    complete(fb, [Message("user", "q2")])  # during cool-down: cloud not even tried
    assert len(cloud.calls) == 1 and "cool-down" in fb.fallback_reason
    now[0] = 61
    complete(fb, [Message("user", "q3")])  # cool-down over: cloud retried
    assert len(cloud.calls) == 2


def test_network_error_falls_back_without_cooldown():
    cloud = FailingProvider("Cannot reach groq: connection refused")
    fb = FallbackProvider(cloud, local())
    complete(fb, [Message("user", "a")])
    complete(fb, [Message("user", "b")])
    assert len(cloud.calls) == 2  # no cool-down for non-rate-limit errors


def test_failure_mid_answer_is_not_silently_restarted():
    fb = FallbackProvider(FailingProvider(after_tokens=3), local())
    try:
        complete(fb, [Message("user", "q")])
        raised = False
    except LLMError:
        raised = True
    assert raised


def test_manager_wraps_cloud_providers_and_keeps_local_vision(tmp_path):
    manager = LLMManager(tmp_path / "llm.json", env={})
    manager.update(LLMSettingsUpdate(provider="groq", api_key="gsk_test_1234567890"))
    provider = manager.provider()
    assert isinstance(provider, FallbackProvider) and provider.model == "qwen/qwen3.8-27b"
    vision = manager.vision_provider()
    assert vision is not None and vision.name == "ollama" and vision.model == "gemma3:4b"
    manager.update(LLMSettingsUpdate(local_fallback=False))
    assert not isinstance(manager.provider(), FallbackProvider)


def test_cloud_vision_with_local_fallback(tmp_path):
    manager = LLMManager(tmp_path / "llm.json", env={})
    manager.update(LLMSettingsUpdate(provider="groq", api_key="gsk_test_1234567890"))
    # Save a Gemini key for vision without changing the chat provider.
    public = manager.update(
        LLMSettingsUpdate(vision_cloud="gemini", key_provider="gemini", api_key="AQ.test_key_123456")
    )
    assert public["provider"] == "groq" and public["vision_cloud"] == "gemini"
    assert public["keys"]["gemini"] == "AQ.t…3456" and public["keys"]["groq"] == "gsk_…7890"
    vision = manager.vision_provider()
    assert isinstance(vision, FallbackProvider)
    assert (vision.primary.name, vision.primary.model) == ("gemini", "gemini-3.5-flash-lite")
    assert (vision.fallback.name, vision.fallback.model) == ("ollama", "gemma3:4b")
    assert vision.supports_vision()
    manager.update(LLMSettingsUpdate(vision_cloud="off"))
    assert manager.vision_provider().name == "ollama"
