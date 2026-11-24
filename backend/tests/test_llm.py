from __future__ import annotations

import json

import httpx
import pytest

from billie.llm import ImagePart, LLMError, LLMManager, LLMSettingsUpdate, Message, complete
from billie.llm.ollama import OllamaProvider
from billie.llm.openai_compat import OpenAICompatibleProvider
from billie.llm.think_filter import strip_think


# --------------------------------------------------------------- think filter
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Hello world"], "Hello world"),
        (["<think>secret</think>Answer"], "Answer"),
        (["<thi", "nk>reason", "ing</th", "ink>Final"], "Final"),
        (["A <", "b> tag"], "A <b> tag"),  # a lone '<' must not be swallowed
        (["pre<think>x</think>post"], "prepost"),
    ],
)
def test_strip_think_handles_split_tags(chunks, expected):
    assert "".join(strip_think(chunks)) == expected


# -------------------------------------------------------------------- ollama
def _ollama_transport(captured: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            captured["payload"] = json.loads(request.content)
            lines = [
                {"message": {"content": "Hel"}, "done": False},
                {"message": {"content": "lo"}, "done": False},
                {"message": {"content": ""}, "done": True},
            ]
            return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m1"}, {"name": "m0"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_ollama_streams_tokens_and_encodes_images():
    captured: dict = {}
    provider = OllamaProvider("m1", transport=_ollama_transport(captured))
    text = complete(provider, [Message("user", "hi", images=(ImagePart(b"\x89PNG"),))])
    assert text == "Hello"
    message = captured["payload"]["messages"][0]
    assert message["images"] == ["iVBORw=="]
    assert captured["payload"]["think"] is False
    assert provider.supports_vision()


def test_ollama_health_reports_missing_model():
    provider = OllamaProvider("not-pulled", transport=_ollama_transport({}))
    health = provider.health()
    assert not health.ok
    assert "ollama pull not-pulled" in health.detail
    assert health.models == ["m0", "m1"]


def test_ollama_connection_error_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    provider = OllamaProvider("m1", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        complete(provider, [Message("user", "hi")])


# ---------------------------------------------------------- openai-compatible
def test_openai_compatible_parses_sse_and_sends_bearer_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        body = (
            'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    provider = OpenAICompatibleProvider(
        name="groq",
        model="x",
        base_url="https://api.test/v1",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
    )
    assert complete(provider, [Message("user", "q", images=(ImagePart(b"ab", "image/jpeg"),))]) == "Hi there"
    assert seen["auth"] == "Bearer sk-test"
    parts = seen["payload"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_compatible_http_error_is_reported():
    provider = OpenAICompatibleProvider(
        name="openai",
        model="x",
        base_url="https://api.test/v1",
        api_key="bad",
        transport=httpx.MockTransport(lambda request: httpx.Response(401, text="invalid key")),
    )
    with pytest.raises(LLMError, match="401"):
        complete(provider, [Message("user", "q")])


# ------------------------------------------------------------------- manager
def test_manager_defaults_to_ollama_and_persists_updates(tmp_path):
    path = tmp_path / "llm.json"
    manager = LLMManager(path, env={})
    assert manager.settings.provider == "ollama"

    manager.update(LLMSettingsUpdate(model="llama3.2", temperature=0.5))
    reloaded = LLMManager(path, env={})
    assert reloaded.settings.model == "llama3.2"
    assert reloaded.settings.temperature == 0.5


def test_manager_requires_key_for_cloud_provider_and_masks_it(tmp_path):
    manager = LLMManager(tmp_path / "llm.json", env={})
    with pytest.raises(LLMError, match="needs an API key"):
        manager.update(LLMSettingsUpdate(provider="groq"))
    assert manager.settings.provider == "ollama"  # failed update leaves settings untouched

    public = manager.update(LLMSettingsUpdate(provider="groq", api_key="gsk_1234567890abcdef"))
    assert public["provider"] == "groq"
    assert public["model"] == "qwen/qwen3.8-27b"  # provider default applied
    assert public["keys"]["groq"] == "gsk_…cdef"
    assert "gsk_1234567890abcdef" not in json.dumps(public)


def test_manager_keeps_keys_per_provider_and_reads_env(tmp_path):
    manager = LLMManager(tmp_path / "llm.json", env={"OPENAI_API_KEY": "sk-from-env-123456"})
    manager.update(LLMSettingsUpdate(provider="groq", api_key="gsk_abcdefghijklmnop"))
    manager.update(LLMSettingsUpdate(provider="openai"))  # key comes from env
    manager.update(LLMSettingsUpdate(provider="groq"))  # stored key is remembered
    assert manager.public_settings()["keys"]["openai"] == "sk-f…3456"
    assert manager.provider().name == "groq"


def test_manager_env_selects_initial_provider(tmp_path):
    manager = LLMManager(tmp_path / "llm.json", env={"BILLIE_LLM_PROVIDER": "ollama", "BILLIE_LLM_MODEL": "qwen3:8b"})
    assert manager.settings.model == "qwen3:8b"


def test_manager_ignores_corrupt_settings_file(tmp_path):
    path = tmp_path / "llm.json"
    path.write_text("{not json", encoding="utf-8")
    assert LLMManager(path, env={}).settings.provider == "ollama"


def test_ollama_localhost_is_rewritten_to_ipv4():
    from billie.llm.ollama import normalize_base_url

    assert normalize_base_url("http://localhost:11434/") == "http://127.0.0.1:11434"
    assert normalize_base_url("http://ollama:11434") == "http://ollama:11434"  # docker service names untouched
