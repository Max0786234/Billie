from __future__ import annotations

from billie.api.services import build_services
from billie.chat.cache import AnswerCache, cache_key
from billie.chat.pipeline import ChatTurn
from tests.conftest import FakeLLMManager, FakeProvider

Q = "Where should phishing emails be reported?"


def run(services, question, **kwargs):
    events = list(services.pipeline.run(ChatTurn(question, **kwargs)))
    return {e["event"]: e["data"] for e in events}


def make(settings, provider=None):
    llm = FakeLLMManager(provider or FakeProvider())
    s = build_services(settings, llm=llm)
    s.corpora.prepare()
    return s, llm


def test_repeat_question_is_served_from_cache_without_calling_the_model(settings):
    s, llm = make(settings)
    first = run(s, Q)
    second = run(s, "  where should PHISHING emails be reported ")  # normalised to the same key
    assert len(llm.chat.calls) == 1
    assert first["done"]["cached"] is False and second["done"]["cached"] is True
    assert second["meta"]["cached"] is True and second["done"]["answer"] == first["done"]["answer"]
    assert second["done"]["conversation_id"] != first["done"]["conversation_id"]  # new chat, own history
    _, messages = s.conversations.get(second["done"]["conversation_id"])
    assert [m.role for m in messages] == ["user", "assistant"] and messages[1].sources


def test_follow_ups_are_never_cached(settings):
    s, llm = make(settings)
    cid = run(s, Q)["done"]["conversation_id"]
    run(s, Q, conversation_id=cid)  # same text, but now it has history
    assert len(llm.chat.calls) == 2


def test_ungrounded_answers_are_not_cached(settings):
    s, llm = make(settings, FakeProvider("Made up [9]."))
    run(s, Q)
    run(s, Q)
    assert len(llm.chat.calls) == 2


def test_key_changes_with_model_and_corpus():
    base = cache_key("medical", "What is diabetes?", "ollama:a", "stamp1")
    assert base == cache_key("medical", "what is diabetes", "ollama:a", "stamp1")
    assert base != cache_key("medical", "What is diabetes?", "ollama:b", "stamp1")
    assert base != cache_key("medical", "What is diabetes?", "ollama:a", "stamp2")
    assert base != cache_key("research", "What is diabetes?", "ollama:a", "stamp1")


def test_cache_is_lru_bounded(tmp_path):
    cache = AnswerCache(tmp_path / "c.sqlite", max_entries=2)
    for i in range(3):
        cache.put(f"k{i}", {"sources": []}, {"answer": str(i)})
    assert cache.get("k0") is None and cache.get("k2") is not None
    assert cache.stats()["entries"] == 2
