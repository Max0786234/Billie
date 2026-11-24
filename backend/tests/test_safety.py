from __future__ import annotations

import pytest

from billie.api.services import build_services
from billie.chat.pipeline import ChatTurn
from billie.nlp.safety import detect_emergency


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I have crushing chest pain and my left arm hurts", "cardiac"),
        ("my father can't breathe properly", "breathing"),
        ("her face is drooping and she has slurred speech", "stroke"),
        ("I want to kill myself", "self_harm"),
        ("i dont want to live, thinking about suicide", "self_harm"),
        ("my son took too many pills", "overdose"),
        ("he passed out and is unresponsive", "unconscious"),
        ("my throat is closing after eating peanuts", "anaphylaxis"),
        ("mujhe seene mein dard ho raha hai", "cardiac"),
        ("main marna chahta hoon", "self_harm"),
        ("सीने में दर्द हो रहा है", "cardiac"),
        ("मुझे आत्महत्या के विचार आ रहे हैं", "self_harm"),
        ("tengo dolor de pecho", "cardiac"),
        # Found by the stress test (missed before):
        ("I don't want to live anymore, I'm thinking about ending my life", "self_harm"),
        ("sometimes I feel everyone would be better off dead without me", "self_harm"),
        ("my child swallowed a whole bottle of pills", "overdose"),
        ("she took a handful of sleeping tablets", "overdose"),
        ("no puedo respirar", "breathing"),
    ],
)
def test_emergencies_are_detected_across_languages(text, category):
    alert = detect_emergency(text)
    assert alert.triggered and category in alert.categories


@pytest.mark.parametrize(
    "text",
    [
        "What is diabetes?",
        "How is glaucoma treated?",
        "Explain transformers attention",
        "my knee hurts after running",
        "I took my pills this morning",
        "how many pills are in a pack of ibuprofen",
    ],
)
def test_ordinary_questions_do_not_trigger(text):
    assert not detect_emergency(text).triggered


def test_self_harm_message_points_to_helplines():
    alert = detect_emergency("I want to end my life")
    assert "14416" in alert.message and "988" in alert.message


def test_pipeline_emits_alert_and_instructs_the_model(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    events = list(services.pipeline.run(ChatTurn("I have chest pain and can't breathe", "general")))
    meta = next(e["data"] for e in events if e["event"] == "meta")
    done = next(e["data"] for e in events if e["event"] == "done")
    assert set(meta["alert"]["categories"]) >= {"cardiac", "breathing"}
    assert done["alert"] == meta["alert"]  # persisted with the answer
    assert fake_llm.chat.calls[-1][-1].content.count("SAFETY:") == 1


def test_no_alert_for_normal_turn(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    events = list(services.pipeline.run(ChatTurn("How is the index refreshed?", "general")))
    assert next(e["data"] for e in events if e["event"] == "meta")["alert"] is None
    assert "SAFETY:" not in fake_llm.chat.calls[-1][-1].content
