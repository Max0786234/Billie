from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from billie.api.app import create_app
from billie.api.services import build_services
from billie.chat.pipeline import ChatTurn
from billie.conversations import ConversationStore

Q = "Where should phishing emails be reported?"


def client_for(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    return TestClient(create_app(settings, services=services, warm=False)), services


def test_thumbs_down_records_feedback_and_evicts_cached_answer(settings, fake_llm):
    client, services = client_for(settings, fake_llm)
    with client:
        done = client.post("/api/chat?stream=false", json={"question": Q}).json()
        assert services.cache.stats()["entries"] == 1
        assert client.put(f"/api/messages/{done['message_id']}/feedback", json={"value": -1}).json() == {"ok": True}
        assert services.cache.stats()["entries"] == 0  # a disliked answer is never replayed
        status = client.get("/api/status").json()
        assert status["feedback"]["general"] == {"up": 0, "down": 1}
        convo = client.get(f"/api/conversations/{done['conversation_id']}").json()
        assert convo["messages"][1]["feedback"] == -1
        assert client.put("/api/messages/999999/feedback", json={"value": 1}).status_code == 404
        assert client.put(f"/api/messages/{done['message_id']}/feedback", json={"value": 5}).status_code == 422


def test_regenerate_replaces_the_last_turn_and_skips_the_cache(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    first = next(e for e in services.pipeline.run(ChatTurn(Q)) if e["event"] == "done")["data"]
    cid = first["conversation_id"]
    regenerated = next(
        e for e in services.pipeline.run(ChatTurn(Q, conversation_id=cid, regenerate=True)) if e["event"] == "done"
    )["data"]
    assert regenerated["cached"] is False and len(fake_llm.chat.calls) == 2
    _, messages = services.conversations.get(cid)
    assert [m.role for m in messages] == ["user", "assistant"]  # replaced, not appended


def test_edit_question_regenerates_with_new_text(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    cid = next(e for e in services.pipeline.run(ChatTurn(Q)) if e["event"] == "done")["data"]["conversation_id"]
    list(services.pipeline.run(ChatTurn("How is the index refreshed?", conversation_id=cid, regenerate=True)))
    _, messages = services.conversations.get(cid)
    assert messages[0].content == "How is the index refreshed?" and len(messages) == 2


def test_old_database_is_migrated_with_feedback_column(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, mode TEXT NOT NULL, "
        "created_at REAL NOT NULL, updated_at REAL NOT NULL);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, "
        "role TEXT NOT NULL, content TEXT NOT NULL, mode TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', "
        "meta TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL);"
    )
    conn.close()
    store = ConversationStore(db)
    cid = store.create("general")
    store.append(cid, "user", "q", "general")
    mid = store.append(cid, "assistant", "a", "general")
    store.set_feedback(mid, 1)
    assert store.get(cid)[1][1].feedback == 1
