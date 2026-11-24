from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from billie.api.app import create_app
from billie.api.services import build_services
from billie.runtime import load_settings

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def make_client(settings, fake_llm) -> TestClient:
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    return TestClient(create_app(settings, services=services, warm=False))


@pytest.fixture()
def client(settings, fake_llm):
    with make_client(settings, fake_llm) as c:
        yield c


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_and_status(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    status = client.get("/api/status").json()
    assert status["llm"]["ok"] is True
    assert {c["name"] for c in status["corpora"]} == {"knowledge", "medquad", "arxiv"}
    assert status["embedding"]["model"] is None  # disabled in tests -> BM25-only


def test_chat_streams_server_sent_events(client):
    response = client.post("/api/chat", json={"question": "Where are phishing emails reported?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "status" and "meta" in names and names[-1] == "done"
    done = events[-1][1]
    assert done["grounded"] is True
    convo = client.get(f"/api/conversations/{done['conversation_id']}").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]


def test_chat_json_mode_and_validation(client):
    body = client.post("/api/chat?stream=false", json={"question": "How is the index refreshed?"}).json()
    assert body["answer"] and body["sources"]
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "x", "mode": "astrology"}).status_code == 422
    assert client.post("/api/chat", json={"question": "x", "conversation_id": "../x"}).status_code == 422


def test_medical_and_research_modes_use_their_corpora(client):
    medical = client.post("/api/chat?stream=false", json={"question": "asthma symptoms", "mode": "medical"}).json()
    assert medical["sources"][0]["source"].startswith("MedQuAD")
    assert "disclaimer" in medical["enrichment"]
    research = client.post("/api/chat?stream=false", json={"question": "retrieval QA", "mode": "research"}).json()
    assert research["sources"][0]["source"] == "arXiv"
    assert research["enrichment"]["concepts"]


def test_image_upload_validates_content_not_name(client):
    ok = client.post("/api/uploads/images", files={"file": ("x.png", PNG, "image/png")})
    assert ok.status_code == 200 and ok.json()["media_type"] == "image/png"
    fake = client.post("/api/uploads/images", files={"file": ("evil.png", b"<?php echo 1;", "image/png")})
    assert fake.status_code == 400
    image_id = ok.json()["id"]
    assert client.get(f"/api/uploads/images/{image_id}").content == PNG
    assert client.get("/api/uploads/images/..%2F..%2Fsessions.sqlite").status_code == 404


def test_chat_rejects_unknown_image_ids_before_streaming(client):
    response = client.post("/api/chat", json={"question": "what is this", "image_ids": ["0" * 32 + ".png"]})
    assert response.status_code == 400


def test_uploaded_document_becomes_searchable_and_can_be_deleted(client):
    doc = b"# Holidays\n\nThe office is closed on the first Monday of every month for maintenance."
    up = client.post("/api/documents", files={"file": ("../../holidays.md", doc, "text/markdown")})
    assert up.status_code == 200 and up.json()["name"] == "holidays.md"  # directory components stripped
    answer = client.post("/api/chat?stream=false", json={"question": "When is the office closed for maintenance?"})
    assert answer.json()["sources"][0]["source"] == "holidays.md"
    assert client.delete("/api/documents/holidays.md").json() == {"ok": True}
    assert client.get("/api/documents").json() == []
    bad = client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert bad.status_code == 400


def test_conversation_rename_and_delete(client):
    done = client.post("/api/chat?stream=false", json={"question": "How is the index refreshed?"}).json()
    cid = done["conversation_id"]
    assert client.patch(f"/api/conversations/{cid}", json={"title": "Refresh"}).json() == {"ok": True}
    assert client.get("/api/conversations").json()[0]["title"] == "Refresh"
    assert client.delete(f"/api/conversations/{cid}").json() == {"ok": True}
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_keys_are_enforced_when_configured(settings, fake_llm):
    secured = replace(settings, api_key="chat-key", admin_key="admin-key")
    with make_client(secured, fake_llm) as c:
        assert c.post("/api/chat", json={"question": "hi"}).status_code == 401
        assert (
            c.post("/api/chat?stream=false", json={"question": "hi"}, headers={"X-API-Key": "chat-key"}).status_code
            == 200
        )
        # Admin endpoints need the admin key, not the chat key.
        doc = {"file": ("a.txt", b"hello", "text/plain")}
        assert c.post("/api/documents", files=doc, headers={"X-API-Key": "chat-key"}).status_code == 401
        assert c.post("/api/documents", files=doc, headers={"X-Admin-Key": "admin-key"}).status_code == 200
        assert c.get("/api/health").status_code == 200  # liveness stays public


def test_production_refuses_to_start_without_keys(tmp_path):
    with pytest.raises(RuntimeError, match="requires BILLIE_ADMIN_KEY"):
        load_settings({"BILLIE_ENV": "production"}, data_dir=tmp_path)
    ok = load_settings({"BILLIE_ENV": "production", "BILLIE_API_KEY": "a", "BILLIE_ADMIN_KEY": "b"}, data_dir=tmp_path)
    assert ok.is_production


def test_metrics_endpoint_counts_chat_turns(client):
    client.post("/api/chat?stream=false", json={"question": "How is the index refreshed?"})
    text = client.get("/metrics").text
    assert 'billie_chat_turns_total{language="en",mode="general"} 1.0' in text


def test_spa_fallback_never_masks_api_routes(client):
    # With a built frontend, unknown non-API paths serve index.html, but /api/* must stay a 404.
    assert client.get("/api/does-not-exist").status_code == 404


def test_request_id_is_returned_and_logged(client, caplog):
    import logging

    logging.getLogger("billie").propagate = True  # let pytest's caplog see the JSON logger
    with caplog.at_level(logging.INFO, logger="billie"):
        response = client.post(
            "/api/chat?stream=false",
            json={"question": "How is the index refreshed?"},
            headers={"X-Request-ID": "abc-123"},
        )
    assert response.headers["x-request-id"] == "abc-123"
    turn = next(r for r in caplog.records if r.getMessage() == "chat_turn")
    assert turn.fields["request_id"] == "abc-123" and turn.fields["mode"] == "general"
    assert "question" not in turn.fields  # user text is not logged
    bad = client.get("/api/health", headers={"X-Request-ID": "x" * 200})
    assert len(bad.headers["x-request-id"]) == 16  # invalid incoming IDs are replaced


def test_closing_the_stream_closes_the_model_connection(settings):
    """Simulates the Stop button: the provider's stream must be closed, so Ollama stops generating."""
    import asyncio

    from billie.api.app import _sse, stream_and_close
    from billie.chat.pipeline import ChatTurn
    from tests.conftest import FakeLLMManager, FakeProvider

    state = {"closed": False, "tokens": 0}

    class EndlessProvider(FakeProvider):
        def stream(self, messages, options=None):
            try:
                while True:
                    state["tokens"] += 1
                    yield f"token{state['tokens']} "  # never repeats, so the loop guard can't be what stops it
            finally:
                state["closed"] = True  # httpx stream context exit happens here in the real provider

    services = build_services(settings, llm=FakeLLMManager(EndlessProvider()))
    services.corpora.prepare()

    async def consume_then_disconnect():
        agen = stream_and_close(_sse(services.pipeline.run(ChatTurn("How is the index refreshed?"))))
        received = 0
        async for chunk in agen:
            if chunk.startswith("event: token"):
                received += 1
                if received == 3:
                    break
        await agen.aclose()  # what Starlette does when the client goes away

    asyncio.run(consume_then_disconnect())
    assert state["closed"] is True
    assert state["tokens"] < 10  # generation stopped promptly instead of running on


def test_no_base_http_middleware_so_client_disconnects_reach_the_stream(settings, fake_llm):
    """BaseHTTPMiddleware (@app.middleware("http")) swallows http.disconnect for streamed responses,
    so the Stop button would no longer stop the local model. Only pure ASGI middleware is allowed."""
    from starlette.middleware.base import BaseHTTPMiddleware

    app = create_app(settings, services=build_services(settings, llm=fake_llm), warm=False)
    offenders = [m.cls.__name__ for m in app.user_middleware if issubclass(m.cls, BaseHTTPMiddleware)]
    assert offenders == []
