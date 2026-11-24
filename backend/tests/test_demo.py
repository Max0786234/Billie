"""Public demo mode: private conversations per visitor, daily quotas, framing, deploy bundle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from billie.api.app import create_app
from billie.api.demo import DailyQuota
from billie.api.services import build_services
from billie.llm.manager import LLMManager
from billie.retrieval import HybridRetriever, PassageIndex, embed_index
from billie.retrieval.index import subset_index
from billie.runtime import load_settings

from .test_index import FakeEmbedder

ALICE = {"X-Visitor-Id": "alice-0123456789abcdef"}
BOB = {"X-Visitor-Id": "bob-0123456789abcdef00"}


def _demo_client(settings, fake_llm, **overrides):
    demo = replace(settings, demo=True, rate_limit_per_minute=0, **overrides)
    services = build_services(demo, llm=fake_llm)
    services.corpora.prepare()
    return TestClient(create_app(demo, services=services, warm=False))


def _ask(client, headers, question="How is the index refreshed?", **extra):
    return client.post("/api/chat?stream=false", json={"question": question, **extra}, headers=headers)


def test_quota_counts_per_client_and_in_total_and_resets_daily():
    now = [0.0]
    quota = DailyQuota(total=3, per_client=2, clock=lambda: now[0])
    assert quota.take("a") is None and quota.take("a") is None
    assert "2 demo questions" in (quota.take("a") or "")  # personal cap
    assert quota.take("b") is None
    assert "daily limit" in (quota.take("c") or "")  # global cap reached
    quota.refund("b")
    assert quota.take("c") is None
    now[0] += 86400  # next UTC day
    assert quota.take("a") is None and quota.stats("a")["remaining_for_you"] == 1


def test_visitors_only_see_their_own_conversations(settings, fake_llm):
    with _demo_client(settings, fake_llm) as client:
        alice_chat = _ask(client, ALICE).json()["conversation_id"]
        bob_chat = _ask(client, BOB).json()["conversation_id"]
        assert [c["id"] for c in client.get("/api/conversations", headers=ALICE).json()] == [alice_chat]
        # Someone else's chat behaves as if it does not exist - for reading, renaming and deleting.
        assert client.get(f"/api/conversations/{bob_chat}", headers=ALICE).status_code == 404
        assert client.patch(f"/api/conversations/{bob_chat}", json={"title": "x"}, headers=ALICE).status_code == 404
        assert client.delete(f"/api/conversations/{bob_chat}", headers=ALICE).status_code == 404
        # Continuing Bob's conversation id as Alice starts a new chat instead of appending to his.
        hijack = _ask(client, ALICE, conversation_id=bob_chat).json()["conversation_id"]
        assert hijack != bob_chat
        assert client.get(f"/api/conversations/{bob_chat}", headers=BOB).json()["message_count"] == 2


def test_feedback_is_scoped_to_the_owner(settings, fake_llm):
    with _demo_client(settings, fake_llm) as client:
        message_id = _ask(client, ALICE).json()["message_id"]
        assert client.put(f"/api/messages/{message_id}/feedback", json={"value": 1}, headers=BOB).status_code == 404
        assert client.put(f"/api/messages/{message_id}/feedback", json={"value": 1}, headers=ALICE).status_code == 200


def test_daily_cap_refuses_with_a_friendly_429_and_cached_answers_are_free(settings, fake_llm):
    with _demo_client(settings, fake_llm, demo_visitor_daily_limit=2) as client:
        assert _ask(client, ALICE).status_code == 200
        assert _ask(client, ALICE).json()["cached"]  # same question again: served from cache, refunded
        assert _ask(client, ALICE, "What does the security policy say?").status_code == 200
        refused = _ask(client, ALICE, "Something new?")
        assert refused.status_code == 429 and "come back tomorrow" in refused.json()["detail"]
        assert client.get("/api/status").json()["demo"]["remaining_for_you"] == 0


def test_demo_keeps_settings_and_documents_admin_only(settings, fake_llm):
    with _demo_client(settings, fake_llm, admin_key="admin") as client:
        assert client.put("/api/settings/llm", json={"temperature": 0.5}).status_code == 401
        doc = {"file": ("a.txt", b"hello", "text/plain")}
        assert client.post("/api/documents", files=doc).status_code == 401


def test_local_mode_is_unscoped(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    client = TestClient(create_app(settings, services=services, warm=False))
    first = client.post("/api/chat?stream=false", json={"question": "hi"}, headers=ALICE).json()
    listed = client.get("/api/conversations", headers=BOB).json()
    assert first["conversation_id"] in [c["id"] for c in listed]
    assert client.get("/api/status").json()["demo"] is None


def test_frame_ancestors_allow_the_space_page_to_embed_the_app(settings, fake_llm):
    with _demo_client(settings, fake_llm, frame_ancestors=("https://huggingface.co",)) as client:
        headers = client.get("/api/health").headers
        assert "frame-ancestors 'self' https://huggingface.co" in headers["content-security-policy"]
        assert "x-frame-options" not in headers


def test_production_demo_needs_only_the_admin_key(tmp_path):
    with pytest.raises(RuntimeError, match="BILLIE_ADMIN_KEY"):
        load_settings({"BILLIE_ENV": "production", "BILLIE_DEMO": "1"}, data_dir=tmp_path)
    demo = load_settings({"BILLIE_ENV": "production", "BILLIE_DEMO": "1", "BILLIE_ADMIN_KEY": "a"}, data_dir=tmp_path)
    assert demo.demo and demo.api_key is None


def test_llm_defaults_for_a_deployment_without_local_models(tmp_path):
    env = {
        "BILLIE_LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "gsk_test",
        "BILLIE_LOCAL_FALLBACK": "0",
        "BILLIE_VISION_MODEL": "",
        "BILLIE_VISION_CLOUD": "gemini",
    }
    settings = LLMManager(tmp_path / "llm.json", env=env).settings
    assert settings.provider == "groq" and not settings.local_fallback
    assert settings.vision_cloud == "gemini" and settings.vision_model == ""


def test_subset_index_keeps_only_embedded_passages_and_search_still_works(tmp_path):
    from billie.retrieval import Passage, build_index

    from .test_index import DOCS

    src, dst = tmp_path / "src", tmp_path / "dst"
    build_index(src, "t", [Passage(id=p, doc_id=p, title=t, text=x, source="test") for p, t, x in DOCS])
    embed_index(src, "t", FakeEmbedder(), where="title != 'Asthma'")
    assert subset_index(src, "t", dst) == 3
    index = PassageIndex(dst, "t")
    info = index.info()
    assert info.passages == info.vectors == 3 and info.meta["embedding_model"] == "fake-embed"
    titles = {p.title for p in index.get(np.load(dst / "t.vec_ids.npy").tolist()).values()}
    assert titles == {"Diabetes", "Glaucoma", "Exercise"}
    hits = HybridRetriever(index, FakeEmbedder()).search("glucose", top_k=1).hits  # dense-only match
    assert hits[0].passage.title == "Diabetes"
