"""Visitors' own temporary documents on the public demo (P2.19)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from billie.api.app import create_app
from billie.api.services import build_services
from billie.private_docs import PrivateDocuments
from billie.uploads import UploadError
from tests.conftest import FakeLLMManager, FakeProvider

ALICE = {"X-Visitor-Id": "alice-0123456789abcdef"}
BOB = {"X-Visitor-Id": "bob-0123456789abcdef00"}
OWNER = "v_" + "a" * 32
OTHER = "v_" + "b" * 32
SECRET = b"# Project Falcon\n\nThe Falcon launch budget is 42 million rupees and the launch site is Sriharikota.\n"


def _client(settings, llm=None, **overrides):
    demo = replace(settings, demo=True, rate_limit_per_minute=0, **overrides)
    services = build_services(demo, llm=llm or FakeLLMManager())
    services.corpora.prepare()
    return TestClient(create_app(demo, services=services, warm=False)), services


def _upload(client, headers, name="falcon.md", data=SECRET):
    return client.post("/api/private-documents", files={"file": (name, data, "text/markdown")}, headers=headers)


def _ask(client, headers, question="What is the Falcon launch budget?"):
    return client.post("/api/chat?stream=false", json={"question": question, "mode": "general"}, headers=headers).json()


# ------------------------------------------------------------------ the store itself
def test_store_indexes_files_and_searches_them(tmp_path):
    store = PrivateDocuments(tmp_path / "private", None)
    docs = store.add(OWNER, "falcon.md", SECRET)
    assert docs[0].name == "falcon.md" and docs[0].passages >= 1
    retriever = store.retriever(OWNER)
    assert retriever is not None
    hit = retriever.search("Falcon budget", top_k=1).hits[0]
    assert hit.passage.source == "Your document · falcon.md" and hit.passage.metadata["private"] is True
    assert hit.passage.id.startswith("private:")
    assert store.retriever(OTHER) is None  # another visitor has nothing


def test_documents_expire_after_an_hour_without_use(tmp_path):
    now = [1000.0]
    store = PrivateDocuments(tmp_path / "private", None, clock=lambda: now[0])
    store.add(OWNER, "falcon.md", SECRET)
    now[0] += 1800
    assert store.retriever(OWNER) is not None  # asking counts as use and resets the timer
    now[0] += 3500
    assert store.sweep() == 0
    now[0] += 200  # 3,700 s since last use
    assert store.sweep() == 1
    assert store.retriever(OWNER) is None and not (tmp_path / "private" / OWNER).exists()


def test_store_is_wiped_on_start(tmp_path):
    PrivateDocuments(tmp_path / "private", None).add(OWNER, "falcon.md", SECRET)
    PrivateDocuments(tmp_path / "private", None)  # a restart
    assert not any((tmp_path / "private").iterdir())


def test_limits_and_bad_files_are_rejected_cleanly(tmp_path):
    store = PrivateDocuments(tmp_path / "private", None, max_files=2, max_passages=3, max_bytes=50_000)
    store.add(OWNER, "a.md", SECRET)
    store.add(OWNER, "b.md", SECRET.replace(b"Falcon", b"Hawk"))
    with pytest.raises(UploadError, match="up to 2 documents"):
        store.add(OWNER, "c.md", SECRET)
    store.add(OWNER, "a.md", SECRET)  # replacing an existing file is fine
    store.delete(OWNER, "b.md")
    long_text = ("Sentence number %d talks about rockets and budgets in some detail. " * 60) % tuple(range(60))
    with pytest.raises(UploadError, match="too long"):
        store.add(OWNER, "long.md", ("# Long\n\n" + long_text).encode())
    with pytest.raises(UploadError, match="not a valid PDF"):
        store.add(OWNER, "fake.pdf", b"not a pdf at all")
    with pytest.raises(UploadError, match=r"could not be read|No text"):
        store.add(OWNER, "broken.pdf", b"%PDF-1.4 broken")
    with pytest.raises(UploadError, match="Unsupported"):
        store.add(OWNER, "virus.exe", b"MZ")
    assert [d.name for d in store.documents(OWNER)] == ["a.md"]  # rejected files left nothing behind


def test_visitor_and_storage_caps(tmp_path):
    store = PrivateDocuments(tmp_path / "private", None, max_visitors=1)
    store.add(OWNER, "a.md", SECRET)
    with pytest.raises(UploadError, match="busy"):
        store.add(OTHER, "a.md", SECRET)
    full = PrivateDocuments(tmp_path / "p2", None, max_total_bytes=10)
    with pytest.raises(UploadError, match="storage is full"):
        full.add(OWNER, "a.md", SECRET)


def test_invalid_owner_is_refused(tmp_path):
    store = PrivateDocuments(tmp_path / "private", None)
    with pytest.raises(UploadError):
        store.add("../../etc", "a.md", SECRET)
    assert store.retriever("../../etc") is None


# ------------------------------------------------------------------ through the API
def test_only_the_uploader_can_see_and_search_their_document(settings):
    client, _ = _client(settings, FakeLLMManager(FakeProvider("The budget is 42 million rupees [1].")))
    with client:
        assert _upload(client, ALICE).status_code == 200
        assert [d["name"] for d in client.get("/api/private-documents", headers=ALICE).json()["documents"]] == [
            "falcon.md"
        ]
        assert client.get("/api/private-documents", headers=BOB).json()["documents"] == []

        mine = _ask(client, ALICE)
        assert mine["sources"][0]["source"] == "Your document · falcon.md" and mine["grounded"]
        theirs = _ask(client, BOB)
        assert all(not s["metadata"].get("private") for s in theirs["sources"])  # Bob never sees Alice's file
        assert client.delete("/api/private-documents/falcon.md", headers=BOB).status_code == 404


def test_answers_using_private_documents_are_never_cached(settings):
    client, services = _client(settings, FakeLLMManager(FakeProvider("The budget is 42 million rupees [1].")))
    with client:
        _upload(client, ALICE)
        first = _ask(client, ALICE)
        again = _ask(client, ALICE)
        assert not first["cached"] and not again["cached"] and first["cache_key"] is None
        # Bob asks the same question: he must not get Alice's answer from the shared cache.
        bob = _ask(client, BOB)
        assert not bob["cached"] and all(not s["metadata"].get("private") for s in bob["sources"])
        assert services.cache is not None and services.cache.stats()["entries"] <= 1  # only Bob's public answer


def test_shared_documents_still_answer_alongside_private_ones(settings):
    client, _ = _client(settings)
    with client:
        _upload(client, ALICE)
        answer = _ask(client, ALICE, "How is the knowledge index refreshed?")
        assert any(not s["metadata"].get("private") for s in answer["sources"])  # Billie's docs still found


def test_local_mode_has_no_private_documents(settings, fake_llm):
    services = build_services(settings, llm=fake_llm)
    services.corpora.prepare()
    with TestClient(create_app(settings, services=services, warm=False)) as client:
        assert services.private is None
        assert client.get("/api/private-documents").status_code == 404


def test_upload_errors_are_clean_400s(settings):
    client, _ = _client(settings)
    with client:
        bad = client.post(
            "/api/private-documents", files={"file": ("x.pdf", b"nope", "application/pdf")}, headers=ALICE
        )
        assert bad.status_code == 400 and "valid PDF" in bad.json()["detail"]
        assert client.get("/api/private-documents", headers=ALICE).json()["limits"] == {
            "max_files": 3,
            "max_mb": 5,
            "ttl_minutes": 60,
        }


class CountingEmbedder:
    """Fake embedder that records how many passages it embedded."""

    model = "fake-embed"

    def __init__(self):
        from .test_index import FakeEmbedder

        self.inner = FakeEmbedder()
        self.embedded = 0

    def embed_documents(self, texts, titles=None):
        self.embedded += len(texts)
        return self.inner.embed_documents(texts, titles)

    def embed_queries(self, texts):
        return self.inner.embed_queries(texts)


def test_each_file_is_embedded_once_and_meaning_search_works(tmp_path):
    embedder = CountingEmbedder()
    store = PrivateDocuments(tmp_path / "private", embedder)
    store.add(OWNER, "eyes.md", b"# Eyes\n\nGlaucoma damages the optic nerve and can cause blindness.\n")
    first = embedder.embedded
    store.add(OWNER, "lungs.md", b"# Lungs\n\nAsthma inhalers relieve wheezing and shortness of breath.\n")
    assert first >= 1 and embedder.embedded - first == 1  # only the new file was embedded
    hits = store.retriever(OWNER).search("vision loss", top_k=1).hits  # no shared words: needs vectors
    assert hits and hits[0].passage.metadata["file"] == "eyes.md"
    store.delete(OWNER, "eyes.md")
    assert not (tmp_path / "private" / OWNER / "files" / "eyes.md.vectors.npz").exists()
    assert embedder.embedded - first == 1  # removing a file re-embeds nothing
