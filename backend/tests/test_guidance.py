"""Coverage overview, as-you-type suggestions and dead-end guidance (P2.18)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from billie.api.app import create_app
from billie.api.services import build_services
from billie.chat.pipeline import ChatPipeline, ChatTurn
from billie.corpora import overview as ov
from tests.conftest import FakeLLMManager, FakeProvider


def _services(settings, llm=None):
    s = build_services(settings, llm=llm or FakeLLMManager())
    s.corpora.prepare()
    return s


def test_overview_reports_coverage_examples_and_limits(settings):
    ov._cache.clear()
    modes = ov.overview(_services(settings).corpora)["modes"]
    assert set(modes) == {"general", "medical", "research"}
    general = modes["general"]
    assert {d["name"] for d in general["documents"]} == {"billie.md", "security.md"}
    assert modes["medical"]["questions"] > 0 and modes["research"]["categories"]
    for mode in modes.values():
        assert mode["examples"] and mode["not_covered"] and mode["good_for"]


def test_overview_without_compute_serves_basic_counts_until_stats_exist(settings):
    ov._cache.clear()
    corpora = _services(settings).corpora
    for path in corpora.index_dir.glob("*.overview.json"):
        path.unlink()
    quick = ov.overview(corpora, compute=False)["modes"]["general"]
    assert quick["passages"] > 0 and "documents" not in quick
    ov.overview(corpora)  # computes and saves to <index>.overview.json
    ov._cache.clear()
    assert "documents" in ov.overview(corpora, compute=False)["modes"]["general"]  # reloaded from disk


def test_suggestions_come_from_real_titles_with_prefix_matching(settings):
    corpora = _services(settings).corpora
    hits = ov.suggest(corpora, "general", "secur")
    assert hits and hits[0]["title"] == "Security"
    assert hits[0]["question"] == "What does the documentation say about Security?"
    assert ov.suggest(corpora, "general", "s") == []  # too short
    assert ov.suggest(corpora, "general", "zzzqqq") == []


def test_clean_title_rewrites_medquad_quirks():
    assert ov.clean_title("medquad", "What is (are) Glaucoma ?") == "What is Glaucoma?"
    assert ov.clean_title("medquad", "What are the treatments for Glaucoma ?") == "How is Glaucoma treated?"
    assert ov.clean_title("medquad", "How to diagnose Glaucoma ?") == "How to diagnose Glaucoma?"
    assert ov.clean_title("medquad", "Who is at risk for Glaucoma? ?") == "Who is at risk for Glaucoma?"
    assert ov.clean_title("knowledge", "Billie › Refresh") == "Refresh"
    assert ov.question_for("arxiv", "Attention Is All You Need").startswith('What does the paper "Attention')


def test_abstained_answer_offers_related_topics(settings):
    s = _services(settings, FakeLLMManager(FakeProvider("I could not find this in the documents.")))
    events = list(s.pipeline.run(ChatTurn("Where should phishing emails be reported?")))
    done = next(e["data"] for e in events if e["event"] == "done")
    assert done["abstained"] and done["related"]
    assert all({"title", "question"} <= set(r) for r in done["related"])
    assert done["suggested_mode"] is None  # no embedding model in tests


def test_grounded_answer_has_no_guidance(settings):
    s = _services(settings)
    done = next(e["data"] for e in s.pipeline.run(ChatTurn("How is the index refreshed?")) if e["event"] == "done")
    assert done["grounded"] and done["related"] == [] and done["suggested_mode"] is None


class _Index:
    def __init__(self, score: float):
        self.score = score
        self.embedding_model = "fake"

    def has_vectors(self) -> bool:
        return True

    def search_dense(self, vector, k):
        return [(1, self.score)]


class _Corpora:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.embedder = type("E", (), {"model": "fake", "embed_queries": lambda self, q: np.ones((1, 3))})()

    def index(self, name):
        return _Index(self.scores[name])


def _better(mode: str, scores: dict[str, float]):
    pipeline = ChatPipeline.__new__(ChatPipeline)
    pipeline.corpora = _Corpora(scores)  # type: ignore[assignment]
    return pipeline._better_mode(mode, "q")


def test_better_mode_needs_a_clear_winner():
    # Calibrated values: a medical question asked in Research mode (glaucoma: 0.73 vs 0.53).
    assert _better("research", {"knowledge": 0.09, "medquad": 0.73, "arxiv": 0.53}) == {
        "mode": "medical",
        "score": 0.73,
    }
    # Off-topic everywhere (movie recommendation): no suggestion.
    assert _better("medical", {"knowledge": 0.21, "medquad": 0.22, "arxiv": 0.36}) is None
    # Another mode slightly better but not clearly: no suggestion.
    assert _better("general", {"knowledge": 0.46, "medquad": 0.30, "arxiv": 0.50}) is None


def test_overview_and_suggest_endpoints(settings, fake_llm):
    services = _services(settings, fake_llm)
    with TestClient(create_app(settings, services=services, warm=False)) as client:
        assert client.get("/api/overview").json()["modes"]["general"]["examples"]
        assert client.get("/api/suggest", params={"mode": "general", "q": "secur"}).json()[0]["title"] == "Security"
        assert client.get("/api/suggest", params={"mode": "nope", "q": "x"}).status_code == 422
