from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from billie.llm.base import LLMError
from billie.retrieval import HybridRetriever, Passage, PassageIndex, build_index, embed_index, reciprocal_rank_fusion
from billie.retrieval.chunking import chunk_text
from billie.retrieval.embeddings import normalize
from billie.retrieval.index import fts_query

DOCS = [
    ("d1", "Diabetes", "Type 2 diabetes is managed with diet, exercise and metformin."),
    ("d2", "Glaucoma", "Glaucoma damages the optic nerve and can cause blindness."),
    ("d3", "Asthma", "Asthma inhalers relieve wheezing and shortness of breath."),
    ("d4", "Exercise", "Regular exercise lowers blood sugar and improves heart health."),
]


class FakeEmbedder:
    """Deterministic bag-of-concepts embedder: lets tests reason about cosine exactly."""

    model = "fake-embed"
    VOCAB: ClassVar[list[str]] = ["sugar", "eye", "lung", "fitness"]
    SYNONYMS: ClassVar[dict[str, set[str]]] = {
        "sugar": {"diabetes", "sugar", "metformin", "glucose"},
        "eye": {"glaucoma", "optic", "eye", "vision", "blindness"},
        "lung": {"asthma", "breath", "wheezing", "lung"},
        "fitness": {"exercise", "workout", "fitness"},
    }

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        words = set(text.lower().replace(",", " ").replace(".", " ").split())
        return [float(len(words & self.SYNONYMS[c])) for c in self.VOCAB]

    def embed_queries(self, texts):
        if self.fail:
            raise LLMError("down")
        return normalize(np.array([self._vec(t) for t in texts], dtype=np.float32))

    def embed_documents(self, texts, titles=None):
        self.calls += 1
        return normalize(np.array([self._vec(t) for t in texts], dtype=np.float32))


@pytest.fixture()
def index(tmp_path):
    passages = [Passage(id=pid, doc_id=pid, title=title, text=text, source="test") for pid, title, text in DOCS]
    build_index(tmp_path, "t", passages, meta={"corpus": "test"})
    return PassageIndex(tmp_path, "t")


def test_rrf_matches_hand_computed_scores():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1]}, k=60)
    scores = {doc: score for doc, score, _ in fused}
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [doc for doc, _, _ in fused] == [1, 3, 2]
    assert fused[0][2] == {"a": 1, "b": 2}


def test_rrf_breaks_ties_by_id():
    fused = reciprocal_rank_fusion({"a": [5], "b": [4]})
    assert [doc for doc, _, _ in fused] == [4, 5]


def test_fts_query_quotes_terms_and_drops_stopwords():
    assert fts_query('What is "glaucoma" AND (eye)?') == '"glaucoma" OR "eye"'
    assert fts_query("what is the") == ""


def test_bm25_ranks_matching_passage_first_and_stems(index):
    result = HybridRetriever(index, mode="bm25").search("blind optic nerves", top_k=2)
    assert result.mode == "bm25"
    assert result.hits[0].passage.id == "d2"  # porter stemming: nerves->nerv, blind->blind
    assert index.info().meta["corpus"] == "test"


def test_hybrid_finds_semantic_match_without_shared_words(index, tmp_path):
    embed_index(tmp_path, "t", FakeEmbedder())
    retriever = HybridRetriever(index, FakeEmbedder(), mode="hybrid", min_dense_score=0.5)
    result = retriever.search("glucose", top_k=3)  # no document contains the word "glucose"
    assert result.mode == "hybrid" and not result.degraded
    assert result.hits[0].passage.id == "d1"
    assert result.hits[0].components == {"dense": 1}


def test_hybrid_degrades_to_bm25_when_embedder_fails(index, tmp_path):
    embed_index(tmp_path, "t", FakeEmbedder())
    result = HybridRetriever(index, FakeEmbedder(fail=True)).search("asthma inhalers")
    assert result.degraded and result.mode == "bm25"
    assert "embedding unavailable" in result.note
    assert result.hits[0].passage.id == "d3"


def test_hybrid_refuses_vectors_from_a_different_embedding_model(index, tmp_path):
    embed_index(tmp_path, "t", FakeEmbedder())
    other = FakeEmbedder()
    other.model = "other-model"
    result = HybridRetriever(index, other).search("asthma")
    assert result.degraded and "embedded with 'fake-embed'" in result.note


def test_embedding_resumes_from_checkpoint_parts(index, tmp_path):
    embedder = FakeEmbedder()
    parts = tmp_path / "t.vec.parts"
    parts.mkdir()
    # Simulate an interrupted run that finished part 0 (rowids 1-2).
    np.savez(parts / "part-00000.npz", ids=np.array([1, 2]), vectors=np.eye(2, 4, dtype=np.float32))
    written = embed_index(tmp_path, "t", embedder, part_size=2)
    assert written == 4
    assert embedder.calls == 1  # only the remaining part was embedded
    assert not parts.exists()
    assert index.has_vectors()


def test_rebuild_invalidates_old_vectors(index, tmp_path):
    embed_index(tmp_path, "t", FakeEmbedder())
    assert index.has_vectors()
    index.close()
    build_index(tmp_path, "t", [Passage(id="x", doc_id="x", title="X", text="new", source="s")])
    assert not index.has_vectors()
    assert index.info().passages == 1


def test_chunk_text_packs_sentences_with_overlap():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_text(text, max_chars=32, overlap_chars=16)
    assert chunks[0] == "One two three. Four five six."
    assert chunks[1].startswith("Four five six.")  # overlap carried forward
    assert all(len(c) <= 32 for c in chunks)
    assert chunk_text("") == []
    long_word_run = "word " * 50
    assert all(len(c) <= 40 for c in chunk_text(long_word_run, max_chars=40))


def test_search_drops_duplicate_chunks_and_copied_passages(tmp_path):
    text = "Diabetes is a complex group of diseases with a variety of causes."
    passages = [
        Passage(id="a#0", doc_id="a", title="Causes of Diabetes", text=text, source="s"),
        Passage(
            id="a#1", doc_id="a", title="Causes of Diabetes", text="Second chunk about diabetes causes.", source="s"
        ),
        Passage(id="b#0", doc_id="b", title="Causes of Diabetes", text=text, source="s"),  # verbatim copy
        Passage(id="c#0", doc_id="c", title="Treating diabetes", text="Diabetes treatment uses insulin.", source="s"),
        Passage(id="k1", doc_id="guide.md", title="Guide › Install", text="Install diabetes app.", source="s"),
        Passage(id="k2", doc_id="guide.md", title="Guide › Usage", text="Use diabetes app daily.", source="s"),
    ]
    build_index(tmp_path, "d", passages)
    hits = HybridRetriever(PassageIndex(tmp_path, "d"), mode="bm25").search("diabetes causes", top_k=5).hits
    ids = [h.passage.id for h in hits]
    assert "a#0" in ids or "b#0" in ids
    assert not ({"a#0", "b#0"} <= set(ids))  # copied text appears once
    assert not ({"a#0", "a#1"} <= set(ids))  # one chunk per answer
    assert {"k1", "k2"} <= set(ids)  # sections of one file are distinct sources
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_embedding_is_incremental_when_the_subset_grows(index, tmp_path):
    first = FakeEmbedder()
    assert embed_index(tmp_path, "t", first, limit=2) == 2
    second = FakeEmbedder()
    assert embed_index(tmp_path, "t", second, part_size=10) == 4  # total vectors
    assert second.calls == 1  # one call for the 2 new rows only; the first 2 were reused
    ids = np.load(tmp_path / "t.vec_ids.npy")
    assert sorted(ids.tolist()) == [1, 2, 3, 4]


def test_embedding_with_a_new_model_recomputes_everything(index, tmp_path):
    embed_index(tmp_path, "t", FakeEmbedder())
    other = FakeEmbedder()
    other.model = "another-model"
    assert embed_index(tmp_path, "t", other) == 4
    assert index.info().meta["embedding_model"] == "another-model"


def test_query_embedding_cache_computes_each_text_once():
    from billie.retrieval.embeddings import OllamaEmbedder

    calls = []

    class Counting(OllamaEmbedder):
        def _embed(self, texts, progress=None):
            calls.append(list(texts))
            return np.eye(len(texts), 4, dtype=np.float32)

    e = Counting("embeddinggemma")
    first = e.embed_queries(["a question", "another"])
    again = e.embed_queries(["another", "a question"])
    assert len(calls) == 1  # second call served from the cache
    assert np.array_equal(first[0], again[1]) and np.array_equal(first[1], again[0])
