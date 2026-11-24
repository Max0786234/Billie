"""Hybrid retrieval: BM25 (FTS5) + dense cosine, fused with Reciprocal Rank Fusion.

RRF(d) = sum_r 1 / (k + rank_r(d))      (Cormack et al., 2009; k = 60)

RRF uses ranks, not raw scores, so BM25 scores and cosine similarities never have
to be put on a common scale. If the embedding model is unavailable the retriever
degrades to BM25 and says so in the result instead of failing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from billie.llm.base import LLMError
from billie.retrieval.base import SearchHit, SearchResult
from billie.retrieval.embeddings import Embedder
from billie.retrieval.index import PassageIndex

Mode = Literal["hybrid", "bm25", "dense"]
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse ranked id lists. Returns (id, score, {retriever: rank}) sorted best-first, ties by id."""
    scores: dict[int, float] = {}
    components: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, doc in enumerate(ids, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            components.setdefault(doc, {})[name] = rank
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(doc, score, components[doc]) for doc, score in ordered]


class HybridRetriever:
    def __init__(
        self,
        index: PassageIndex,
        embedder: Embedder | None = None,
        *,
        mode: Mode = "hybrid",
        candidates: int = 30,
        rrf_k: int = 60,
        min_dense_score: float = 0.30,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        # Dense search always returns *something*; below this cosine a dense-only hit is noise.
        self.min_dense_score = min_dense_score

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        timings: dict[str, float] = {}
        rankings: dict[str, list[int]] = {}
        notes: list[str] = []
        degraded = False

        def run_bm25() -> list[int]:
            started = time.perf_counter()
            ids = [rowid for rowid, _ in self.index.search_bm25(query, self.candidates)]
            timings["bm25_ms"] = _ms(started)
            return ids

        # BM25 (SQLite, releases the GIL) and dense (embedding over HTTP) are independent: run them
        # in parallel. Same rankings, same fusion - only the wall-clock time changes.
        bm25_future = _POOL.submit(run_bm25) if self.mode in ("hybrid", "bm25") else None

        if self.mode in ("hybrid", "dense"):
            dense, reason = self._dense(query, timings)
            if dense is None:
                degraded = True
                notes.append(reason)
                if self.mode == "dense":  # dense-only requested but impossible: fall back to BM25
                    rankings["bm25"] = [rowid for rowid, _ in self.index.search_bm25(query, self.candidates)]
            else:
                rankings["dense"] = dense
        if bm25_future is not None:
            rankings = {"bm25": bm25_future.result(), **rankings}  # keep bm25 first, as before

        # Over-fetch, then drop duplicates: MedQuAD copies some answers verbatim into several
        # documents, and long answers are split into chunks. Each source slot should add new information.
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)[: top_k * 3]
        passages = self.index.get(doc for doc, _, _ in fused)
        hits: list[SearchHit] = []
        seen_docs: set[tuple[str, str]] = set()
        seen_content: set[str] = set()
        for doc, score, components in fused:
            passage = passages.get(doc)
            if passage is None:
                continue
            content_key = content_fingerprint(passage.title, passage.text)
            # (doc, title): chunks of one answer collapse; different sections of one file do not.
            doc_key = (passage.doc_id, passage.title)
            if doc_key in seen_docs or content_key in seen_content:
                continue
            seen_docs.add(doc_key)
            seen_content.add(content_key)
            hits.append(SearchHit(passage=passage, score=score, rank=len(hits) + 1, components=components))
            if len(hits) == top_k:
                break
        active = "+".join(name for name in ("bm25", "dense") if name in rankings) or "none"
        return SearchResult(
            hits=hits,
            mode={"bm25+dense": "hybrid"}.get(active, active),
            degraded=degraded,
            note="; ".join(notes),
            timings_ms=timings,
        )

    def _dense(self, query: str, timings: dict[str, float]) -> tuple[list[int] | None, str]:
        if self.embedder is None:
            return None, "no embedding model configured"
        if not self.index.has_vectors():
            return None, f"index '{self.index.name}' has no embeddings yet"
        index_model = self.index.embedding_model
        if index_model and index_model != self.embedder.model:
            return None, f"index embedded with '{index_model}', but '{self.embedder.model}' is configured"
        started = time.perf_counter()
        try:
            vector = self.embedder.embed_queries([query])[0]
        except LLMError as exc:
            return None, f"embedding unavailable ({exc})"
        timings["embed_ms"] = _ms(started)
        started = time.perf_counter()
        hits = self.index.search_dense(vector, self.candidates)
        timings["dense_ms"] = _ms(started)
        return [rowid for rowid, score in hits if score >= self.min_dense_score], ""


def content_fingerprint(title: str, text: str) -> str:
    """Case/whitespace-insensitive key for 'the same passage published twice'."""
    return " ".join(f"{title} {text[:300]}".lower().split())


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
