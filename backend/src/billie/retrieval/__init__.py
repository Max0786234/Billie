"""Retrieval: passage contract, on-disk BM25 (SQLite FTS5) + dense index, hybrid RRF retriever."""

from billie.retrieval.base import Passage, Retriever, SearchHit, SearchResult
from billie.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from billie.retrieval.index import PassageIndex, build_index, embed_index

__all__ = [
    "HybridRetriever",
    "Passage",
    "PassageIndex",
    "Retriever",
    "SearchHit",
    "SearchResult",
    "build_index",
    "embed_index",
    "reciprocal_rank_fusion",
]
