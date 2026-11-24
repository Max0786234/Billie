"""Shared retrieval data contract.

A :class:`Passage` is the unit that gets indexed, retrieved, shown in the UI as a
citation card, and quoted to the LLM. Every corpus (knowledge base, MedQuAD,
arXiv) is converted into passages, so the chat pipeline is corpus-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Passage:
    id: str  # stable, unique within a corpus, e.g. "medquad:0001234-1#0"
    doc_id: str  # the source document the passage came from
    title: str
    text: str
    source: str  # human-readable origin: file name, "MedQuAD / CancerGov", "arXiv"
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float  # fused score (RRF) or the single retriever's score
    rank: int  # 1-based final rank
    components: dict[str, int] = field(default_factory=dict)  # retriever name -> rank


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    mode: str  # what actually ran: "hybrid", "bm25", "dense"
    degraded: bool = False  # True if a requested component was unavailable
    note: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> SearchResult: ...
