"""Owns the three searchable corpora and their indexes.

* ``knowledge`` - small; rebuilt automatically whenever its files change.
* ``medquad`` / ``arxiv`` - large; built once by ``python -m billie.ingest``.
  If the full index is missing, a tiny sample index is built so every mode
  works out of the box; ``status()`` reports which one is active.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from billie.corpora.arxiv import iter_arxiv
from billie.corpora.knowledge import fingerprint, iter_knowledge, knowledge_files
from billie.corpora.medquad import iter_medquad_sample
from billie.retrieval.embeddings import Embedder
from billie.retrieval.hybrid import HybridRetriever
from billie.retrieval.index import PassageIndex, build_index, embed_index

log = logging.getLogger(__name__)

CorpusName = Literal["knowledge", "medquad", "arxiv"]
CORPUS_FOR_MODE: dict[str, CorpusName] = {"general": "knowledge", "medical": "medquad", "research": "arxiv"}


@dataclass(frozen=True)
class CorpusStatus:
    name: str
    ready: bool
    variant: str  # "full", "sample", "documents" or "missing"
    passages: int
    vectors: int
    embedding_model: str | None
    built_at: str | None
    search_mode: str  # what the retriever would run now: hybrid / bm25


class CorpusService:
    def __init__(
        self,
        index_dir: Path,
        *,
        knowledge_dirs: list[Path],
        repo_root: Path,
        medquad_sample: Path,
        arxiv_sample: Path,
        embedder: Embedder | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.knowledge_dirs = knowledge_dirs
        self.repo_root = repo_root
        self.medquad_sample = medquad_sample
        self.arxiv_sample = arxiv_sample
        self.embedder = embedder
        self._indexes: dict[str, PassageIndex] = {
            name: PassageIndex(index_dir, name) for name in ("knowledge", "medquad", "arxiv")
        }
        self._build_lock = threading.Lock()

    # ------------------------------------------------------------- access
    def index(self, name: CorpusName) -> PassageIndex:
        return self._indexes[name]

    def retriever(self, name: CorpusName) -> HybridRetriever:
        # Chunk collapsing suits MedQuAD (one answer split in chunks), not documents.
        return HybridRetriever(self._indexes[name], self.embedder, collapse_chunks=name != "knowledge")

    # ------------------------------------------------------------ startup
    def prepare(self) -> None:
        """Make every corpus searchable. Cheap when nothing changed."""
        self.sync_knowledge()
        if not self._indexes["medquad"].exists():
            log.warning("MedQuAD index missing - using bundled sample. Build: python -m billie.ingest medquad")
            self._build("medquad", iter_medquad_sample(self.medquad_sample), {"variant": "sample"})
        if not self._indexes["arxiv"].exists():
            log.warning("arXiv index missing - using bundled sample. Build: python -m billie.ingest arxiv")
            self._build("arxiv", iter_arxiv(self.arxiv_sample), {"variant": "sample"})

    def sync_knowledge(self, force: bool = False) -> bool:
        """Rebuild the knowledge index if its files changed. Returns True if rebuilt."""
        files = knowledge_files(self.knowledge_dirs)
        current = fingerprint(files)
        index = self._indexes["knowledge"]
        if not force and index.exists() and index.info().meta.get("fingerprint") == current:
            if self.embedder is not None and not index.has_vectors():
                self._embed("knowledge")
            return False
        self._build(
            "knowledge",
            iter_knowledge(files, self.repo_root),
            {"variant": "documents", "fingerprint": current, "files": str(len(files))},
        )
        if self.embedder is not None:
            self._embed("knowledge")
        return True

    def _build(self, name: CorpusName, passages: Iterable, meta: dict[str, str]) -> int:
        with self._build_lock:
            # Windows cannot replace a file another handle has open: close readers first.
            self._indexes[name].close()
            count = build_index(self.index_dir, name, passages, meta={"corpus": name, **meta})
            log.info("built %s index: %d passages", name, count)
            return count

    def _embed(self, name: CorpusName) -> None:
        assert self.embedder is not None
        if not getattr(self.embedder, "available", lambda: True)():
            log.warning("embedding model %s unavailable - %s search is BM25-only", self.embedder.model, name)
            return
        with self._build_lock:
            self._indexes[name].close()
            try:
                embed_index(self.index_dir, name, self.embedder)
            except Exception as exc:  # embedding is an enhancement; never block startup
                log.warning("embedding %s failed: %s", name, exc)

    # ------------------------------------------------------------- status
    def status(self) -> list[CorpusStatus]:
        statuses = []
        for name, index in self._indexes.items():
            info = index.info()
            has_vectors = info.vectors > 0
            dense_ok = (
                has_vectors and self.embedder is not None and info.meta.get("embedding_model") == self.embedder.model
            )
            statuses.append(
                CorpusStatus(
                    name=name,
                    ready=info.passages > 0,
                    variant=info.meta.get("variant", "missing") if index.exists() else "missing",
                    passages=info.passages,
                    vectors=info.vectors,
                    embedding_model=info.meta.get("embedding_model"),
                    built_at=info.meta.get("built_at"),
                    search_mode="hybrid" if dense_ok else "bm25",
                )
            )
        return statuses
