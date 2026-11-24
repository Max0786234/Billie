"""Private, temporary documents for visitors of the public demo (DEVLOG P2.19).

A visitor can upload a few of their own files and ask about them in General mode.
Each visitor gets a separate small index (BM25 + embeddings, same as the shared
corpora) that only their own questions search. Files are deleted:

* after ``ttl_seconds`` without use (default 1 hour; uploads, listing and questions
  all count as use) - a browser cannot reliably report that a tab was closed, so an
  inactivity timer is the dependable way to "release it when they leave";
* when the visitor removes them;
* on every start: the folder is wiped, because the store is temporary by design.

Limits keep one visitor from filling the disk or monopolising the 2-CPU server:
files per visitor, bytes per file, passages per file (which bounds embedding time),
total bytes on disk and the number of visitors holding documents at once.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from billie.corpora.knowledge import iter_knowledge
from billie.retrieval.base import Passage
from billie.retrieval.embeddings import Embedder
from billie.retrieval.hybrid import HybridRetriever
from billie.retrieval.index import PassageIndex, build_index
from billie.uploads import DocumentStore, UploadError

log = logging.getLogger(__name__)

_OWNER = re.compile(r"^v_[0-9a-f]{32}$")  # see api.demo.visitor_owner
INDEX_NAME = "docs"


@dataclass(frozen=True)
class PrivateDocument:
    name: str
    size: int
    passages: int


class PrivateDocuments:
    def __init__(
        self,
        root: Path,
        embedder: Embedder | None,
        *,
        ttl_seconds: int = 3600,
        max_files: int = 3,
        max_bytes: int = 5 * 1024 * 1024,
        max_passages: int = 150,
        max_total_bytes: int = 300 * 1024 * 1024,
        max_visitors: int = 100,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = root
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.max_passages = max_passages
        self.max_total_bytes = max_total_bytes
        self.max_visitors = max_visitors
        self.clock = clock
        self._last_used: dict[str, float] = {}
        self._indexes: dict[str, PassageIndex] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        shutil.rmtree(root, ignore_errors=True)  # temporary by design: nothing survives a restart
        root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    def _dir(self, owner: str) -> Path:
        if not _OWNER.match(owner):
            raise UploadError("Invalid visitor.")
        return self.root / owner

    def _owner_lock(self, owner: str) -> threading.Lock:
        with self._lock:
            return self._locks.setdefault(owner, threading.Lock())

    def _store(self, owner: str) -> DocumentStore:
        return DocumentStore(self._dir(owner) / "files", self.max_bytes)

    def _index(self, owner: str) -> PassageIndex:
        with self._lock:
            index = self._indexes.get(owner)
            if index is None:
                index = self._indexes[owner] = PassageIndex(self._dir(owner) / "index", INDEX_NAME)
            return index

    def _touch(self, owner: str) -> None:
        with self._lock:
            self._last_used[owner] = self.clock()

    def _disk_usage(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    # --------------------------------------------------------------- read
    def documents(self, owner: str) -> list[PrivateDocument]:
        files = self._store(owner).list()
        if files:
            self._touch(owner)
        counts = self._passage_counts(owner)
        return [PrivateDocument(f.name, f.size, counts.get(f.name, 0)) for f in files]

    def expires_at(self, owner: str) -> float | None:
        with self._lock:
            last = self._last_used.get(owner)
        return None if last is None or not self._store(owner).list() else last + self.ttl_seconds

    def retriever(self, owner: str | None) -> HybridRetriever | None:
        """This visitor's search over their own files, or None if they have none. Counts as use."""
        if owner is None or not _OWNER.match(owner):
            return None
        index = self._index(owner)
        if not index.exists():
            return None
        self._touch(owner)
        return HybridRetriever(index, self.embedder)

    # -------------------------------------------------------------- write
    def add(self, owner: str, filename: str, data: bytes) -> list[PrivateDocument]:
        self.sweep()
        with self._owner_lock(owner):
            store = self._store(owner)
            existing = {f.name for f in store.list()}
            name = DocumentStore._safe_name(filename)
            if name not in existing and len(existing) >= self.max_files:
                raise UploadError(f"You can keep up to {self.max_files} documents. Remove one to add another.")
            with self._lock:
                new_visitor = owner not in self._last_used
                active = len(self._last_used)
            if new_visitor and active >= self.max_visitors:
                raise UploadError("The demo is busy right now. Please try uploading again in a few minutes.")
            if self._disk_usage() + len(data) > self.max_total_bytes:
                raise UploadError("The demo's document storage is full right now. Please try again later.")
            stored = store.save(filename, data)  # validates type, size and encoding
            self._touch(owner)
            path = store.directory / stored.name
            try:
                # Check the new file on its own first: parsing is cheap, embedding is not, so a file
                # that is too long or has no text is refused before any embedding work (measured:
                # an oversized file used to take 88 s to be rejected).
                added = sum(1 for _ in _private_passages([path], self._dir(owner)))
                if added == 0:
                    raise UploadError("No text could be read from this file (a scanned PDF has no text layer).")
                if added > self.max_passages:
                    raise UploadError(
                        f"This document is too long for the demo (about {added} passages; the limit is "
                        f"{self.max_passages}, roughly 40 pages). Try a shorter document."
                    )
            except UploadError:
                self._discard(path)
                raise
            except Exception as exc:  # unreadable or malformed file (e.g. a broken PDF)
                log.info("private document rejected: %s", type(exc).__name__)
                self._discard(path)
                raise UploadError(
                    "This file could not be read. Check that it is a valid PDF, Markdown or text file."
                ) from exc
            self._rebuild(owner)
            return self.documents(owner)

    def delete(self, owner: str, name: str) -> list[PrivateDocument]:
        with self._owner_lock(owner):
            self._store(owner).delete(name)
            self._rebuild(owner)
            return self.documents(owner)

    def clear(self, owner: str) -> None:
        with self._owner_lock(owner):
            self._drop(owner)

    def sweep(self) -> int:
        """Delete every visitor's documents unused for ``ttl_seconds``. Returns how many were removed."""
        now = self.clock()
        with self._lock:
            expired = [o for o, last in self._last_used.items() if now - last > self.ttl_seconds]
        for owner in expired:
            with self._owner_lock(owner):
                self._drop(owner)
        return len(expired)

    # ----------------------------------------------------------- internals
    def _drop(self, owner: str) -> None:
        with self._lock:
            index = self._indexes.pop(owner, None)
            self._last_used.pop(owner, None)
        if index is not None:
            index.close()
        shutil.rmtree(self._dir(owner), ignore_errors=True)

    @staticmethod
    def _discard(path: Path) -> None:
        path.unlink(missing_ok=True)
        _vector_cache(path).unlink(missing_ok=True)

    def _rebuild(self, owner: str) -> list[Passage]:
        """Rebuild this visitor's index from their files (small: a few hundred passages at most).

        The keyword index is rebuilt from scratch (milliseconds). Embeddings are cached per file
        next to it, so adding a second or third file only embeds the new one.
        """
        store = self._store(owner)
        files = [store.directory / f.name for f in store.list()]
        names = {f.name for f in files}
        for cache in store.directory.glob("*.vectors.npz") if store.directory.exists() else []:
            if cache.name.removesuffix(".vectors.npz") not in names:  # cache of a removed file
                cache.unlink(missing_ok=True)
        index = self._index(owner)
        index.close()  # Windows cannot replace an open file
        if not files:
            shutil.rmtree(index.directory, ignore_errors=True)
            return []
        passages = list(_private_passages(files, self._dir(owner)))
        build_index(index.directory, INDEX_NAME, passages, meta={"corpus": "private", "variant": "documents"})
        if self.embedder is not None and passages:
            try:
                self._attach_vectors(index, files, passages)
            except Exception as exc:  # embeddings are an enhancement: keyword search still works
                log.warning("embedding private documents failed: %s", exc)
        return passages

    def _attach_vectors(self, index: PassageIndex, files: list[Path], passages: list[Passage]) -> None:
        """Write the index's vector matrix, embedding only passages without a cached vector."""
        assert self.embedder is not None
        model = self.embedder.model
        by_id: dict[str, np.ndarray] = {}
        for path in files:
            cache = _vector_cache(path)
            if cache.exists():
                with np.load(cache) as data:
                    if str(data["model"]) == model:
                        by_id.update(zip((str(i) for i in data["ids"]), data["vectors"], strict=True))
        for path in files:
            mine = [p for p in passages if p.metadata.get("file") == path.name]
            missing = [p for p in mine if p.id not in by_id]
            if missing:
                vectors = self.embedder.embed_documents([p.text[:600] for p in missing], [p.title for p in missing])
                by_id.update(zip((p.id for p in missing), vectors, strict=True))
            np.savez(
                _vector_cache(path),
                ids=np.array([p.id for p in mine]),
                vectors=np.array([by_id[p.id] for p in mine], dtype=np.float32),
                model=np.array(model),
            )
        conn = sqlite3.connect(index.db_path)
        try:
            rows = conn.execute("SELECT rowid, id FROM passages ORDER BY rowid").fetchall()
            dim = len(next(iter(by_id.values())))
            conn.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                [("embedding_model", model), ("embedding_dim", str(dim))],
            )
            conn.commit()
        finally:
            conn.close()
        np.save(index.directory / f"{INDEX_NAME}.vec_ids.npy", np.array([r[0] for r in rows], dtype=np.int64))
        np.save(index.directory / f"{INDEX_NAME}.vec.npy", np.array([by_id[r[1]] for r in rows], dtype=np.float32))

    def _passage_counts(self, owner: str) -> dict[str, int]:
        index = self._index(owner)
        if not index.exists():
            return {}
        conn = sqlite3.connect(f"file:{index.db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT json_extract(metadata, '$.file'), COUNT(*) FROM passages GROUP BY 1").fetchall()
        finally:
            conn.close()
        return {str(name): int(n) for name, n in rows}


def _vector_cache(path: Path) -> Path:
    return path.with_name(f"{path.name}.vectors.npz")


def _private_passages(files: list[Path], root: Path) -> Iterator[Passage]:
    """Knowledge-style passages, labelled as the visitor's own and namespaced away from shared ids."""
    # 600-character chunks: the prompt shows at most 650 characters per source (chat/prompts.py),
    # so a longer chunk would hide its end from the model (found testing a real handbook).
    for passage in iter_knowledge(files, root, max_chars=600):
        file = Path(passage.doc_id).name
        yield replace(
            passage,
            id=f"private:{passage.id}",
            source=f"Your document · {file}",
            metadata={**passage.metadata, "file": file, "private": True},
        )
