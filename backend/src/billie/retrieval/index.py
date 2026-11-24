"""On-disk passage index: SQLite (passages + FTS5/BM25) with optional dense vectors.

Files for an index named ``medquad`` under ``data/index/``::

    medquad.sqlite        passages table, FTS5 index, and a ``meta`` manifest
    medquad.vec.npy       float32 L2-normalised embedding matrix (optional)
    medquad.vec_ids.npy   int64 passage rowids, one per matrix row
    medquad.vec.parts/    resumable checkpoints while embeddings are computed

Design notes
* FTS5's built-in ``bm25()`` ranks on disk, so the 800k-paper arXiv corpus does not
  need to fit in RAM.
* Builds go to a temporary file then ``os.replace`` - an interrupted build (e.g. power
  loss) never corrupts the live index.
* Embedding is checkpointed in parts so a long run on CPU can resume.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from billie.retrieval.base import Passage
from billie.retrieval.embeddings import Embedder

SCHEMA_VERSION = "2"
_TOKEN = re.compile(r"\w+", re.UNICODE)
# Very common words add cost to OR queries on big corpora without adding signal.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it "
    "its me my of on or our should so than that the their them then there these they this to was we were "
    "what when where which who whom why will with would you your about tell explain please".split()
)

ProgressFn = Callable[[int, int], None]


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms ("" if nothing searchable)."""
    terms: list[str] = []
    for token in _TOKEN.findall(text.casefold()):
        if token in _STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        if token not in terms:
            terms.append(token)
    return " OR ".join(f'"{term}"' for term in terms[:32])


@dataclass(frozen=True)
class IndexInfo:
    name: str
    passages: int
    vectors: int
    meta: dict[str, str]


class PassageIndex:
    """Read-side of an index. Thread-safe for concurrent searches."""

    def __init__(self, directory: Path, name: str) -> None:
        self.directory = directory
        self.name = name
        self.db_path = directory / f"{name}.sqlite"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._vectors: np.ndarray | None = None
        self._vector_ids: np.ndarray | None = None
        self._loaded_mtime: float | None = None

    # ---------------------------------------------------------------- state
    def exists(self) -> bool:
        return self.db_path.exists()

    def _connection(self) -> sqlite3.Connection:
        mtime = self.db_path.stat().st_mtime
        if self._conn is None or self._loaded_mtime != mtime:
            # Index was rebuilt on disk: reopen and drop cached vectors.
            if self._conn is not None:
                self._conn.close()
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            self._vectors = self._vector_ids = None
            self._loaded_mtime = mtime
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def info(self) -> IndexInfo:
        if not self.exists():
            return IndexInfo(self.name, 0, 0, {})
        with self._lock:
            conn = self._connection()
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            vectors, _ = self._load_vectors()
        return IndexInfo(self.name, count, 0 if vectors is None else len(vectors), meta)

    @property
    def embedding_model(self) -> str | None:
        return self.info().meta.get("embedding_model")

    # --------------------------------------------------------------- search
    def search_bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match or not self.exists():
            return []
        with self._lock:
            rows = (
                self._connection()
                .execute(
                    "SELECT rowid, bm25(passages_fts, 2.0, 1.0) FROM passages_fts WHERE passages_fts MATCH ? "
                    "ORDER BY bm25(passages_fts, 2.0, 1.0) LIMIT ?",
                    (match, k),
                )
                .fetchall()
            )
        # FTS5 bm25() is "lower is better"; flip sign so higher = more relevant.
        return [(int(rowid), -float(score)) for rowid, score in rows]

    def has_vectors(self) -> bool:
        if not self.exists():
            return False
        with self._lock:
            self._connection()
            vectors, _ = self._load_vectors()
        return vectors is not None and len(vectors) > 0

    def search_dense(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            self._connection()
            vectors, ids = self._load_vectors()
        if vectors is None or ids is None or len(vectors) == 0:
            return []
        scores = vectors @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top]

    def get(self, rowids: Iterable[int]) -> dict[int, Passage]:
        rowids = list(rowids)
        if not rowids:
            return {}
        placeholders = ",".join("?" * len(rowids))
        with self._lock:
            rows = (
                self._connection()
                .execute(
                    f"SELECT rowid, id, doc_id, title, text, source, url, metadata FROM passages WHERE rowid IN ({placeholders})",
                    rowids,
                )
                .fetchall()
            )
        return {row[0]: _row_to_passage(row) for row in rows}

    def _load_vectors(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._vectors is None:
            vec_path = self.directory / f"{self.name}.vec.npy"
            ids_path = self.directory / f"{self.name}.vec_ids.npy"
            if vec_path.exists() and ids_path.exists():
                # Loaded into RAM (not memory-mapped): on Windows a mapped file cannot be
                # replaced by a rebuild while the app is running.
                self._vectors = np.load(vec_path)
                self._vector_ids = np.load(ids_path)
        return self._vectors, self._vector_ids


def _row_to_passage(row: tuple[Any, ...]) -> Passage:
    _, pid, doc_id, title, text, source, url, metadata = row
    return Passage(
        id=pid, doc_id=doc_id, title=title, text=text, source=source, url=url, metadata=json.loads(metadata or "{}")
    )


# ======================================================================= build
def build_index(
    directory: Path,
    name: str,
    passages: Iterable[Passage],
    *,
    meta: dict[str, str] | None = None,
    progress: ProgressFn | None = None,
    batch_size: int = 2000,
) -> int:
    """(Re)build the SQLite part of an index atomically. Returns the passage count.

    Existing vectors are invalidated because rowids change on rebuild.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{name}.sqlite"
    tmp = directory / f"{name}.sqlite.building"
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE passages (
                rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, doc_id TEXT NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL,
                url TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        count = 0
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for passage in passages:
            batch.append(
                (
                    passage.id,
                    passage.doc_id,
                    passage.title,
                    passage.text,
                    passage.source,
                    passage.url,
                    json.dumps(passage.metadata, ensure_ascii=False),
                )
            )
            if len(batch) >= batch_size:
                count += _insert(conn, batch)
                batch = []
                if progress:
                    progress(count, 0)
        count += _insert(conn, batch)
        # External-content FTS table: the text lives once, in `passages`.
        conn.executescript(
            """
            CREATE VIRTUAL TABLE passages_fts USING fts5(
                title, text, content='passages', content_rowid='rowid',
                tokenize='porter unicode61 remove_diacritics 2'
            );
            INSERT INTO passages_fts(passages_fts) VALUES ('rebuild');
            INSERT INTO passages_fts(passages_fts) VALUES ('optimize');
            """
        )
        manifest = {"schema_version": SCHEMA_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
        conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", manifest.items())
        conn.commit()
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)  # never leave a half-built multi-GB file behind
        raise
    conn.close()
    _remove_vectors(directory, name)
    os.replace(tmp, final)
    if progress:
        progress(count, count)
    return count


def _insert(conn: sqlite3.Connection, batch: Sequence[tuple[str, ...]]) -> int:
    if batch:
        conn.executemany(
            "INSERT INTO passages(id, doc_id, title, text, source, url, metadata) VALUES (?,?,?,?,?,?,?)",
            batch,
        )
    return len(batch)


def _remove_vectors(directory: Path, name: str) -> None:
    for suffix in (".vec.npy", ".vec_ids.npy"):
        (directory / f"{name}{suffix}").unlink(missing_ok=True)
    shutil.rmtree(directory / f"{name}.vec.parts", ignore_errors=True)


def embed_index(
    directory: Path,
    name: str,
    embedder: Embedder,
    *,
    where: str = "",
    limit: int | None = None,
    newest_first: bool = False,
    max_chars: int | None = None,
    part_size: int = 512,
    progress: ProgressFn | None = None,
) -> int:
    """Compute dense vectors for (a subset of) an index, resumably.

    ``where`` is an optional SQL filter on ``passages`` used to embed a subset of a
    huge corpus (e.g. recent arXiv papers); ``max_chars`` truncates each passage before
    embedding. Vectors that already exist for the same model are kept, so raising ``limit``
    only embeds the new rows. Returns the total number of vectors in the index.
    """
    db_path = directory / f"{name}.sqlite"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        order = "rowid DESC" if newest_first else "rowid"
        sql = "SELECT rowid, title, text FROM passages" + (f" WHERE {where}" if where else "") + f" ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()

    # Incremental: keep vectors already computed with the same model and embed only new rows,
    # e.g. growing the arXiv dense subset from 20k to 100k papers without redoing the first 20k.
    vec_path, ids_path = directory / f"{name}.vec.npy", directory / f"{name}.vec_ids.npy"
    existing_ids = np.zeros(0, dtype=np.int64)
    existing_vecs: np.ndarray | None = None
    if vec_path.exists() and ids_path.exists() and meta.get("embedding_model") == embedder.model:
        existing_ids = np.load(ids_path)
        existing_vecs = np.load(vec_path)
        done = set(existing_ids.tolist())
        rows = [r for r in rows if r[0] not in done]

    parts_dir = directory / f"{name}.vec.parts"
    parts_dir.mkdir(exist_ok=True)
    total = len(rows)
    for part_no, start in enumerate(range(0, total, part_size)):
        part_path = parts_dir / f"part-{part_no:05d}.npz"
        if part_path.exists():  # done in a previous (interrupted) run
            if progress:
                progress(min(start + part_size, total), total)
            continue
        chunk = rows[start : start + part_size]
        # Truncation trades a little recall for a lot of speed on CPU (cost grows with tokens).
        texts = [r[2][:max_chars] if max_chars else r[2] for r in chunk]
        vectors = embedder.embed_documents(texts, [r[1] for r in chunk])
        tmp = part_path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array([r[0] for r in chunk], dtype=np.int64), vectors=vectors)
        os.replace(tmp, part_path)
        if progress:
            progress(min(start + part_size, total), total)

    # Assemble parts into the final matrix.
    ids_list, vec_list = [], []
    for part_path in sorted(parts_dir.glob("part-*.npz")):
        if part_path.name.endswith(".tmp.npz"):
            continue
        with np.load(part_path) as data:
            ids_list.append(data["ids"])
            vec_list.append(data["vectors"])
    if existing_vecs is not None and len(existing_ids):
        ids_list.insert(0, existing_ids)
        vec_list.insert(0, existing_vecs)
    if not vec_list:
        return 0
    ids = np.concatenate(ids_list)
    vectors = np.concatenate(vec_list).astype(np.float32)
    np.save(directory / f"{name}.vec_ids.npy.tmp.npy", ids)
    np.save(directory / f"{name}.vec.npy.tmp.npy", vectors)
    os.replace(directory / f"{name}.vec_ids.npy.tmp.npy", directory / f"{name}.vec_ids.npy")
    os.replace(directory / f"{name}.vec.npy.tmp.npy", directory / f"{name}.vec.npy")
    _set_meta(db_path, {"embedding_model": embedder.model, "embedding_dim": str(vectors.shape[1])})
    shutil.rmtree(parts_dir, ignore_errors=True)
    return len(ids)


def subset_index(src_dir: Path, name: str, dst_dir: Path, *, progress: ProgressFn | None = None) -> int:
    """Copy only the passages that have embeddings into a new, smaller index.

    Used to ship a deployable bundle: the full arXiv index is 2.6 GB (980k papers) but only
    the 50k newest are embedded. The copy keeps hybrid search identical for those papers.
    Rowids change, so the vector ids are remapped to the new rows; the vectors are copied as-is.
    Returns the number of passages written.
    """
    src = PassageIndex(src_dir, name)
    with src._lock:
        conn = src._connection()
        vectors, ids = src._load_vectors()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    if vectors is None or ids is None:
        raise ValueError(f"index '{name}' has no embeddings to subset by")
    order = [int(i) for i in ids]

    def passages() -> Iterable[Passage]:
        for start in range(0, len(order), 2000):  # same order as the vector matrix
            chunk = src.get(order[start : start + 2000])
            for rowid in order[start : start + 2000]:
                yield chunk[rowid]

    keep = {k: v for k, v in meta.items() if k not in ("schema_version", "built_at")}
    count = build_index(
        dst_dir, name, passages(), meta={**keep, "subset_of": str(meta.get("variant", ""))}, progress=progress
    )
    src.close()
    # build_index assigns rowids 1..N in insertion order, i.e. matrix row i -> rowid i+1.
    np.save(dst_dir / f"{name}.vec_ids.npy", np.arange(1, count + 1, dtype=np.int64))
    np.save(dst_dir / f"{name}.vec.npy", vectors.astype(np.float32))
    return count


def _set_meta(db_path: Path, values: dict[str, str]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", values.items())
        conn.commit()
    finally:
        conn.close()
