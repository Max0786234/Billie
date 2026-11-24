"""Answer cache for repeated stand-alone questions.

On a CPU-only laptop a fresh answer costs 15-45 s, but people often re-ask the same
question (demos, refreshes, shared links). A cached answer streams back instantly.

Correctness rules:
* Only first-turn, text-only questions are cached: follow-ups depend on history and
  image questions depend on the image.
* The key includes the mode, the normalised question, the LLM provider/model and the
  corpus *build stamp*, so rebuilding an index or switching models invalidates entries.
* Only answers produced by the LLM are cached (never the retrieval-only fallback),
  and never ungrounded ones.
* LRU-bounded, stored in SQLite (survives restarts, safe across threads).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key TEXT PRIMARY KEY,
    meta TEXT NOT NULL,
    done TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
"""


def cache_key(mode: str, question: str, model: str, corpus_stamp: str) -> str:
    normalised = " ".join(question.casefold().split()).rstrip(" ?!.")
    return hashlib.sha256(f"{mode}\x1f{normalised}\x1f{model}\x1f{corpus_stamp}".encode()).hexdigest()


class AnswerCache:
    def __init__(self, db_path: Path, max_entries: int = 500) -> None:
        self.db_path = db_path
        self.max_entries = max_entries
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT meta, done FROM answers WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE answers SET last_used = ?, hits = hits + 1 WHERE key = ?", (time.time(), key))
        return json.loads(row[0]), json.loads(row[1])

    def put(self, key: str, meta: dict[str, Any], done: dict[str, Any]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers(key, meta, done, created_at, last_used, hits) VALUES (?,?,?,?,?,0)",
                (key, json.dumps(meta, ensure_ascii=False), json.dumps(done, ensure_ascii=False), now, now),
            )
            # LRU eviction
            conn.execute(
                "DELETE FROM answers WHERE key NOT IN (SELECT key FROM answers ORDER BY last_used DESC LIMIT ?)",
                (self.max_entries,),
            )

    def forget(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM answers WHERE key = ?", (key,))

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            entries, hits = conn.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
        return {"entries": int(entries), "hits": int(hits)}

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM answers")
