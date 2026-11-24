"""Persistent chat history in SQLite.

Replaces the original JSON-file memory, which rewrote the whole file on every
turn (not crash-safe), grew without bound and could not be shared between
processes. SQLite in WAL mode gives atomic appends and safe concurrent readers.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    mode TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""


@dataclass
class StoredMessage:
    id: int
    role: str
    content: str
    mode: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    feedback: int | None = None


@dataclass
class ConversationSummary:
    id: str
    title: str
    mode: str
    created_at: float
    updated_at: float
    message_count: int = 0


class ConversationNotFound(KeyError):
    pass


class MessageNotFound(KeyError):
    pass


class ConversationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Migration: databases created before feedback existed lack the column.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            if "feedback" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN feedback INTEGER")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation keeps this safe across threads.
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------- conversations
    def create(self, mode: str, title: str = "New chat") -> str:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id, title, mode, created_at, updated_at) VALUES (?,?,?,?,?)",
                (conversation_id, title[:120], mode, now, now),
            )
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is not None

    def recent(self, limit: int = 50) -> list[ConversationSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.mode, c.created_at, c.updated_at, COUNT(m.id) "
                "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ConversationSummary(*row) for row in rows]

    def get(self, conversation_id: str) -> tuple[ConversationSummary, list[StoredMessage]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, mode, created_at, updated_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                raise ConversationNotFound(conversation_id)
            messages = conn.execute(
                "SELECT id, role, content, mode, sources, meta, created_at, feedback FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        cid, title, mode, created, updated = row
        summary = ConversationSummary(cid, title, mode, created, updated, message_count=len(messages))
        return summary, [_to_message(m) for m in messages]

    def rename(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title.strip()[:120] or "Untitled", conversation_id)
            ).rowcount
        if not updated:
            raise ConversationNotFound(conversation_id)

    def delete(self, conversation_id: str) -> None:
        with self._connect() as conn:
            if not conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount:
                raise ConversationNotFound(conversation_id)

    # --------------------------------------------------------------- messages
    def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
        mode: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages(conversation_id, role, content, mode, sources, meta, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    conversation_id,
                    role,
                    content,
                    mode,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            return int(cursor.lastrowid or 0)

    def set_feedback(self, message_id: int, value: int | None) -> dict[str, Any]:
        """Record +1 / -1 (or clear with None) on an assistant message. Returns its meta."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT meta FROM messages WHERE id = ? AND role = 'assistant'", (message_id,)
            ).fetchone()
            if row is None:
                raise MessageNotFound(message_id)
            conn.execute("UPDATE messages SET feedback = ? WHERE id = ?", (value, message_id))
        return json.loads(row[0])

    def feedback_stats(self) -> dict[str, dict[str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT mode, SUM(feedback = 1), SUM(feedback = -1) FROM messages "
                "WHERE role = 'assistant' AND feedback IS NOT NULL GROUP BY mode"
            ).fetchall()
        return {mode: {"up": int(up or 0), "down": int(down or 0)} for mode, up, down in rows}

    def pop_last_turn(self, conversation_id: str) -> str | None:
        """Remove the last user+assistant pair (for regenerate/edit). Returns the removed question."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 2",
                (conversation_id,),
            ).fetchall()
            if len(rows) != 2 or rows[0][1] != "assistant" or rows[1][1] != "user":
                return None
            conn.execute("DELETE FROM messages WHERE id IN (?, ?)", (rows[0][0], rows[1][0]))
            return str(rows[1][2])

    def recent_turns(self, conversation_id: str, max_turns: int) -> list[tuple[str, str]]:
        """Last ``max_turns`` complete (user, assistant) exchanges, oldest first."""
        if max_turns <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, max_turns * 2 + 1),
            ).fetchall()
        rows.reverse()
        turns: list[tuple[str, str]] = []
        pending_user: str | None = None
        for role, content in rows:
            if role == "user":
                pending_user = content
            elif pending_user is not None:
                turns.append((pending_user, content))
                pending_user = None
        return turns[-max_turns:]


def _to_message(row: tuple[Any, ...]) -> StoredMessage:
    mid, role, content, mode, sources, meta, created, feedback = row
    return StoredMessage(mid, role, content, mode, json.loads(sources), json.loads(meta), created, feedback)
