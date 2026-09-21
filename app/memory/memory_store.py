"""MemoryStore com SQLite (equivalente a ``src/memory/memory-store.ts``)."""

from __future__ import annotations

import os
import sqlite3
import struct
import time
import uuid
from pathlib import Path

from app.domain.errors import InvalidMemoryInputError
from app.domain.types import Embedder, RecalledMemory, RememberResult
from app.memory.embeddings import EMBEDDING_DIM, get_default_embedder

_DEDUP_THRESHOLD = 0.92
_RECALL_MIN_SCORE = 0.3
_RECALL_TOP_K = 3


def _require_non_empty(label: str, value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise InvalidMemoryInputError(f"{label} must be non-empty")
    return trimmed


def dot(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def embedding_to_buffer(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def buffer_to_embedding(buf: bytes) -> list[float]:
    count = len(buf) // 4
    values = list(struct.unpack(f"<{count}f", buf))
    if len(values) != EMBEDDING_DIM:
        raise InvalidMemoryInputError(
            f"Stored embedding has dimension {len(values)}; expected {EMBEDDING_DIM}"
        )
    return values


class SqliteMemoryStore:
    """MemoryStore com SQLite. Caminho: ``OPSPILOT_DB`` (default
    ``./data/opspilot.db``); use ``:memory:`` em testes."""

    def __init__(self, path: str | None = None, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or get_default_embedder()
        resolved_path = path or os.environ.get("OPSPILOT_DB", "./data/opspilot.db")

        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        self.database = sqlite3.connect(resolved_path, check_same_thread=False)
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories (user_id);
            """
        )
        self.database.commit()

    def _all_for_user(self, user_id: str) -> list[tuple[str, str, list[float]]]:
        rows = self.database.execute(
            "SELECT id, fact, embedding FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [(row[0], row[1], buffer_to_embedding(row[2])) for row in rows]

    async def remember(self, user_id: str, fact: str) -> RememberResult:
        uid = _require_non_empty("userId", user_id)
        text = _require_non_empty("fact", fact)
        embedding = await self._embedder.embed(text)

        for existing_id, _fact, existing_embedding in self._all_for_user(uid):
            if dot(embedding, existing_embedding) > _DEDUP_THRESHOLD:
                return RememberResult(id=existing_id, stored=False)

        memory_id = str(uuid.uuid4())
        created_at = int(time.time() * 1000)
        self.database.execute(
            "INSERT INTO memories (id, user_id, fact, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, uid, text, embedding_to_buffer(embedding), created_at),
        )
        self.database.commit()
        return RememberResult(id=memory_id, stored=True)

    async def recall(
        self, user_id: str, query: str, k: int = _RECALL_TOP_K
    ) -> list[RecalledMemory]:
        """Top-k por cosseno (dot de vetores normalizados).
        Ordem: pontua tudo → filtra >= 0.3 → ordena desc → slice(0, k)
        (filtra antes de cortar para não perder itens intermediários acima do piso)."""
        uid = _require_non_empty("userId", user_id)
        q_text = _require_non_empty("query", query)
        q = await self._embedder.embed(q_text)

        scored = [
            RecalledMemory(id=mid, fact=fact, score=dot(q, embedding))
            for mid, fact, embedding in self._all_for_user(uid)
        ]
        scored = [m for m in scored if m.score >= _RECALL_MIN_SCORE]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:k]

    async def forget(self, user_id: str, id: str) -> bool:
        uid = _require_non_empty("userId", user_id)
        memory_id = _require_non_empty("id", id)
        cursor = self.database.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, uid)
        )
        self.database.commit()
        return cursor.rowcount == 1
