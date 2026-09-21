"""ConversationStore com SQLite
(equivalente a ``src/store/sqlite-conversation-store.ts``)."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

from app.domain.errors import ConversationNotFoundError
from app.domain.types import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationSummaryRecord,
)


class SqliteConversationStore:
    """ConversationStore com SQLite. Caminho: ``OPSPILOT_DB`` (default
    ``./data/opspilot.db``); use ``:memory:`` em testes. Compartilha o mesmo
    arquivo do SqliteOpsStore (conexão separada, DDL idempotente)."""

    def __init__(self, path: str | None = None) -> None:
        resolved_path = path or os.environ.get("OPSPILOT_DB", "./data/opspilot.db")

        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        self.database = sqlite3.connect(resolved_path, check_same_thread=False)
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON messages (conversation_id, created_at);

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                covered_count INTEGER NOT NULL CHECK (covered_count >= 0),
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            """
        )
        self.database.commit()

    def create(self) -> str:
        conversation_id = str(uuid.uuid4())
        self.database.execute(
            "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
            (conversation_id, int(time.time() * 1000)),
        )
        self.database.commit()
        return conversation_id

    def append(
        self, conversation_id: str, role: ConversationMessageRole, content: str
    ) -> ConversationMessage:
        self._require_conversation(conversation_id)
        message_id = str(uuid.uuid4())
        created_at = int(time.time() * 1000)
        self.database.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, created_at),
        )
        self.database.commit()
        return ConversationMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
        )

    def last_messages(self, conversation_id: str, limit: int) -> list[ConversationMessage]:
        self._require_conversation(conversation_id)
        rows = self.database.execute(
            "SELECT id, conversation_id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY rowid DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [self._map_message(r) for r in reversed(rows)]

    def count_messages(self, conversation_id: str) -> int:
        self._require_conversation(conversation_id)
        row = self.database.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return int(row[0] if row else 0)

    def messages_ascending(
        self, conversation_id: str, offset: int, limit: int
    ) -> list[ConversationMessage]:
        self._require_conversation(conversation_id)
        rows = self.database.execute(
            "SELECT id, conversation_id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY rowid ASC LIMIT ? OFFSET ?",
            (conversation_id, limit, offset),
        ).fetchall()
        return [self._map_message(r) for r in rows]

    def get_summary(self, conversation_id: str) -> ConversationSummaryRecord | None:
        self._require_conversation(conversation_id)
        row = self.database.execute(
            "SELECT conversation_id, summary_text, covered_count, updated_at "
            "FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not row:
            return None
        return ConversationSummaryRecord(
            conversation_id=row[0], text=row[1], covered_count=row[2], updated_at=row[3]
        )

    def upsert_summary(self, conversation_id: str, text: str, covered_count: int) -> None:
        self._require_conversation(conversation_id)
        self.database.execute(
            """
            INSERT INTO conversation_summaries
                (conversation_id, summary_text, covered_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                covered_count = excluded.covered_count,
                updated_at = excluded.updated_at
            """,
            (conversation_id, text, covered_count, int(time.time() * 1000)),
        )
        self.database.commit()

    @staticmethod
    def _map_message(row: tuple) -> ConversationMessage:
        return ConversationMessage(
            id=row[0], conversation_id=row[1], role=row[2], content=row[3], created_at=row[4]
        )

    def _require_conversation(self, conversation_id: str) -> None:
        row = self.database.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            raise ConversationNotFoundError(conversation_id)
