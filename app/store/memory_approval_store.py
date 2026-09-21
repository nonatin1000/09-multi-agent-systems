"""ApprovalStore em memória (equivalente a ``src/store/memory-approval-store.ts``)."""

from __future__ import annotations

import uuid

from app.domain.types import ChatRequestSnapshot, PendingApproval


class MemoryApprovalStore:
    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def save(
        self,
        request_id: str,
        created_at: int,
        summary: str,
        chat_request: ChatRequestSnapshot,
        conversation_id: str | None,
        approval_id: str | None = None,
    ) -> PendingApproval:
        resolved_id = approval_id or str(uuid.uuid4())
        record = PendingApproval(
            approval_id=resolved_id,
            request_id=request_id,
            created_at=created_at,
            summary=summary,
            chat_request=chat_request,
            conversation_id=conversation_id,
        )
        self._pending[resolved_id] = record
        return record

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._pending.get(approval_id)

    def take(self, approval_id: str) -> PendingApproval | None:
        return self._pending.pop(approval_id, None)


def truncate_summary(message: str, max: int = 240) -> str:
    trimmed = message.strip()
    if len(trimmed) <= max:
        return trimmed
    return f"{trimmed[: max - 1]}…"
