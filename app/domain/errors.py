"""Erros de domínio do OpsPilot (equivalente a ``src/domain/errors.ts``)."""

from __future__ import annotations


class IncidentNotFoundError(Exception):
    def __init__(self, id: str) -> None:
        super().__init__(f"Incident not found: {id}")
        self.id = id


class RunbookNotFoundError(Exception):
    def __init__(self, service: str) -> None:
        super().__init__(f"Runbook not found: {service}")
        self.service = service


class UnknownStrategyError(Exception):
    def __init__(self, strategy: str) -> None:
        super().__init__(f"Unknown strategy: {strategy}")
        self.strategy = strategy


class ChatTimeoutError(Exception):
    def __init__(self, timeout_ms: int) -> None:
        super().__init__(f"Chat timed out after {timeout_ms}ms")
        self.timeout_ms = timeout_ms


class ConversationNotFoundError(Exception):
    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation not found: {conversation_id}")
        self.conversation_id = conversation_id


class EmbeddingError(Exception):
    pass


class InvalidMemoryInputError(Exception):
    pass


class ModelUnavailableError(Exception):
    """Primário (+ fallback opcional) esgotados — HTTP 503 em /chat."""

    def __init__(self, message: str = "All configured language models failed") -> None:
        super().__init__(message)


class RequestNotFoundError(Exception):
    """Id de auditoria não encontrado — HTTP 404 em GET /requests/{id}."""

    code = "request_not_found"

    def __init__(self, request_id: str) -> None:
        super().__init__(f"Request not found: {request_id}")
        self.request_id = request_id


class ApprovalNotFoundError(Exception):
    """Aprovação pendente ausente ou já consumida — HTTP 404 em POST /approvals/{id}."""

    code = "approval_not_found"

    def __init__(self, approval_id: str) -> None:
        super().__init__(f"Approval not found: {approval_id}")
        self.approval_id = approval_id
