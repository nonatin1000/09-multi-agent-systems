"""Tipos de domínio do OpsPilot — dataclasses + Protocols (equivalente aos
`interface`/`type` do TS original em ``src/domain/types.ts``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

AlertStatus = Literal["firing", "resolved"]
IncidentStatus = Literal["open", "resolved"]
Severity = Literal["critical", "high", "medium", "low"]
ServiceTier = Literal["critical", "high", "standard"]
TraceEventType = Literal[
    "thought",
    "action",
    "observation",
    "plan",
    "critique",
    "answer",
    "summarize",
    "route",
    "fallback",
    "handoff",
]
ConversationMessageRole = Literal["user", "assistant"]
RequestStatus = Literal["success", "error"]
LogLevel = Literal["info", "warn", "error"]
ApprovalDecisionValue = Literal["approve", "deny"]

# Metadados escalares apenas (sem payloads de message/answer/trace).
LogMeta = dict[str, str | int | float | bool | None]


@dataclass(slots=True)
class Service:
    name: str
    tier: ServiceTier


@dataclass(slots=True)
class Alert:
    id: str
    service: str
    description: str
    severity: Severity
    status: AlertStatus


@dataclass(slots=True)
class Incident:
    id: str
    title: str
    service: str
    severity: Severity
    status: IncidentStatus
    created_at: int
    resolved_at: int | None = None
    summary: str | None = None


@dataclass(slots=True)
class Runbook:
    service: str
    content: str


@dataclass(slots=True)
class SeedPayload:
    services: list[Service]
    alerts: list[Alert]
    runbooks: list[Runbook]


@dataclass(slots=True)
class TraceEvent:
    type: TraceEventType
    content: str
    node: str
    tool: str | None = None
    tool_args: dict[str, object] | None = None
    round: int | None = None
    approved: bool | None = None
    timestamp_ms: int | None = None
    route: str | None = None
    override: bool | None = None
    reason: str | None = None
    to: str | None = None

    def replace_node(self, node: str) -> TraceEvent:
        """Retorna uma cópia carimbada com outro node (equivalente a stampNode)."""
        from dataclasses import replace

        return replace(self, node=node)


@dataclass(slots=True)
class ContextBreakdown:
    """Estimativa de contribuição do prompt por fonte (chars/4). Sempre 5 chaves."""

    system: int
    history: int
    memories: int
    message: int
    summary: int


@dataclass(slots=True)
class ExecutionMetrics:
    llm_calls: int
    latency_ms: int
    history_messages: int | None = None
    recalled_memories: int | None = None
    prompt_tokens: int | None = None
    context_breakdown: ContextBreakdown | None = None
    route: str | None = None
    route_reason: str | None = None
    model_used: str | None = None


@dataclass(slots=True)
class MemoryFact:
    id: str
    user_id: str
    fact: str
    created_at: int


@dataclass(slots=True)
class RecalledMemory:
    id: str
    fact: str
    score: float


@dataclass(slots=True)
class RememberResult:
    id: str
    stored: bool


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class MemoryStore(Protocol):
    async def remember(self, user_id: str, fact: str) -> RememberResult: ...
    async def recall(
        self, user_id: str, query: str, k: int = 3
    ) -> list[RecalledMemory]: ...
    async def forget(self, user_id: str, id: str) -> bool: ...


@dataclass(slots=True)
class ConversationMessage:
    id: str
    conversation_id: str
    role: ConversationMessageRole
    content: str
    created_at: int


@dataclass(slots=True)
class ConversationSummaryRecord:
    conversation_id: str
    text: str
    covered_count: int
    updated_at: int


class ConversationStore(Protocol):
    def create(self) -> str: ...
    def append(
        self, conversation_id: str, role: ConversationMessageRole, content: str
    ) -> ConversationMessage: ...
    def last_messages(
        self, conversation_id: str, limit: int
    ) -> list[ConversationMessage]: ...
    def count_messages(self, conversation_id: str) -> int: ...
    def messages_ascending(
        self, conversation_id: str, offset: int, limit: int
    ) -> list[ConversationMessage]: ...
    def get_summary(
        self, conversation_id: str
    ) -> ConversationSummaryRecord | None: ...
    def upsert_summary(
        self, conversation_id: str, text: str, covered_count: int
    ) -> None: ...


@dataclass(slots=True)
class StrategyRunInput:
    """Entrada de um turno de estratégia (mensagem atual + histórico anterior)."""

    message: str
    history: list[ConversationMessage] = field(default_factory=list)


@dataclass(slots=True)
class StrategyResult:
    answer: str
    trace: list[TraceEvent]
    metrics: ExecutionMetrics


class ReasoningStrategy(Protocol):
    name: str

    async def run(self, input: StrategyRunInput) -> StrategyResult: ...


class OpsStore(Protocol):
    def seed(self, data: SeedPayload) -> None: ...
    def get_alerts(self, status: AlertStatus | None = None) -> list[Alert]: ...
    def get_incidents(
        self, status: IncidentStatus | None = None
    ) -> list[Incident]: ...
    def create_incident(
        self, title: str, service: str, severity: Severity
    ) -> Incident: ...
    def resolve_incident(self, id: str, summary: str | None = None) -> Incident: ...
    def get_runbook(self, service: str) -> Runbook: ...


class Logger(Protocol):
    def info(self, event: str, meta: LogMeta | None = None) -> None: ...
    def warn(self, event: str, meta: LogMeta | None = None) -> None: ...
    def error(self, event: str, meta: LogMeta | None = None) -> None: ...


@dataclass(slots=True)
class RequestRecord:
    id: str
    created_at: int
    finished_at: int
    status: RequestStatus
    http_status: int
    conversation_id: str | None
    user_id: str | None
    metrics: ExecutionMetrics
    latency_ms: int | None
    llm_calls: int | None
    route: str | None
    model_used: str | None


@dataclass(slots=True)
class SaveRequestInput:
    id: str
    created_at: int
    finished_at: int
    status: RequestStatus
    http_status: int
    metrics: ExecutionMetrics
    trace: list[TraceEvent]
    conversation_id: str | None = None
    user_id: str | None = None


@dataclass(slots=True)
class RequestStatsBucket:
    total: int = 0
    errors: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class RequestStatsSummary:
    total: int
    errors: int
    tokens: int
    cost_usd: float
    latency: dict[str, float | None]
    by_route: dict[str, RequestStatsBucket]
    by_model: dict[str, RequestStatsBucket]


class RequestStore(Protocol):
    def save(self, input: SaveRequestInput) -> None: ...
    def get_by_id(
        self, id: str
    ) -> tuple[RequestRecord, list[TraceEvent]] | None: ...
    def stats(self, since_ms: int) -> RequestStatsSummary: ...


@dataclass(slots=True)
class ChatRequestSnapshot:
    """Snapshot de um /chat validado, adiado para aprovação humana."""

    message: str
    user_id: str
    strategy: str | None = None
    reflect: bool = False
    conversation_id: str | None = None


@dataclass(slots=True)
class PendingApproval:
    approval_id: str
    request_id: str
    created_at: int
    summary: str
    chat_request: ChatRequestSnapshot
    conversation_id: str | None


class ApprovalStore(Protocol):
    def save(
        self,
        request_id: str,
        created_at: int,
        summary: str,
        chat_request: ChatRequestSnapshot,
        conversation_id: str | None,
        approval_id: str | None = None,
    ) -> PendingApproval: ...
    def get(self, approval_id: str) -> PendingApproval | None: ...
    def take(self, approval_id: str) -> PendingApproval | None: ...
