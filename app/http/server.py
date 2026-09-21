"""API FastAPI do OpsPilot (equivalente a ``src/http/server.ts``).

Nota de fidelidade: o payload JSON usa ``snake_case`` (convenção Python/FastAPI)
onde o original TS usava ``camelCase`` — é a única divergência de contrato de
API proposital desta conversão.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.agents.model import OpsChatModel
from app.agents.registry import create_registry, list_strategies, resolve_strategy
from app.chat.history_summarizer import ConversationSummarizer
from app.domain.errors import (
    ApprovalNotFoundError,
    ChatTimeoutError,
    ConversationNotFoundError,
    EmbeddingError,
    InvalidMemoryInputError,
    ModelUnavailableError,
    RequestNotFoundError,
    UnknownStrategyError,
)
from app.domain.types import (
    ApprovalStore,
    ChatRequestSnapshot,
    ConversationStore,
    Logger,
    MemoryStore,
    RequestStore,
)
from app.graph.production_graph import (
    ProductionGraphDeps,
    ProductionStrategies,
    ProductionTurnInput,
    run_production_turn,
)
from app.graph.router import ProductionRoute, parse_override_strategy
from app.http.chat_schema import (
    ApprovalDecisionBody,
    ChatRequest,
    RememberRequest,
    StatsQuery,
)
from app.http.cors import CorsMiddleware, resolve_cors_origins
from app.memory.learning_reflector import LearningReflectorFn
from app.obs.request_stats import parse_since_duration
from app.store.memory_approval_store import truncate_summary

__all__ = ["create_registry", "list_strategies", "resolve_strategy", "ChatAppDeps", "create_app"]

_DEFAULT_TIMEOUT_MS = 180_000


@dataclass(slots=True)
class ChatAppDeps:
    conversations: ConversationStore
    memories: MemoryStore
    strategies: ProductionStrategies
    #: Roteador determinístico para testes; produção usa route_model_factory.
    classify_route: object | None = None
    route_model_factory: Callable[[], OpsChatModel] | None = None
    timeout_ms: int = _DEFAULT_TIMEOUT_MS
    learning_reflector: LearningReflectorFn | None = None
    summarizer: ConversationSummarizer | None = None
    budgets: dict[str, float] | None = None
    requests: RequestStore | None = None
    logger: Logger | None = None
    approvals: ApprovalStore | None = None
    #: Sobrescreve allowlist de CORS; default permite todas as origens.
    cors_origins: list[str] | str | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _resolve_override_route(strategy: str | None, reflect: bool) -> ProductionRoute | None:
    if strategy is not None:
        parsed = parse_override_strategy(strategy)
        if not parsed:
            raise UnknownStrategyError(strategy)
        return parsed
    return "reflect" if reflect else None


def _error_code(error: Exception) -> str:
    if isinstance(error, UnknownStrategyError):
        return "unknown_strategy"
    if isinstance(error, ConversationNotFoundError):
        return "conversation_not_found"
    if isinstance(error, ChatTimeoutError):
        return "timeout"
    if isinstance(error, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(error, RequestNotFoundError):
        return "request_not_found"
    if isinstance(error, ApprovalNotFoundError):
        return "approval_not_found"
    if isinstance(error, (ValidationError, InvalidMemoryInputError)):
        return "validation_error"
    if isinstance(error, EmbeddingError):
        return "internal_error"
    return "internal_error"


def _metrics_dict(metrics) -> dict:
    return {k: v for k, v in asdict(metrics).items() if v is not None}


def _trace_dict(events) -> list[dict]:
    return [{k: v for k, v in asdict(e).items() if v is not None} for e in events]


def create_app(deps: ChatAppDeps) -> FastAPI:
    timeout_s = deps.timeout_ms / 1000
    app = FastAPI(title="OpsPilot")
    app.add_middleware(CorsMiddleware, origins=deps.cors_origins or resolve_cors_origins())

    async def execute_turn(
        message: str,
        user_id: str,
        conversation_id: str | None,
        strategy: str | None,
        reflect: bool,
        request_id: str,
        request_created_at: int,
    ):
        override_route = _resolve_override_route(strategy, reflect)
        turn = run_production_turn(
            ProductionGraphDeps(
                conversations=deps.conversations,
                memories=deps.memories,
                strategies=deps.strategies,
                classify_route=deps.classify_route,  # type: ignore[arg-type]
                route_model_factory=deps.route_model_factory,
                learning_reflector=deps.learning_reflector,
                summarizer=deps.summarizer,
                budgets=deps.budgets,
                requests=deps.requests,
                logger=deps.logger,
            ),
            ProductionTurnInput(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id,
                override_route=override_route,
                request_id=request_id,
                request_created_at=request_created_at,
            ),
        )
        try:
            return await asyncio.wait_for(turn, timeout=timeout_s)
        except TimeoutError as error:
            raise ChatTimeoutError(deps.timeout_ms) from error

    @app.post("/chat")
    async def chat(request: Request):
        request_id = str(uuid.uuid4())
        request_created_at = _now_ms()
        if deps.logger:
            deps.logger.info("chat_request_start", {"requestId": request_id})

        try:
            body = await request.json()
        except Exception:
            body = None

        try:
            parsed = ChatRequest.model_validate(body or {})
        except ValidationError as error:
            strategy_issue = next(
                (i for i in error.errors() if i.get("loc") and i["loc"][0] == "strategy"),
                None,
            )
            if strategy_issue and isinstance((body or {}).get("strategy"), str):
                if deps.logger:
                    deps.logger.warn(
                        "chat_request_error",
                        {
                            "requestId": request_id,
                            "httpStatus": 422,
                            "errorCode": "unknown_strategy",
                        },
                    )
                return JSONResponse(
                    status_code=422,
                    content={"error": "unknown_strategy", "strategy": body["strategy"]},
                    headers={"X-Request-Id": request_id},
                )
            if deps.logger:
                deps.logger.warn(
                    "chat_request_error",
                    {"requestId": request_id, "httpStatus": 400, "errorCode": "validation_error"},
                )
            return JSONResponse(
                status_code=400,
                content={"error": "validation_error", "issues": error.errors()},
                headers={"X-Request-Id": request_id},
            )

        try:
            if parsed.await_human_approval:
                if not deps.approvals:
                    return JSONResponse(
                        status_code=503,
                        content={"error": "approvals_unavailable"},
                        headers={"X-Request-Id": request_id},
                    )

                pending = deps.approvals.save(
                    request_id=request_id,
                    created_at=_now_ms(),
                    summary=truncate_summary(parsed.message),
                    conversation_id=parsed.conversation_id,
                    chat_request=ChatRequestSnapshot(
                        message=parsed.message,
                        user_id=parsed.user_id,
                        strategy=parsed.strategy,
                        reflect=parsed.reflect,
                        conversation_id=parsed.conversation_id,
                    ),
                )
                if deps.logger:
                    deps.logger.info(
                        "chat_request_end",
                        {
                            "requestId": request_id,
                            "httpStatus": 202,
                            "approvalId": pending.approval_id,
                        },
                    )
                return JSONResponse(
                    status_code=202,
                    content={
                        "requestId": request_id,
                        "conversationId": parsed.conversation_id,
                        "pending": {
                            "approvalId": pending.approval_id,
                            "summary": pending.summary,
                            "createdAt": pending.created_at,
                        },
                    },
                    headers={"X-Request-Id": request_id},
                )

            result = await execute_turn(
                parsed.message,
                parsed.user_id,
                parsed.conversation_id,
                parsed.strategy,
                parsed.reflect,
                request_id,
                request_created_at,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "requestId": request_id,
                    "answer": result.answer,
                    "trace": _trace_dict(result.trace),
                    "metrics": _metrics_dict(result.metrics),
                    "conversationId": result.conversation_id,
                },
                headers={"X-Request-Id": request_id},
            )
        except Exception as error:  # noqa: BLE001
            if deps.logger:
                deps.logger.error(
                    "chat_request_error", {"requestId": request_id, "errorCode": _error_code(error)}
                )
            return _error_response(error, request_id)

    @app.post("/approvals/{approval_id}")
    async def decide_approval(approval_id: str, request: Request):
        request_id = str(uuid.uuid4())
        request_created_at = _now_ms()
        try:
            uuid.UUID(approval_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "validation_error", "issues": [{"msg": "invalid approvalId"}]},
                headers={"X-Request-Id": request_id},
            )

        try:
            body = ApprovalDecisionBody.model_validate(await request.json())
        except ValidationError as error:
            return JSONResponse(
                status_code=400,
                content={"error": "validation_error", "issues": error.errors()},
                headers={"X-Request-Id": request_id},
            )

        try:
            if not deps.approvals:
                raise ApprovalNotFoundError(approval_id)
            pending = deps.approvals.take(approval_id)
            if not pending:
                raise ApprovalNotFoundError(approval_id)

            if body.user_id != pending.chat_request.user_id and deps.logger:
                deps.logger.warn(
                    "approval_user_mismatch",
                    {
                        "requestId": request_id,
                        "approvalId": pending.approval_id,
                        "priorRequestId": pending.request_id,
                    },
                )

            if body.decision == "deny":
                answer = "Ação cancelada pelo plantonista."
                from app.domain.types import ExecutionMetrics, TraceEvent

                return JSONResponse(
                    status_code=200,
                    content={
                        "requestId": request_id,
                        "answer": answer,
                        "trace": _trace_dict(
                            [TraceEvent(type="answer", content=answer, node="approval")]
                        ),
                        "metrics": _metrics_dict(
                            ExecutionMetrics(
                                llm_calls=0, latency_ms=_now_ms() - request_created_at
                            )
                        ),
                        "conversationId": pending.conversation_id,
                    },
                    headers={"X-Request-Id": request_id},
                )

            if deps.logger:
                deps.logger.info(
                    "approval_approve",
                    {
                        "requestId": request_id,
                        "approvalId": pending.approval_id,
                        "priorRequestId": pending.request_id,
                    },
                )

            snap = pending.chat_request
            result = await execute_turn(
                snap.message,
                snap.user_id,
                snap.conversation_id,
                snap.strategy,
                snap.reflect,
                request_id,
                request_created_at,
            )
            return JSONResponse(
                status_code=200,
                content={
                    "requestId": request_id,
                    "answer": result.answer,
                    "trace": _trace_dict(result.trace),
                    "metrics": _metrics_dict(result.metrics),
                    "conversationId": result.conversation_id,
                },
                headers={"X-Request-Id": request_id},
            )
        except Exception as error:  # noqa: BLE001
            return _error_response(error, request_id)

    @app.get("/requests/{request_id}")
    async def get_request(request_id: str):
        try:
            uuid.UUID(request_id)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "validation_error", "issues": [{"msg": "invalid id"}]},
            )
        try:
            if not deps.requests:
                raise RequestNotFoundError(request_id)
            found = deps.requests.get_by_id(request_id)
            if not found:
                raise RequestNotFoundError(request_id)
            record, trace = found
            return JSONResponse(
                status_code=200,
                content={
                    "request": asdict(record),
                    "trace": _trace_dict(trace),
                },
            )
        except Exception as error:  # noqa: BLE001
            return _error_response(error, None)

    @app.get("/stats")
    async def stats(since: str = "24h"):
        try:
            parsed = StatsQuery(since=since)
        except ValidationError as error:
            return JSONResponse(
                status_code=400, content={"error": "validation_error", "issues": error.errors()}
            )

        window_ms = parse_since_duration(parsed.since)
        if window_ms is None:
            return JSONResponse(
                status_code=400,
                content={"error": "validation_error", "message": "invalid since duration"},
            )

        if not deps.requests:
            return JSONResponse(
                status_code=200,
                content={
                    "since": parsed.since,
                    "total": 0,
                    "errors": 0,
                    "tokens": 0,
                    "costUsd": 0,
                    "latency": {"p50": None, "p95": None},
                    "byRoute": {},
                    "byModel": {},
                },
            )

        since_ms = _now_ms() - window_ms
        summary = deps.requests.stats(since_ms)
        return JSONResponse(
            status_code=200,
            content={
                "since": parsed.since,
                "total": summary.total,
                "errors": summary.errors,
                "tokens": summary.tokens,
                "costUsd": summary.cost_usd,
                "latency": summary.latency,
                "byRoute": {k: asdict(v) for k, v in summary.by_route.items()},
                "byModel": {k: asdict(v) for k, v in summary.by_model.items()},
            },
        )

    @app.post("/memories")
    async def remember(request: Request):
        try:
            body = RememberRequest.model_validate(await request.json())
        except ValidationError as error:
            return JSONResponse(
                status_code=400, content={"error": "validation_error", "issues": error.errors()}
            )
        try:
            result = await deps.memories.remember(body.user_id, body.fact)
            return JSONResponse(
                status_code=201 if result.stored else 200,
                content={
                    "id": result.id,
                    "stored": result.stored,
                    "userId": body.user_id,
                    "fact": body.fact.strip(),
                },
            )
        except Exception as error:  # noqa: BLE001
            return _error_response(error, None)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400, content={"error": "validation_error", "issues": exc.errors()}
        )

    return app


def _error_response(error: Exception, request_id: str | None) -> JSONResponse:
    headers = {"X-Request-Id": request_id} if request_id else None

    if isinstance(error, UnknownStrategyError):
        return JSONResponse(
            status_code=422,
            content={"error": "unknown_strategy", "strategy": error.strategy},
            headers=headers,
        )
    if isinstance(error, ConversationNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "conversation_not_found", "conversationId": error.conversation_id},
            headers=headers,
        )
    if isinstance(error, RequestNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "request_not_found", "requestId": error.request_id},
            headers=headers,
        )
    if isinstance(error, ApprovalNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "approval_not_found", "approvalId": error.approval_id},
            headers=headers,
        )
    if isinstance(error, EmbeddingError):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": str(error)},
            headers=headers,
        )
    if isinstance(error, InvalidMemoryInputError):
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "message": str(error)},
            headers=headers,
        )
    if isinstance(error, ChatTimeoutError):
        return JSONResponse(
            status_code=504, content={"error": "timeout", "message": str(error)}, headers=headers
        )
    if isinstance(error, ModelUnavailableError):
        return JSONResponse(
            status_code=503,
            content={"error": "model_unavailable", "message": str(error)},
            headers=headers,
        )
    return JSONResponse(
        status_code=500, content={"error": "internal_error", "message": str(error)}, headers=headers
    )
