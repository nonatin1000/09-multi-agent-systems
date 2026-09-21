"""Grafo de produção do OpsPilot: contexto → roteador → estratégia → resposta
(equivalente a ``src/graph/production-graph.ts``)."""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.system_prompt import OPSPILOT_SYSTEM_PROMPT
from app.chat.history_summarizer import (
    HISTORY_LIMIT,
    ConversationSummarizer,
    maybe_summarize,
)
from app.context.context_builder import BuildContext, BuildInput, build_context
from app.context.tokens import build_context_breakdown
from app.domain.types import (
    ConversationStore,
    ExecutionMetrics,
    Logger,
    MemoryStore,
    ReasoningStrategy,
    RequestStore,
    SaveRequestInput,
    TraceEvent,
)
from app.graph.router import (
    ClassifyRouteFn,
    ProductionRoute,
    create_classify_route,
    is_production_route,
)
from app.graph.stamp_node import stamp_node
from app.llm.model_telemetry import (
    create_empty_telemetry,
    get_model_telemetry,
    run_with_model_telemetry,
)
from app.memory.chat_user_context import run_with_chat_user_async
from app.memory.learning_reflector import LearningReflectorFn, prepare_memories_for_turn


@dataclass(slots=True)
class ProductionStrategies:
    react: ReasoningStrategy
    plan_execute: ReasoningStrategy
    reflect: ReasoningStrategy
    team: ReasoningStrategy


@dataclass(slots=True)
class ProductionGraphDeps:
    conversations: ConversationStore
    memories: MemoryStore
    strategies: ProductionStrategies
    #: Injetado em testes; produção usa route_model_factory.
    classify_route: ClassifyRouteFn | None = None
    route_model_factory: object | None = None  # Callable[[], OpsChatModel]
    learning_reflector: LearningReflectorFn | None = None
    summarizer: ConversationSummarizer | None = None
    budgets: dict[str, float] | None = None
    execute: object | None = None  # Callable[[Awaitable], Awaitable]
    #: Store de auditoria opcional — persistido no nó "resposta".
    requests: RequestStore | None = None
    logger: Logger | None = None


@dataclass(slots=True)
class ProductionTurnInput:
    message: str
    user_id: str
    conversation_id: str | None = None
    #: Quando setado, roteador pula o LLM e marca override.
    override_route: ProductionRoute | None = None
    #: Id de correlação vindo do HTTP (X-Request-Id).
    request_id: str | None = None
    #: Epoch ms de quando o handler HTTP mintou o requestId.
    request_created_at: int | None = None


@dataclass(slots=True)
class ChatTurnResult:
    conversation_id: str
    answer: str
    trace: list[TraceEvent]
    metrics: ExecutionMetrics


def _now_ms() -> int:
    return int(time.time() * 1000)


class _GraphState(TypedDict):
    message: str
    user_id: str
    conversation_id: str
    request_id: str
    request_created_at: int
    override_route: ProductionRoute | None
    built: BuildContext | None
    route: ProductionRoute | None
    answer: str
    trace: Annotated[list[TraceEvent], operator.add]
    strategy_metrics: ExecutionMetrics | None
    router_llm_calls: int


def _resolve_classifier(deps: ProductionGraphDeps) -> ClassifyRouteFn:
    if deps.classify_route:
        return deps.classify_route
    if deps.route_model_factory:
        return create_classify_route(deps.route_model_factory)  # type: ignore[arg-type]

    async def _fallback(message: str):
        from app.graph.router import RouterDecision

        return RouterDecision(
            route="react", reason="fallback: no classifier configured; using react"
        )

    return _fallback


def _persist_turn_audit(
    deps: ProductionGraphDeps,
    state: _GraphState,
    trace: list[TraceEvent],
    metrics: ExecutionMetrics,
) -> None:
    if not deps.requests or not state["request_id"]:
        return
    try:
        deps.requests.save(
            SaveRequestInput(
                id=state["request_id"],
                created_at=state["request_created_at"] or _now_ms(),
                finished_at=_now_ms(),
                status="success",
                http_status=200,
                conversation_id=state["conversation_id"] or None,
                user_id=state["user_id"],
                metrics=metrics,
                trace=trace,
            )
        )
    except Exception as error:  # noqa: BLE001
        if deps.logger:
            deps.logger.error(
                "request_persist_failed",
                {
                    "requestId": state["request_id"],
                    "errorName": type(error).__name__,
                    "node": "resposta",
                },
            )


def create_production_graph(deps: ProductionGraphDeps):
    classify = _resolve_classifier(deps)

    async def context_node(state: _GraphState) -> dict:
        conversation_id = state["conversation_id"] or deps.conversations.create()

        trace: list[TraceEvent] = []
        if deps.summarizer:
            summarized = await maybe_summarize(
                deps.conversations, conversation_id, deps.summarizer
            )
            if summarized:
                trace.append(replace(summarized.event, node="contexto"))

        history = deps.conversations.last_messages(conversation_id, HISTORY_LIMIT)
        summary_record = deps.conversations.get_summary(conversation_id)
        recalled = await prepare_memories_for_turn(
            deps.memories, state["user_id"], state["message"], deps.learning_reflector
        )

        built = build_context(
            BuildInput(
                system=OPSPILOT_SYSTEM_PROMPT,
                summary=summary_record.text if summary_record else None,
                history=history,
                memories=recalled,
                message=state["message"],
            ),
            budgets=deps.budgets,
        )

        deps.conversations.append(conversation_id, "user", state["message"])

        return {"conversation_id": conversation_id, "built": built, "trace": trace}

    async def router_node(state: _GraphState) -> dict:
        if state["override_route"]:
            reason = "override from request"
            return {
                "route": state["override_route"],
                "router_llm_calls": 0,
                "trace": [
                    TraceEvent(
                        type="route",
                        node="roteador",
                        content=reason,
                        reason=reason,
                        route=state["override_route"],
                        override=True,
                    )
                ],
            }

        # Classificadores injetados são fakes — não inflam llm_calls.
        router_llm_calls = 0 if deps.classify_route else 1
        try:
            decision = await classify(state["message"])
            if not is_production_route(decision.route):
                from app.graph.router import RouterDecision

                decision = RouterDecision(
                    route="react", reason="fallback: invalid route; using react"
                )
        except Exception:  # noqa: BLE001
            from app.graph.router import RouterDecision

            decision = RouterDecision(
                route="react", reason="fallback: router failed; using react"
            )

        return {
            "route": decision.route,
            "router_llm_calls": router_llm_calls,
            "trace": [
                TraceEvent(
                    type="route",
                    node="roteador",
                    content=decision.reason,
                    reason=decision.reason,
                    route=decision.route,
                    override=False,
                )
            ],
        }

    async def run_strategy(
        state: _GraphState, strategy: ReasoningStrategy, node: str, stamp_trace: bool = True
    ) -> dict:
        from app.domain.types import StrategyRunInput

        built = state["built"]
        if not built:
            raise RuntimeError("production graph: missing built context")

        async def _run():
            return await strategy.run(
                StrategyRunInput(message=built.enriched_message, history=built.history)
            )

        result = await (deps.execute(_run()) if deps.execute else _run())  # type: ignore[misc]

        return {
            "answer": result.answer,
            "strategy_metrics": result.metrics,
            "trace": stamp_node(node, result.trace) if stamp_trace else result.trace,
        }

    async def react_node(state: _GraphState) -> dict:
        return await run_strategy(state, deps.strategies.react, "react")

    async def plan_execute_node(state: _GraphState) -> dict:
        return await run_strategy(state, deps.strategies.plan_execute, "planExecute")

    async def reflect_node(state: _GraphState) -> dict:
        return await run_strategy(state, deps.strategies.reflect, "reflect")

    async def team_node(state: _GraphState) -> dict:
        # Eventos de equipe são pré-assinados por papel (supervisor/analista/...)
        # — não recarimba.
        return await run_strategy(state, deps.strategies.team, "team", stamp_trace=False)

    async def answer_node(state: _GraphState) -> dict:
        """Resposta: grava histórico, persiste audit + log metadata.
        O aprendizado roda em prepare_memories_for_turn (aguarda para organizar;
        senão é adiado)."""
        deps.conversations.append(state["conversation_id"], "assistant", state["answer"])

        built = state["built"]
        strategy_metrics = state["strategy_metrics"] or ExecutionMetrics(
            llm_calls=0, latency_ms=0
        )
        tel = get_model_telemetry()
        model_used = (tel.model_used if tel else None) or (
            tel.primary_model if tel else "unknown"
        )

        extra_trace: list[TraceEvent] = []
        if tel and tel.fallback_used and tel.fallback_model:
            already = any(e.type == "fallback" for e in state["trace"])
            if not already:
                extra_trace.append(
                    TraceEvent(
                        type="fallback",
                        node="resposta",
                        content=f"{tel.primary_model} → {tel.fallback_model}",
                    )
                )

        full_trace = [*state["trace"], *extra_trace]
        route_event = next((e for e in full_trace if e.type == "route"), None)
        route = state["route"] or (route_event.route if route_event else "react")
        route_reason = (
            (route_event.reason if route_event else None)
            or (route_event.content if route_event else None)
            or "route unavailable"
        )

        metrics = ExecutionMetrics(
            llm_calls=strategy_metrics.llm_calls + (state["router_llm_calls"] or 0),
            latency_ms=strategy_metrics.latency_ms,
            prompt_tokens=strategy_metrics.prompt_tokens,
            route=route,
            route_reason=route_reason,
            model_used=model_used,
        )
        if built:
            metrics.history_messages = built.history_messages
            metrics.recalled_memories = built.recalled_memories
            metrics.context_breakdown = build_context_breakdown(
                system=built.system,
                history=built.history_text,
                memories=built.memories_text,
                message=built.message,
                summary=built.summary_text,
            )

        _persist_turn_audit(deps, state, full_trace, metrics)

        if deps.logger:
            deps.logger.info(
                "chat_request_end",
                {
                    "requestId": state["request_id"] or None,
                    "node": "resposta",
                    "type": "done",
                    "route": route,
                    "httpStatus": 200,
                    "latencyMs": metrics.latency_ms,
                    "llmCalls": metrics.llm_calls,
                    "promptTokens": metrics.prompt_tokens,
                    "traceEventCount": len(full_trace),
                    "modelUsed": model_used,
                },
            )

        return {"trace": extra_trace} if extra_trace else {}

    builder = StateGraph(_GraphState)
    builder.add_node("contexto", context_node)
    builder.add_node("roteador", router_node)
    builder.add_node("react", react_node)
    builder.add_node("planExecute", plan_execute_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("team", team_node)
    builder.add_node("resposta", answer_node)
    builder.add_edge(START, "contexto")
    builder.add_edge("contexto", "roteador")
    builder.add_conditional_edges(
        "roteador",
        lambda s: s["route"] or "react",
        {"react": "react", "planExecute": "planExecute", "reflect": "reflect", "team": "team"},
    )
    builder.add_edge("react", "resposta")
    builder.add_edge("planExecute", "resposta")
    builder.add_edge("reflect", "resposta")
    builder.add_edge("team", "resposta")
    builder.add_edge("resposta", END)
    return builder.compile()


async def run_production_turn(
    deps: ProductionGraphDeps, input: ProductionTurnInput
) -> ChatTurnResult:
    from app.agents.model import OpsResilientChatModel

    graph = create_production_graph(deps)

    # Prefere telemetria semeada a partir de route_model_factory quando é um
    # OpsResilientChatModel.
    seed_model = deps.route_model_factory() if deps.route_model_factory else None  # type: ignore[misc]
    if isinstance(seed_model, OpsResilientChatModel):
        telemetry = seed_model.create_telemetry()
    else:
        import os

        telemetry = create_empty_telemetry(
            os.environ.get("OPENROUTER_MODEL", "").strip() or "openai/gpt-4o-mini",
            os.environ.get("OPENROUTER_MODEL_FALLBACK", "").strip() or None,
        )

    async def _invoke():
        async def _with_user():
            return await graph.ainvoke(
                {
                    "message": input.message,
                    "user_id": input.user_id,
                    "conversation_id": input.conversation_id or "",
                    "request_id": input.request_id or "",
                    "request_created_at": input.request_created_at or _now_ms(),
                    "override_route": input.override_route,
                    "built": None,
                    "route": None,
                    "answer": "",
                    "trace": [],
                    "strategy_metrics": None,
                    "router_llm_calls": 0,
                }
            )

        return await run_with_chat_user_async(input.user_id, _with_user)

    final_state = await run_with_model_telemetry(telemetry, _invoke)

    built = final_state["built"]
    if not built:
        raise RuntimeError("production graph: turn finished without context")

    strategy_metrics = final_state["strategy_metrics"] or ExecutionMetrics(
        llm_calls=0, latency_ms=0
    )

    context_breakdown = build_context_breakdown(
        system=built.system,
        history=built.history_text,
        memories=built.memories_text,
        message=built.message,
        summary=built.summary_text,
    )

    route_event = next((e for e in final_state["trace"] if e.type == "route"), None)
    route = final_state["route"] or (route_event.route if route_event else "react")
    route_reason = (
        (route_event.reason if route_event else None)
        or (route_event.content if route_event else None)
        or "route unavailable"
    )

    tel = get_model_telemetry() or telemetry
    model_used = tel.model_used or tel.primary_model
    # Evento de fallback é carimbado em "resposta" quando a telemetria marca
    # reserva; mantém uma rede de segurança se "resposta" não rodou os extras.
    trace = final_state["trace"]
    if tel.fallback_used and tel.fallback_model:
        already = any(e.type == "fallback" for e in trace)
        if not already:
            trace = [
                *trace,
                TraceEvent(
                    type="fallback",
                    node="resposta",
                    content=f"{tel.primary_model} → {tel.fallback_model}",
                ),
            ]

    metrics = ExecutionMetrics(
        llm_calls=strategy_metrics.llm_calls + (final_state["router_llm_calls"] or 0),
        latency_ms=strategy_metrics.latency_ms,
        prompt_tokens=strategy_metrics.prompt_tokens,
        history_messages=built.history_messages,
        recalled_memories=built.recalled_memories,
        context_breakdown=context_breakdown,
        route=route,
        route_reason=route_reason,
        model_used=model_used,
    )

    return ChatTurnResult(
        conversation_id=final_state["conversation_id"],
        answer=final_state["answer"],
        trace=trace,
        metrics=metrics,
    )
