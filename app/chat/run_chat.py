"""Executa um turno de chat "standalone" (sem roteador de produção) —
persiste turno + roda a estratégia com histórico, resumo opcional e memória
semântica. Equivalente a ``src/chat/run-chat.ts``.

Fluxo: create/load → maybe_summarize → last_messages(8) → recall →
build_context → append user → run → append assistant → schedule_learning.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agents.system_prompt import OPSPILOT_SYSTEM_PROMPT
from app.chat.history_summarizer import (
    HISTORY_LIMIT,
    ConversationSummarizer,
    maybe_summarize,
)
from app.context.context_builder import (
    BuildInput,
    build_context,
    format_history_text,
    format_memories_for_prompt,
    format_memories_text,
)
from app.context.tokens import build_context_breakdown
from app.domain.types import (
    ConversationMessage,
    ConversationStore,
    ExecutionMetrics,
    MemoryStore,
    ReasoningStrategy,
    StrategyRunInput,
    TraceEvent,
)
from app.memory.chat_user_context import run_with_chat_user_async
from app.memory.learning_reflector import LearningReflectorFn, schedule_learning

__all__ = [
    "HISTORY_LIMIT",
    "format_memories_for_prompt",
    "format_history_text",
    "format_memories_text",
    "ChatInput",
    "ChatTurnResult",
    "RunChatOptions",
    "run_chat",
    "format_history_for_prompt",
]


@dataclass(slots=True)
class ChatInput:
    message: str
    user_id: str
    conversation_id: str | None = None


@dataclass(slots=True)
class ChatTurnResult:
    conversation_id: str
    answer: str
    trace: list[TraceEvent]
    metrics: ExecutionMetrics


@dataclass(slots=True)
class RunChatOptions:
    #: Envolve a promise da estratégia (ex.: timeout HTTP). Default: identidade.
    execute: Callable[[Awaitable], Awaitable] | None = None
    learning_reflector: LearningReflectorFn | None = None
    summarizer: ConversationSummarizer | None = None
    budgets: dict[str, float] | None = None


async def run_chat(
    conversations: ConversationStore,
    memories: MemoryStore,
    strategy: ReasoningStrategy,
    input: ChatInput,
    options: RunChatOptions | None = None,
) -> ChatTurnResult:
    options = options or RunChatOptions()
    conversation_id = input.conversation_id or conversations.create()

    summarize_event: TraceEvent | None = None
    if options.summarizer:
        summarized = await maybe_summarize(
            conversations, conversation_id, options.summarizer
        )
        if summarized:
            summarize_event = summarized.event

    history = conversations.last_messages(conversation_id, HISTORY_LIMIT)
    summary_record = conversations.get_summary(conversation_id)
    summary_text = summary_record.text if summary_record else None
    recalled = await memories.recall(input.user_id, input.message)

    built = build_context(
        BuildInput(
            system=OPSPILOT_SYSTEM_PROMPT,
            summary=summary_text,
            history=history,
            memories=recalled,
            message=input.message,
        ),
        budgets=options.budgets,
    )

    conversations.append(conversation_id, "user", input.message)

    async def _run_strategy():
        return await strategy.run(
            StrategyRunInput(message=built.enriched_message, history=built.history)
        )

    async def _turn():
        run_awaitable = _run_strategy()
        if options.execute:
            return await options.execute(run_awaitable)
        return await run_awaitable

    result = await run_with_chat_user_async(input.user_id, _turn)

    conversations.append(conversation_id, "assistant", result.answer)

    if options.learning_reflector:
        asyncio.ensure_future(
            schedule_learning(
                options.learning_reflector, memories, input.user_id, input.message
            )
        )

    context_breakdown = build_context_breakdown(
        system=built.system,
        history=built.history_text,
        memories=built.memories_text,
        message=built.message,
        summary=built.summary_text,
    )

    metrics = ExecutionMetrics(
        llm_calls=result.metrics.llm_calls,
        latency_ms=result.metrics.latency_ms,
        prompt_tokens=result.metrics.prompt_tokens,
        route=result.metrics.route,
        route_reason=result.metrics.route_reason,
        model_used=result.metrics.model_used,
        history_messages=built.history_messages,
        recalled_memories=built.recalled_memories,
        context_breakdown=context_breakdown,
    )

    trace = [summarize_event, *result.trace] if summarize_event else result.trace

    return ChatTurnResult(
        conversation_id=conversation_id,
        answer=result.answer,
        trace=trace,
        metrics=metrics,
    )


def format_history_for_prompt(
    history: list[ConversationMessage], current_message: str
) -> str:
    """Formata histórico para estratégias que ainda consomem um único blob de
    texto (ex.: plan-execute)."""
    if not history:
        return current_message
    lines = "\n".join(f"{m.role}: {m.content}" for m in history)
    return f"Previous conversation:\n{lines}\n\nCurrent message:\n{current_message}"
