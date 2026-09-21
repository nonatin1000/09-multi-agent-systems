"""Camada de reflexão (crítico) sobre qualquer ReasoningStrategy
(equivalente a ``src/strategies/reflect.ts``)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agents.model import OpsChatModel
from app.context.tokens import add_prompt_tokens
from app.domain.types import (
    ExecutionMetrics,
    ReasoningStrategy,
    StrategyResult,
    StrategyRunInput,
    TraceEvent,
)
from app.graph.stamp_node import stamp_node


def _now_ms() -> int:
    return int(time.time() * 1000)


class CritiqueResult(BaseModel):
    approved: bool
    feedback: str = Field(
        description="se não aprovado: o que corrigir, em específico e acionável"
    )


#: Callable crítico. ``original_input`` é o pedido do usuário para o crítico
#: julgar a resposta contra o pedido + observações do trace.
CriticFn = Callable[[str, list[TraceEvent], str], Awaitable[CritiqueResult]]

_CRITIC_PROMPT = (
    "Você é um crítico rigoroso de respostas de um agente de operações. "
    "Avalie APENAS contra as observações do trace e o pedido original. "
    "Não invente fatos que não apareçam nas observações. "
    "Se a resposta estiver completa e fiel às observações, approved=true. "
    "Se faltar informação exigida pelo pedido ou houver inconsistência com as "
    "observações, approved=false e descreva o que corrigir de forma específica "
    "e acionável."
)


def _observations_of(trace: list[TraceEvent]) -> str:
    observations = [e.content for e in trace if e.type == "observation"]
    return "\n".join(observations) if observations else "(nenhuma)"


def enrich_input_with_feedback(original_input: str, round: int, feedback: str) -> str:
    feedback_body = feedback if feedback.strip() else "(sem feedback adicional)"
    return (
        f"[Critique - Round {round}]:\n{feedback_body}\n\n"
        f"Original request:\n{original_input}"
    )


def create_llm_critic(model_factory: Callable[[], OpsChatModel]) -> CriticFn:
    async def _critic(
        answer: str, trace: list[TraceEvent], original_input: str
    ) -> CritiqueResult:
        try:
            raw = await model_factory().with_structured_output(CritiqueResult).ainvoke(
                [
                    ("system", _CRITIC_PROMPT),
                    (
                        "user",
                        f"Pedido: {original_input}\n"
                        f"Observações: {_observations_of(trace)}\nResposta: {answer}",
                    ),
                ]
            )
            return raw if isinstance(raw, CritiqueResult) else CritiqueResult(**raw)
        except Exception:  # noqa: BLE001
            # FR-012: fail-safe — trata saída inválida do crítico como aprovação.
            return CritiqueResult(approved=True, feedback="")

    return _critic


@dataclass(slots=True)
class ReflectionOpts:
    max_reflections: int = 2
    critic: CriticFn | None = None
    model_factory: Callable[[], OpsChatModel] | None = None


def _resolve_critic(opts: ReflectionOpts, max_reflections: int) -> CriticFn | None:
    if opts.critic:
        return opts.critic
    if max_reflections > 0 and opts.model_factory:
        return create_llm_critic(opts.model_factory)
    return None


def _critique_event_content(feedback: str) -> str:
    return feedback if feedback.strip() else "[Crítico aprovou sem feedback]"


class _ReflectingStrategy(ReasoningStrategy):
    def __init__(self, strategy: ReasoningStrategy, opts: ReflectionOpts) -> None:
        self.name = f"reflect:{strategy.name}"
        self._strategy = strategy
        max_reflections = opts.max_reflections
        self._critic = _resolve_critic(opts, max_reflections)
        self._effective_max = 0 if self._critic is None else max_reflections

    async def run(self, input: StrategyRunInput) -> StrategyResult:
        started_at = _now_ms()
        total_llm_calls = 0
        critic_call_count = 0
        total_prompt_tokens: int | None = None
        accumulated_trace: list[TraceEvent] = []

        current_result = await self._strategy.run(input)
        total_llm_calls += current_result.metrics.llm_calls
        total_prompt_tokens = add_prompt_tokens(
            total_prompt_tokens, current_result.metrics.prompt_tokens
        )
        accumulated_trace.extend(current_result.trace)

        if self._effective_max == 0 or self._critic is None:
            return StrategyResult(
                answer=current_result.answer,
                trace=stamp_node(self.name, accumulated_trace),
                metrics=ExecutionMetrics(
                    llm_calls=total_llm_calls,
                    latency_ms=_now_ms() - started_at,
                    prompt_tokens=total_prompt_tokens,
                ),
            )

        for round_ in range(1, self._effective_max + 1):
            critique = await self._critic(
                current_result.answer, current_result.trace, input.message
            )
            critic_call_count += 1

            accumulated_trace.append(
                TraceEvent(
                    type="critique",
                    content=_critique_event_content(critique.feedback),
                    node=self.name,
                    round=round_,
                    approved=critique.approved,
                    timestamp_ms=_now_ms(),
                )
            )

            if critique.approved:
                break

            enriched_input = StrategyRunInput(
                message=enrich_input_with_feedback(
                    input.message, round_, critique.feedback
                ),
                history=input.history,
            )
            current_result = await self._strategy.run(enriched_input)
            total_llm_calls += current_result.metrics.llm_calls
            total_prompt_tokens = add_prompt_tokens(
                total_prompt_tokens, current_result.metrics.prompt_tokens
            )
            accumulated_trace.extend(current_result.trace)

        return StrategyResult(
            answer=current_result.answer,
            trace=stamp_node(self.name, accumulated_trace),
            metrics=ExecutionMetrics(
                llm_calls=total_llm_calls + critic_call_count,
                latency_ms=_now_ms() - started_at,
                prompt_tokens=total_prompt_tokens,
            ),
        )


def with_reflection(
    strategy: ReasoningStrategy, opts: ReflectionOpts | None = None
) -> ReasoningStrategy:
    return _ReflectingStrategy(strategy, opts or ReflectionOpts())
