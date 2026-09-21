"""Estratégia ReAct sobre ``langgraph.prebuilt.create_react_agent``
(equivalente a ``src/agents/react.ts``)."""

from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from app.agents.model import OpsChatModel
from app.agents.system_prompt import OPSPILOT_SYSTEM_PROMPT
from app.context.tokens import sum_prompt_tokens_from_messages
from app.domain.types import (
    ExecutionMetrics,
    ReasoningStrategy,
    StrategyResult,
    StrategyRunInput,
    TraceEvent,
)
from app.graph.stamp_node import stamp_node
from app.trace.builder import build_trace_from_messages


def _now_ms() -> int:
    return int(time.time() * 1000)


class ReactStrategy(ReasoningStrategy):
    name = "react"

    def __init__(
        self,
        model_factory: Callable[[], OpsChatModel],
        tools: list[StructuredTool],
        max_iterations: int,
    ) -> None:
        self._model_factory = model_factory
        self._tools = tools
        self._max_iterations = max_iterations

    async def run(self, input: StrategyRunInput) -> StrategyResult:
        started_at = _now_ms()
        model = self._model_factory()
        agent = create_react_agent(
            model, self._tools, prompt=OPSPILOT_SYSTEM_PROMPT
        )

        try:
            messages = [
                {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
                for m in input.history
            ]
            messages.append({"role": "user", "content": input.message})
            result = await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": max(3, self._max_iterations * 3)},
            )
            trace = build_trace_from_messages(result["messages"], self.name)
            answers = [e for e in trace if e.type == "answer"]
            answer = answers[-1].content if answers else "No answer generated."
            llm_calls = sum(1 for m in result["messages"] if isinstance(m, AIMessage))
            prompt_tokens = sum_prompt_tokens_from_messages(result["messages"])

            return StrategyResult(
                answer=answer,
                trace=trace,
                metrics=ExecutionMetrics(
                    llm_calls=llm_calls,
                    latency_ms=_now_ms() - started_at,
                    prompt_tokens=prompt_tokens,
                ),
            )
        except GraphRecursionError:
            answer = (
                f"[Iteration limit reached after {self._max_iterations} steps. "
                "Partial result unavailable.]"
            )
            return StrategyResult(
                answer=answer,
                trace=stamp_node(
                    self.name,
                    [TraceEvent(type="answer", content=answer, node=self.name)],
                ),
                metrics=ExecutionMetrics(
                    llm_calls=0, latency_ms=_now_ms() - started_at
                ),
            )
