"""Estratégia plan-and-execute sobre LangGraph (planner → executor → replanner)
(equivalente a ``src/strategies/plan-execute.ts``)."""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.agents.model import OpsChatModel
from app.agents.system_prompt import OPSPILOT_SYSTEM_PROMPT
from app.chat.run_chat import format_history_for_prompt
from app.context.tokens import sum_prompt_tokens_from_messages
from app.domain.types import (
    ExecutionMetrics,
    ReasoningStrategy,
    StrategyResult,
    StrategyRunInput,
    TraceEvent,
)
from app.graph.stamp_node import stamp_node

_MAX_STEPS = 8


def _now_ms() -> int:
    return int(time.time() * 1000)


class _Plan(BaseModel):
    steps: list[str] = Field(
        min_length=1,
        max_length=_MAX_STEPS,
        description="passos curtos, ordenados, executaveis com as ferramentas disponiveis",
    )


class _Replan(BaseModel):
    """Schema plano (não discriminated union) — modelos via OpenRouter quebram
    facilmente com uniões discriminadas."""

    decision: str = Field(description="adjust | continue | finish")
    steps: list[str] | None = Field(
        default=None,
        max_length=_MAX_STEPS,
        description="obrigatório quando decision=adjust",
    )
    answer: str | None = Field(
        default=None, description="obrigatório quando decision=finish"
    )


class _PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    done: Annotated[list[tuple[str, str]], operator.add]
    answer: str
    trace: Annotated[list[TraceEvent], operator.add]
    iterations: int
    llm_calls: int
    prompt_tokens: int
    prompt_token_hits: int


def _format_plan(steps: list[str]) -> str:
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def _done_as_text(done: list[tuple[str, str]]) -> str:
    if not done:
        return "none"
    return "\n".join(
        f"{i + 1}. {step}\n   -> {result}" for i, (step, result) in enumerate(done)
    )


def _build_executor_user_message(
    original_input: str, done: list[tuple[str, str]], current_step: str
) -> str:
    sections = [
        f"Pedido original (contexto — cumpra as restrições literais):\n{original_input}",
        "Regras: preserve nomes de serviço exatamente como no pedido; severity "
        "canônica do banco = critical|high|medium|low (sev1→critical, sev2→high, "
        "sev3→medium, sev4→low).",
    ]
    if done:
        sections.append(f"Progresso anterior (use IDs/dados já obtidos):\n{_done_as_text(done)}")
    sections.append(f"Passo atual a executar agora:\n{current_step}")
    return "\n\n".join(sections)


def _extract_answer(messages: list) -> str:
    for message in reversed(messages):
        if (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            return message.content.strip()
    return "Step executed without textual summary."


def _extract_action_observation_trace(messages: list) -> list[TraceEvent]:
    import json

    events: list[TraceEvent] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                args = call.get("args") or {}
                events.append(
                    TraceEvent(
                        type="action",
                        node="plan-and-execute",
                        content=f"{call.get('name')}({json.dumps(args)})",
                        tool=call.get("name"),
                        tool_args=args,
                    )
                )
            continue
        if isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content:
                events.append(
                    TraceEvent(type="observation", node="plan-and-execute", content=content)
                )
    return events


class PlanExecuteStrategy(ReasoningStrategy):
    name = "plan-and-execute"

    def __init__(
        self,
        model_factory: Callable[[], OpsChatModel],
        tools: list[StructuredTool],
        max_iterations: int,
        enable_replanner: bool = True,
    ) -> None:
        self._model_factory = model_factory
        self._tools = tools
        self._max_iterations = max_iterations
        self._enable_replanner = enable_replanner

    async def run(self, input: StrategyRunInput) -> StrategyResult:
        started_at = _now_ms()
        input_text = format_history_for_prompt(input.history, input.message)
        step_limit = min(_MAX_STEPS, self._max_iterations)
        # Instâncias de modelo frescas: create_react_agent vincula tools ao LLM e
        # não deve compartilhar a mesma instância usada por with_structured_output.
        planner_model = self._model_factory().with_structured_output(_Plan)
        replanner_model = self._model_factory().with_structured_output(_Replan)

        async def planner(state: _PlanExecuteState) -> dict:
            try:
                plan = await planner_model.ainvoke(
                    [
                        (
                            "system",
                            "Você é o planner operacional do OpsPilot. "
                            f"Produza no máximo {step_limit} passos curtos, ordenados e "
                            "executáveis com as tools disponíveis. Não invente ferramentas "
                            "fora de list_alerts, open_incident, resolve_incident, "
                            "list_incidents, consultar_runbook. Preserve literais do pedido "
                            "(nomes de serviço, IDs, ordem). Se o pedido falar sev1/sev2/etc, "
                            "nos passos use severity canônica do banco: "
                            "critical|high|medium|low (sev1→critical, sev2→high, "
                            "sev3→medium, sev4→low). Passos que dependem de dados "
                            "anteriores devem dizer para reutilizar IDs/resultados do "
                            "progresso.",
                        ),
                        ("user", state["input"]),
                    ]
                )
                parsed = plan if isinstance(plan, _Plan) else _Plan(**plan)
                steps = parsed.steps[:step_limit]
            except Exception:  # noqa: BLE001
                steps = [f"Execute o pedido com as tools disponíveis: {state['input']}"]

            return {
                "plan": steps,
                "trace": [
                    TraceEvent(type="plan", content=_format_plan(steps), node="plan-and-execute")
                ],
                "llm_calls": state["llm_calls"] + 1,
            }

        async def executor(state: _PlanExecuteState) -> dict:
            if not state["plan"] or state["iterations"] >= step_limit:
                return {}

            current_step, *remaining_plan = state["plan"]
            agent = create_react_agent(
                self._model_factory(), self._tools, prompt=OPSPILOT_SYSTEM_PROMPT
            )
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": _build_executor_user_message(
                                state["input"], state["done"], current_step
                            ),
                        }
                    ]
                },
                config={"recursion_limit": max(3, step_limit * 3)},
            )

            step_result = _extract_answer(result["messages"])
            step_trace = _extract_action_observation_trace(result["messages"])
            ai_messages = sum(1 for m in result["messages"] if isinstance(m, AIMessage))
            step_prompt_tokens = sum_prompt_tokens_from_messages(result["messages"])

            return {
                "plan": remaining_plan,
                "done": [(current_step, step_result)],
                "trace": step_trace,
                "iterations": state["iterations"] + 1,
                "llm_calls": state["llm_calls"] + ai_messages,
                "prompt_tokens": (
                    state["prompt_tokens"] + step_prompt_tokens
                    if step_prompt_tokens is not None
                    else state["prompt_tokens"]
                ),
                "prompt_token_hits": (
                    state["prompt_token_hits"] + 1
                    if step_prompt_tokens is not None
                    else state["prompt_token_hits"]
                ),
            }

        def finish_from_done(state: _PlanExecuteState, prefix: str | None = None) -> str:
            if state["done"]:
                body = _done_as_text(state["done"])
                return f"{prefix}\n{body}" if prefix else (
                    f"Execução concluída com {len(state['done'])} passo(s):\n{body}"
                )
            return prefix or "Nenhum passo executado."

        async def replanner(state: _PlanExecuteState) -> dict:
            if state["iterations"] >= step_limit:
                answer = finish_from_done(
                    state, f"Limite de {step_limit} passos atingido. Progresso:"
                )
                return {
                    "answer": answer,
                    "trace": [
                        TraceEvent(type="answer", content=answer, node="plan-and-execute")
                    ],
                }

            try:
                raw = await replanner_model.ainvoke(
                    [
                        (
                            "system",
                            "Você é o replanner operacional do OpsPilot. Decida "
                            "estritamente entre: adjust, continue, finish. Escolha finish "
                            "quando o pedido original já puder ser respondido com o "
                            "progresso (preencha answer). A answer de finish deve responder "
                            "ao pedido original de forma completa (números, serviços, "
                            "status) — não apenas listar passos. A answer de finish DEVE "
                            "seguir o formato OpsPilot: **Resumo**, depois **Achados** "
                            "(lista com -), depois **Próximos passos** (máx. 3 ou Nenhum); "
                            "sem emojis de decoração. Escolha adjust quando o plano "
                            "restante precisar de correção (preencha steps com literais "
                            "preservados). Escolha continue quando o plano atual ainda "
                            "estiver válido.",
                        ),
                        (
                            "user",
                            "\n\n".join(
                                [
                                    f"Pedido original: {state['input']}",
                                    f"Passos concluídos ({len(state['done'])}):\n"
                                    f"{_done_as_text(state['done'])}",
                                    f"Plano restante ({len(state['plan'])}):\n"
                                    f"{_format_plan(state['plan']) or 'nenhum'}",
                                ]
                            ),
                        ),
                    ]
                )
                decision = raw if isinstance(raw, _Replan) else _Replan(**raw)
            except Exception:  # noqa: BLE001
                answer = finish_from_done(state)
                return {
                    "answer": answer,
                    "trace": [
                        TraceEvent(type="answer", content=answer, node="plan-and-execute")
                    ],
                    "llm_calls": state["llm_calls"] + 1,
                }

            if decision.decision == "finish":
                answer = (decision.answer or "").strip() or finish_from_done(state)
                return {
                    "answer": answer,
                    "trace": [
                        TraceEvent(type="answer", content=answer, node="plan-and-execute")
                    ],
                    "llm_calls": state["llm_calls"] + 1,
                }

            if decision.decision == "adjust":
                remaining_budget = step_limit - state["iterations"]
                revised_plan = [
                    s for s in (decision.steps or []) if s.strip()
                ][:remaining_budget]
                if not revised_plan:
                    answer = finish_from_done(state)
                    return {
                        "answer": answer,
                        "trace": [
                            TraceEvent(type="answer", content=answer, node="plan-and-execute")
                        ],
                        "llm_calls": state["llm_calls"] + 1,
                    }
                return {
                    "plan": revised_plan,
                    "trace": [
                        TraceEvent(
                            type="critique",
                            content=_format_plan(revised_plan),
                            node="plan-and-execute",
                        )
                    ],
                    "llm_calls": state["llm_calls"] + 1,
                }

            if not state["plan"]:
                answer = finish_from_done(state)
                return {
                    "answer": answer,
                    "trace": [
                        TraceEvent(type="answer", content=answer, node="plan-and-execute")
                    ],
                    "llm_calls": state["llm_calls"] + 1,
                }

            return {
                "trace": [
                    TraceEvent(
                        type="critique",
                        content="Plano atual segue válido; continuar execução.",
                        node="plan-and-execute",
                    )
                ],
                "llm_calls": state["llm_calls"] + 1,
            }

        async def finish_without_replanner(state: _PlanExecuteState) -> dict:
            answer = finish_from_done(state)
            return {
                "answer": answer,
                "trace": [TraceEvent(type="answer", content=answer, node="plan-and-execute")],
            }

        builder = StateGraph(_PlanExecuteState)
        builder.add_node("planner", planner)
        builder.add_node("executor", executor)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")

        if self._enable_replanner:
            builder.add_node("replanner", replanner)
            builder.add_edge("executor", "replanner")
            builder.add_conditional_edges(
                "replanner",
                lambda s: END if (s["answer"] or s["iterations"] >= step_limit) else "executor",
                {"executor": "executor", END: END},
            )
        else:
            builder.add_node("finish", finish_without_replanner)
            builder.add_conditional_edges(
                "executor",
                lambda s: (
                    "finish"
                    if (not s["plan"] or s["iterations"] >= step_limit)
                    else "executor"
                ),
                {"executor": "executor", "finish": "finish"},
            )
            builder.add_edge("finish", END)

        graph = builder.compile()

        result = await graph.ainvoke(
            {
                "input": input_text,
                "plan": [],
                "done": [],
                "answer": "",
                "trace": [],
                "iterations": 0,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "prompt_token_hits": 0,
            },
            config={"recursion_limit": max(10, step_limit * 5)},
        )

        raw_trace = result["trace"] or [
            TraceEvent(type="answer", content="No answer generated.", node=self.name)
        ]
        trace = stamp_node(self.name, raw_trace)
        answers = [e for e in trace if e.type == "answer"]
        answer = result["answer"] or (answers[-1].content if answers else "No answer generated.")
        prompt_tokens = result["prompt_tokens"] if result["prompt_token_hits"] > 0 else None

        return StrategyResult(
            answer=answer,
            trace=trace,
            metrics=ExecutionMetrics(
                llm_calls=result["llm_calls"],
                latency_ms=_now_ms() - started_at,
                prompt_tokens=prompt_tokens,
            ),
        )
