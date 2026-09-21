"""Grafo do modo equipe: supervisor delega a analista/planejador/executor até
``done`` (equivalente a ``src/team/team-graph.ts``)."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain.types import TraceEvent
from app.team.blackboard import BlackboardEntry, render_blackboard
from app.team.roles import RoleRunner
from app.team.supervisor import DecideNextFn, TeamRole

#: Teto de domínio de delegações do supervisor por turno (spec 018, US5).
MAX_HANDOFFS = 8

#: Prefixos estáveis validados por testes e documentados em contracts/trace-handoff.md.
CAP_REACHED_PREFIX = "teto de handoffs atingido"
INVALID_DECISION_PREFIX = "decisão inválida do supervisor"


@dataclass(slots=True)
class TeamGraphDeps:
    decide_next: DecideNextFn
    role_runners: dict[TeamRole, RoleRunner]
    #: LLM calls consumidas por decisão real do supervisor (0 para fakes injetados).
    supervisor_llm_calls: int = 0


class _TeamState(TypedDict):
    message: str
    blackboard: Annotated[list[BlackboardEntry], operator.add]
    handoff_count: int
    brief: str
    next: str | None
    answer: str
    trace: Annotated[list[TraceEvent], operator.add]
    llm_calls: Annotated[int, operator.add]


def _handoff_event(to: str, content: str) -> TraceEvent:
    return TraceEvent(type="handoff", node="supervisor", to=to, content=content)


def create_team_graph(deps: TeamGraphDeps):
    supervisor_llm_calls = deps.supervisor_llm_calls

    async def supervisor_node(state: _TeamState) -> dict:
        if state["handoff_count"] >= MAX_HANDOFFS:
            content = f"{CAP_REACHED_PREFIX}: encerrando com o conteúdo do blackboard"
            return {"next": "done", "brief": "", "trace": [_handoff_event("done", content)]}

        try:
            decision = await deps.decide_next(
                state["message"], state["blackboard"], state["handoff_count"]
            )
        except Exception as error:  # noqa: BLE001
            content = f"{INVALID_DECISION_PREFIX}: {error}"
            return {
                "next": "done",
                "brief": "",
                "llm_calls": supervisor_llm_calls,
                "trace": [_handoff_event("done", content)],
            }

        is_delegation = decision.next != "done"
        return {
            "next": decision.next,
            "brief": decision.brief,
            "handoff_count": state["handoff_count"] + (1 if is_delegation else 0),
            "llm_calls": supervisor_llm_calls,
            "trace": [_handoff_event(decision.next, decision.brief)],
        }

    def make_role_node(role: TeamRole):
        async def _role_node(state: _TeamState) -> dict:
            from app.team.roles import RoleRunInput

            runner = deps.role_runners[role]
            try:
                result = await runner.run(
                    RoleRunInput(
                        message=state["message"],
                        brief=state["brief"],
                        blackboard=state["blackboard"],
                    )
                )
                return {
                    "blackboard": [result.entry],
                    "trace": result.trace,
                    "llm_calls": result.llm_calls,
                }
            except Exception as error:  # noqa: BLE001
                content = f"erro no papel {role}: {error}"
                return {
                    "blackboard": [
                        BlackboardEntry(
                            role=role, kind="error", brief=state["brief"], content=content
                        )
                    ],
                    "trace": [TraceEvent(type="observation", content=content, node=role)],
                }

        return _role_node

    async def done_node(state: _TeamState) -> dict:
        """done: brief carrega o resumo final do supervisor (contrato com o
        usuário); encerramentos forçados/anômalos chegam com brief vazio e caem
        para o blackboard."""
        brief = state["brief"].strip()
        if brief:
            answer = brief
        elif state["blackboard"]:
            answer = f"Resumo do blackboard:\n{render_blackboard(state['blackboard'])}"
        else:
            answer = f"A equipe encerrou sem contribuições no blackboard para: {state['message']}"
        return {
            "answer": answer,
            "trace": [TraceEvent(type="answer", content=answer, node="supervisor")],
        }

    builder = StateGraph(_TeamState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("analista", make_role_node("analista"))
    builder.add_node("planejador", make_role_node("planejador"))
    builder.add_node("executor", make_role_node("executor"))
    builder.add_node("done", done_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: s["next"] or "done",
        {
            "analista": "analista",
            "planejador": "planejador",
            "executor": "executor",
            "done": "done",
        },
    )
    builder.add_edge("analista", "supervisor")
    builder.add_edge("planejador", "supervisor")
    builder.add_edge("executor", "supervisor")
    builder.add_edge("done", END)
    return builder.compile()


@dataclass(slots=True)
class TeamTurnResult:
    answer: str
    trace: list[TraceEvent]
    llm_calls: int


async def run_team_graph(deps: TeamGraphDeps, message: str) -> TeamTurnResult:
    graph = create_team_graph(deps)
    final_state = await graph.ainvoke(
        {
            "message": message,
            "blackboard": [],
            "handoff_count": 0,
            "brief": "",
            "next": None,
            "answer": "",
            "trace": [],
            "llm_calls": 0,
        },
        # Dimensionado acima do teto para GraphRecursionError nunca disparar
        # antes do MAX_HANDOFFS.
        config={"recursion_limit": MAX_HANDOFFS * 3 + 10},
    )
    return TeamTurnResult(
        answer=final_state["answer"],
        trace=final_state["trace"],
        llm_calls=final_state["llm_calls"],
    )
