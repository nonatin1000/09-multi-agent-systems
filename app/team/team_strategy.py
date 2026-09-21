"""Estratégia "team": delega a estratégia para o grafo multi-agente
(equivalente a ``src/team/team-strategy.ts``)."""

from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.tools import StructuredTool

from app.agents.model import OpsChatModel
from app.domain.types import (
    ExecutionMetrics,
    ReasoningStrategy,
    StrategyResult,
    StrategyRunInput,
)
from app.team.roles import (
    RoleRunner,
    create_analista_runner,
    create_executor_runner,
    create_planejador_runner,
)
from app.team.supervisor import DecideNextFn, TeamRole, create_decide_next
from app.team.team_graph import TeamGraphDeps, run_team_graph


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compose_message(input: StrategyRunInput) -> str:
    if not input.history:
        return input.message
    history = "\n".join(f"{m.role}: {m.content}" for m in input.history)
    return "\n".join(["Histórico recente da conversa:", history, "", input.message])


class TeamStrategy(ReasoningStrategy):
    name = "team"

    def __init__(
        self,
        model_factory: Callable[[], OpsChatModel],
        #: Tools somente-leitura (restrição estrutural — FR-007).
        analista_tools: list[StructuredTool],
        #: Tools só de incidente (restrição estrutural — FR-007).
        executor_tools: list[StructuredTool],
        #: Supervisor determinístico para testes (pula o LLM; não conta em llm_calls).
        decide_next: DecideNextFn | None = None,
        #: Overrides por papel para testes.
        role_runners: dict[TeamRole, RoleRunner] | None = None,
    ) -> None:
        role_runners = role_runners or {}
        self._deps = TeamGraphDeps(
            decide_next=decide_next or create_decide_next(model_factory),
            supervisor_llm_calls=0 if decide_next else 1,
            role_runners={
                "analista": role_runners.get("analista")
                or create_analista_runner(model_factory, analista_tools),
                "planejador": role_runners.get("planejador")
                or create_planejador_runner(model_factory),
                "executor": role_runners.get("executor")
                or create_executor_runner(model_factory, executor_tools),
            },
        )

    async def run(self, input: StrategyRunInput) -> StrategyResult:
        started_at = _now_ms()
        result = await run_team_graph(self._deps, _compose_message(input))
        return StrategyResult(
            answer=result.answer,
            trace=result.trace,
            metrics=ExecutionMetrics(
                llm_calls=result.llm_calls, latency_ms=_now_ms() - started_at
            ),
        )
