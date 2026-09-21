"""Decisão do supervisor do modo equipe (equivalente a ``src/team/supervisor.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.model import OpsChatModel
from app.team.blackboard import BlackboardEntry, render_blackboard
from app.team.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT

TEAM_ROLES = ("analista", "planejador", "executor")
TeamRole = Literal["analista", "planejador", "executor"]


class SupervisorDecision(BaseModel):
    next: Literal["analista", "planejador", "executor", "done"]
    brief: str = Field(
        description="instrução de trabalho para o próximo papel (nó) ou resumo final se done"
    )


DecideNextFn = Callable[[str, list[BlackboardEntry], int], Awaitable[SupervisorDecision]]


def create_decide_next(model_factory: Callable[[], OpsChatModel]) -> DecideNextFn:
    async def _decide(
        message: str, blackboard: list[BlackboardEntry], handoff_count: int
    ) -> SupervisorDecision:
        raw = await model_factory().with_structured_output(SupervisorDecision).ainvoke(
            [
                ("system", SUPERVISOR_SYSTEM_PROMPT),
                (
                    "user",
                    "\n".join(
                        [
                            f"Pedido do plantonista: {message}",
                            f"Delegações já usadas: {handoff_count}",
                            "",
                            "Blackboard:",
                            render_blackboard(blackboard),
                        ]
                    ),
                ),
            ]
        )
        return raw if isinstance(raw, SupervisorDecision) else SupervisorDecision(**raw)

    return _decide
