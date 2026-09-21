"""Roteador de produção: classifica a mensagem em uma rota
(equivalente a ``src/graph/router.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.model import OpsChatModel
from app.graph.router_prompt import ROUTER_SYSTEM_PROMPT

PRODUCTION_ROUTES = ("react", "planExecute", "reflect", "team")
ProductionRoute = Literal["react", "planExecute", "reflect", "team"]


class RouterDecision(BaseModel):
    route: ProductionRoute
    reason: str = Field(description="uma frase justificando a escolha")


ClassifyRouteFn = Callable[[str], Awaitable[RouterDecision]]

_FALLBACK_REASON = "fallback: router failed; using react"


def create_classify_route(
    model_factory: Callable[[], OpsChatModel],
) -> ClassifyRouteFn:
    async def _classify(message: str) -> RouterDecision:
        try:
            raw = await model_factory().with_structured_output(RouterDecision).ainvoke(
                [("system", ROUTER_SYSTEM_PROMPT), ("user", message)]
            )
            return raw if isinstance(raw, RouterDecision) else RouterDecision(**raw)
        except Exception:  # noqa: BLE001
            return RouterDecision(route="react", reason=_FALLBACK_REASON)

    return _classify


def parse_override_strategy(strategy: str) -> ProductionRoute | None:
    """Mapeia a string HTTP ``strategy`` para uma rota de produção (aliases permitidos)."""
    if is_production_route(strategy):
        return strategy  # type: ignore[return-value]
    if strategy == "plan-and-execute":
        return "planExecute"
    return None


def is_production_route(value: str) -> bool:
    return value in PRODUCTION_ROUTES
