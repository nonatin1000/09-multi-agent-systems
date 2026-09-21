"""Executores de papel do modo equipe: analista, planejador, executor
(equivalente a ``src/team/roles.ts``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from app.agents.model import OpsChatModel
from app.domain.types import TraceEvent
from app.team.blackboard import BlackboardEntry, BlackboardKind, render_blackboard
from app.team.supervisor import TeamRole
from app.trace.builder import build_trace_from_messages

ANALISTA_SYSTEM_PROMPT = "\n".join(
    [
        "Você é o ANALISTA do plantão. Sua única função: produzir diagnóstico "
        "FACTUAL do estado atual, usando as ferramentas de leitura.",
        "Liste: alertas disparando (com severidade), incidentes recentes (abertos "
        "e resolvidos), runbooks relevantes, status dos provedores e fatos "
        "conhecidos do time.",
        "NÃO proponha soluções. NÃO abra nem resolva nada.",
        "Formato: tópicos telegráficos. Seja cético: se um dado não está nas "
        "observações, não afirme.",
    ]
)

PLANEJADOR_SYSTEM_PROMPT = "\n".join(
    [
        "Você é o PLANEJADOR do plantão. Sua única função: transformar os fatos "
        "do blackboard em um plano de ação.",
        "Produza um plano numerado, em passos curtos e executáveis, baseado "
        "APENAS no que está no blackboard.",
        "Você não tem ferramentas: NÃO execute nada, NÃO invente dados que não "
        "estejam no blackboard.",
        "Se os fatos forem insuficientes, diga exatamente qual informação falta.",
    ]
)

EXECUTOR_SYSTEM_PROMPT = "\n".join(
    [
        "Você é o EXECUTOR do plantão. Sua única função: executar ações de "
        "incidente (abrir, resolver, listar) conforme o brief e o plano do "
        "blackboard.",
        "Use somente as ferramentas disponíveis; não invente IDs de incidente — "
        "quando necessário, liste antes de agir.",
        "Reporte cada ação executada e o resultado observado.",
        "Não faça diagnóstico novo nem replaneje: siga o plano.",
    ]
)


@dataclass(slots=True)
class RoleRunInput:
    message: str
    brief: str
    blackboard: list[BlackboardEntry]


@dataclass(slots=True)
class RoleRunResult:
    entry: BlackboardEntry
    trace: list[TraceEvent]
    llm_calls: int


class RoleRunner(Protocol):
    role: TeamRole
    #: Contrato de capacidade estrutural — inspecionado por testes (FR-007).
    tools: list[StructuredTool]

    async def run(self, input: RoleRunInput) -> RoleRunResult: ...


_DEFAULT_MAX_ITERATIONS = 6


def _role_user_message(input: RoleRunInput) -> str:
    return "\n".join(
        [
            f"Pedido original do plantonista: {input.message}",
            f"Sua tarefa (brief do supervisor): {input.brief}",
            "",
            "Blackboard atual:",
            render_blackboard(input.blackboard),
        ]
    )


class _ToolRoleRunner:
    def __init__(
        self,
        role: TeamRole,
        kind: BlackboardKind,
        prompt: str,
        model_factory: Callable[[], OpsChatModel],
        tools: list[StructuredTool],
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.role = role
        self.tools = tools
        self._kind = kind
        self._prompt = prompt
        self._model_factory = model_factory
        self._max_iterations = max_iterations

    async def run(self, input: RoleRunInput) -> RoleRunResult:
        agent = create_react_agent(
            self._model_factory(), self.tools, prompt=self._prompt
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": _role_user_message(input)}]},
            config={"recursion_limit": max(3, self._max_iterations * 3)},
        )
        trace = build_trace_from_messages(result["messages"], self.role)
        answers = [e for e in trace if e.type == "answer"]
        content = answers[-1].content if answers else ""
        llm_calls = sum(1 for m in result["messages"] if isinstance(m, AIMessage))
        return RoleRunResult(
            entry=BlackboardEntry(
                role=self.role, kind=self._kind, brief=input.brief, content=content
            ),
            trace=trace,
            llm_calls=llm_calls,
        )


def create_analista_runner(
    model_factory: Callable[[], OpsChatModel],
    tools: list[StructuredTool],
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> RoleRunner:
    return _ToolRoleRunner(
        "analista", "facts", ANALISTA_SYSTEM_PROMPT, model_factory, tools, max_iterations
    )


def create_executor_runner(
    model_factory: Callable[[], OpsChatModel],
    tools: list[StructuredTool],
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> RoleRunner:
    return _ToolRoleRunner(
        "executor", "execution", EXECUTOR_SYSTEM_PROMPT, model_factory, tools, max_iterations
    )


class _PlanejadorRunner:
    """Planejador não tem ferramentas por assinatura: chamada pura ao modelo
    sobre o blackboard."""

    role: TeamRole = "planejador"
    tools: list[StructuredTool] = []

    def __init__(self, model_factory: Callable[[], OpsChatModel]) -> None:
        self._model_factory = model_factory

    async def run(self, input: RoleRunInput) -> RoleRunResult:
        response = await self._model_factory().ainvoke(
            [
                ("system", PLANEJADOR_SYSTEM_PROMPT),
                ("user", _role_user_message(input)),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return RoleRunResult(
            entry=BlackboardEntry(
                role="planejador", kind="plan", brief=input.brief, content=content
            ),
            trace=[TraceEvent(type="plan", content=content, node="planejador")],
            llm_calls=1,
        )


def create_planejador_runner(model_factory: Callable[[], OpsChatModel]) -> RoleRunner:
    return _PlanejadorRunner(model_factory)
