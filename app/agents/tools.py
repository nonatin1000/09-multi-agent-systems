"""Fábricas de tools LangChain do OpsPilot (equivalente a ``src/agents/tools.ts``).

Compartilhado com o servidor MCP — única fonte de verdade dos schemas de input.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.domain.errors import IncidentNotFoundError, RunbookNotFoundError
from app.domain.severity import normalize_severity
from app.domain.types import MemoryStore, OpsStore
from app.memory.chat_user_context import get_chat_user_id
from app.tools.check_provider_status import FetchFn, fetch_provider_status


class ListAlertsInput(BaseModel):
    status: Literal["firing", "resolved", "all"] = Field(
        default="firing",
        description=(
            'Filtro: "firing" (ativos), "resolved" (encerrados), "all" (todos). '
            "Default firing."
        ),
    )


class OpenIncidentInput(BaseModel):
    title: str = Field(min_length=1, description="Título curto do incidente")
    service: str = Field(
        min_length=1,
        description="Nome exato do serviço afetado (ex.: payments, checkout, auth)",
    )
    severity: Literal["critical", "high", "medium", "low"] = Field(
        description=(
            "Severidade canônica do banco: critical | high | medium | low. "
            "Aliases aceitos: sev1→critical, sev2→high, sev3→medium, sev4→low."
        )
    )

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return normalize_severity(value)


class ResolveIncidentInput(BaseModel):
    id: str = Field(
        min_length=1,
        description="ID do incidente local a resolver (ex.: inc-1722103456789-a3f2)",
    )


class ListIncidentsInput(BaseModel):
    status: Literal["open", "resolved", "all"] = Field(
        default="open", description='Filtro: "open" (default), "resolved", ou "all".'
    )


class ConsultarRunbookInput(BaseModel):
    service: str = Field(
        min_length=1, description="Nome do serviço cujo runbook será consultado"
    )


class CheckProviderStatusInput(BaseModel):
    provider: Literal["github", "cloudflare"] = Field(
        default="github",
        description=(
            'Provedor externo cuja statuspage pública será consultada: '
            '"github" (default) ou "cloudflare".'
        ),
    )


class ForgetPreferenceInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="Descrição da preferência ou fato a remover da memória do usuário atual",
    )


def create_list_alerts_tool(store: OpsStore) -> StructuredTool:
    async def _run(status: str = "firing") -> str:
        alerts = store.get_alerts(None if status == "all" else status)  # type: ignore[arg-type]
        if not alerts:
            return "No alerts found." if status == "all" else f"No {status} alerts found."
        scope = "alert(s)" if status == "all" else f"{status} alert(s)"
        header = f"Found {len(alerts)} {scope}:"
        lines = [
            f"- [{a.id}] {a.service} | {a.severity} | {a.description}" for a in alerts
        ]
        return "\n".join([header, *lines])

    return StructuredTool.from_function(
        coroutine=_run,
        name="list_alerts",
        description=(
            "Lista alertas por status. Quando usar: inventário de alertas operacionais. "
            "Quando não usar: para incidentes formais (use list_incidents) ou "
            "procedimentos (use consultar_runbook)."
        ),
        args_schema=ListAlertsInput,
    )


def create_open_incident_tool(store: OpsStore) -> StructuredTool:
    async def _run(title: str, service: str, severity: str) -> str:
        incident = store.create_incident(title, service, severity)  # type: ignore[arg-type]
        return "\n".join(
            [
                f"Incident created successfully. ID: {incident.id}",
                f"Title: {incident.title}",
                f"Service: {incident.service}",
                f"Severity: {incident.severity}",
                f"Status: {incident.status}",
            ]
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="open_incident",
        description=(
            "Abre um incidente formal para um serviço. Quando usar: após identificar "
            "problema que exige registro (ex. alerta crítico em investigação). Quando não "
            "usar: só para consultar alertas; não use se o incidente já existe (aí "
            "list_incidents / resolve_incident). severity MUST ser o valor do banco: "
            "critical | high | medium | low (aliases sev1→critical, sev2→high, "
            "sev3→medium, sev4→low). Use nomes de serviço exatos do pedido."
        ),
        args_schema=OpenIncidentInput,
    )


def create_resolve_incident_tool(store: OpsStore) -> StructuredTool:
    async def _run(id: str) -> str:
        try:
            incident = store.resolve_incident(id)
            resolved_at = incident.resolved_at or int(time.time() * 1000)
            iso = datetime.fromtimestamp(resolved_at / 1000, tz=UTC).isoformat()
            return "\n".join(
                [
                    f"Incident {incident.id} has been resolved.",
                    f"Service: {incident.service}",
                    f"Resolved at: {iso}",
                ]
            )
        except IncidentNotFoundError as error:
            return f"Error: {error}"

    return StructuredTool.from_function(
        coroutine=_run,
        name="resolve_incident",
        description=(
            "Marca como resolvido um incidente já aberto no store local do OpsPilot "
            "(SQLite/in-memory). Quando usar: mitigação concluída e o incidente local "
            "deve fechar. Quando não usar: para criar incidente novo, listar estado, ou "
            "qualquer ação fora do store local — não é API externa."
        ),
        args_schema=ResolveIncidentInput,
    )


def create_list_incidents_tool(store: OpsStore) -> StructuredTool:
    async def _run(status: str = "open") -> str:
        incidents = store.get_incidents(None if status == "all" else status)  # type: ignore[arg-type]
        if not incidents:
            return "No incidents found." if status == "all" else f"No {status} incidents found."
        scope = "incident(s)" if status == "all" else f"{status} incident(s)"
        header = f"Found {len(incidents)} {scope}:"
        lines = []
        for incident in incidents:
            summary_part = f" | summary: {incident.summary}" if incident.summary else ""
            lines.append(
                f"- [{incident.id}] {incident.title} | {incident.service} | "
                f"{incident.severity} | {incident.status}{summary_part}"
            )
        return "\n".join([header, *lines])

    return StructuredTool.from_function(
        coroutine=_run,
        name="list_incidents",
        description=(
            "Lista incidentes por status. Quando usar: ver incidentes "
            "abertos/resolvidos/todos no plantão. Quando não usar: para alertas crus "
            "(list_alerts) ou texto de runbook (consultar_runbook)."
        ),
        args_schema=ListIncidentsInput,
    )


def create_consultar_runbook_tool(store: OpsStore) -> StructuredTool:
    async def _run(service: str) -> str:
        try:
            runbook = store.get_runbook(service)
            return f"Runbook for {runbook.service}:\n{runbook.content}"
        except RunbookNotFoundError as error:
            return f"Error: {error}"

    return StructuredTool.from_function(
        coroutine=_run,
        name="consultar_runbook",
        description=(
            "Consulta o runbook operacional de um serviço. Quando usar: precisa dos "
            "passos de checkout, payments ou auth. Quando não usar: para inventar "
            "procedimento sem runbook; não substitui abrir/resolver incidente."
        ),
        args_schema=ConsultarRunbookInput,
    )


def create_check_provider_status_tool(fetch: FetchFn | None = None) -> StructuredTool:
    async def _run(provider: str = "github") -> str:
        return await fetch_provider_status(provider, fetch)  # type: ignore[arg-type]

    return StructuredTool.from_function(
        coroutine=_run,
        name="check_provider_status",
        description=(
            "Consulta o status público de um provedor externo (GitHub ou Cloudflare) via "
            'statuspage. Quando usar: suspeita de problema externo; dúvida "é o nosso ou '
            'do provedor?"; dependência aparentemente fora do ar. Quando não usar: '
            "inventário local de alertas/incidentes/runbooks (use list_alerts, "
            "list_incidents ou consultar_runbook)."
        ),
        args_schema=CheckProviderStatusInput,
    )


def create_forget_preference_tool(memories: MemoryStore) -> StructuredTool:
    async def _run(query: str) -> str:
        user_id = get_chat_user_id()
        if not user_id:
            return "Error: no active chat user context."
        try:
            hits = await memories.recall(user_id, query)
            if not hits:
                return "No matching preference found."
            target = hits[0]
            ok = await memories.forget(user_id, target.id)
            if not ok:
                return "Error: could not forget preference."
            return f"Forgot preference: {target.fact}"
        except Exception as error:  # noqa: BLE001
            return f"Error: {error}"

    return StructuredTool.from_function(
        coroutine=_run,
        name="forget_preference",
        description=(
            "Remove uma preferência ou fato previamente memorizado do usuário atual. "
            'Quando usar: plantonista pede para esquecer preferência/fato ("esqueça que '
            'priorizo checkout"). Quando não usar: apagar alertas/incidentes; listar '
            "memórias; pedidos operacionais pontuais sem intenção de esquecer preferência."
        ),
        args_schema=ForgetPreferenceInput,
    )


def create_tools(
    store: OpsStore, memories: MemoryStore | None = None
) -> list[StructuredTool]:
    tools = [
        create_list_alerts_tool(store),
        create_open_incident_tool(store),
        create_resolve_incident_tool(store),
        create_list_incidents_tool(store),
        create_consultar_runbook_tool(store),
        create_check_provider_status_tool(),
    ]
    if memories is not None:
        tools.append(create_forget_preference_tool(memories))
    return tools


__all__ = [
    "ListAlertsInput",
    "OpenIncidentInput",
    "ResolveIncidentInput",
    "ListIncidentsInput",
    "ConsultarRunbookInput",
    "CheckProviderStatusInput",
    "ForgetPreferenceInput",
    "create_list_alerts_tool",
    "create_open_incident_tool",
    "create_resolve_incident_tool",
    "create_list_incidents_tool",
    "create_consultar_runbook_tool",
    "create_check_provider_status_tool",
    "create_forget_preference_tool",
    "create_tools",
]
