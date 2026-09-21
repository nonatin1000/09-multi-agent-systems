"""Monta o servidor MCP do OpsPilot (equivalente a ``src/mcp/create-server.ts``)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.agents.tools import (
    ListAlertsInput,
    OpenIncidentInput,
    ResolveIncidentInput,
    create_list_alerts_tool,
    create_open_incident_tool,
    create_resolve_incident_tool,
)
from app.domain.types import OpsStore

#: Alinhado com pyproject.toml [project.version].
OPSPILOT_MCP_VERSION = "0.1.0"


def create_ops_mcp_server(store: OpsStore) -> FastMCP:
    """Monta um servidor MCP do OpsPilot (nome: opspilot) com o catálogo v1:
    list_alerts, open_incident, resolve_incident. Não conecta transporte —
    o chamador conecta stdio ou in-memory."""
    server = FastMCP("opspilot")

    list_alerts = create_list_alerts_tool(store)
    open_incident = create_open_incident_tool(store)
    resolve_incident = create_resolve_incident_tool(store)

    @server.tool(name="list_alerts", description=list_alerts.description)
    async def _list_alerts(input: ListAlertsInput) -> str:
        return await list_alerts.ainvoke(input.model_dump())

    @server.tool(name="open_incident", description=open_incident.description)
    async def _open_incident(input: OpenIncidentInput) -> str:
        return await open_incident.ainvoke(input.model_dump())

    @server.tool(name="resolve_incident", description=resolve_incident.description)
    async def _resolve_incident(input: ResolveIncidentInput) -> str:
        return await resolve_incident.ainvoke(input.model_dump())

    return server
