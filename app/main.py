"""Bootstrap + entrypoint do OpsPilot (equivalente a ``src/index.ts``)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from app.agents.model import create_model
from app.agents.react import ReactStrategy
from app.agents.tools import (
    create_check_provider_status_tool,
    create_consultar_runbook_tool,
    create_list_alerts_tool,
    create_list_incidents_tool,
    create_open_incident_tool,
    create_resolve_incident_tool,
    create_tools,
)
from app.chat.history_summarizer import create_llm_conversation_summarizer
from app.domain.types import (
    ApprovalStore,
    ConversationStore,
    Logger,
    MemoryStore,
    RequestStore,
)
from app.graph.production_graph import ProductionStrategies
from app.http.server import ChatAppDeps, create_app
from app.memory.learning_reflector import create_llm_learning_reflector
from app.memory.memory_store import SqliteMemoryStore
from app.obs.logger import create_logger
from app.store.memory_approval_store import MemoryApprovalStore
from app.store.seed import seed_ops_store
from app.store.sqlite_conversation_store import SqliteConversationStore
from app.store.sqlite_ops_store import SqliteOpsStore
from app.store.sqlite_request_store import SqliteRequestStore
from app.strategies.plan_execute import PlanExecuteStrategy
from app.strategies.reflect import ReflectionOpts, with_reflection
from app.team.team_strategy import TeamStrategy

load_dotenv()


@dataclass(slots=True)
class OpsPilotContext:
    store: SqliteOpsStore
    conversations: ConversationStore
    memories: MemoryStore
    requests: RequestStore
    approvals: ApprovalStore
    logger: Logger
    tools: list
    learning_reflector: object
    summarizer: object
    strategies: ProductionStrategies


def bootstrap_ops_pilot() -> OpsPilotContext:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    db_path = os.environ.get("OPSPILOT_DB", "./data/opspilot.db")
    store = SqliteOpsStore(db_path)
    seed_ops_store(store)
    conversations = SqliteConversationStore(db_path)
    memories = SqliteMemoryStore(db_path)
    requests = SqliteRequestStore(db_path)
    approvals = MemoryApprovalStore()
    logger = create_logger()
    tools = create_tools(store, memories)
    learning_reflector = create_llm_learning_reflector(create_model)
    summarizer = create_llm_conversation_summarizer(create_model)

    react = ReactStrategy(model_factory=create_model, tools=tools, max_iterations=10)
    plan_execute = PlanExecuteStrategy(
        model_factory=create_model, tools=tools, max_iterations=10
    )
    reflect = with_reflection(react, ReflectionOpts(model_factory=create_model))

    # Partição estrutural de tools por papel da equipe (spec 018 FR-007):
    # analista só lê; executor só muta incidentes; planejador não tem nenhuma.
    team = TeamStrategy(
        model_factory=create_model,
        analista_tools=[
            create_list_alerts_tool(store),
            create_list_incidents_tool(store),
            create_consultar_runbook_tool(store),
            create_check_provider_status_tool(),
        ],
        executor_tools=[
            create_open_incident_tool(store),
            create_resolve_incident_tool(store),
            create_list_incidents_tool(store),
        ],
    )

    return OpsPilotContext(
        store=store,
        conversations=conversations,
        memories=memories,
        requests=requests,
        approvals=approvals,
        logger=logger,
        tools=tools,
        learning_reflector=learning_reflector,
        summarizer=summarizer,
        strategies=ProductionStrategies(
            react=react, plan_execute=plan_execute, reflect=reflect, team=team
        ),
    )


def build_app() -> FastAPI:
    ctx = bootstrap_ops_pilot()
    return create_app(
        ChatAppDeps(
            strategies=ctx.strategies,
            conversations=ctx.conversations,
            memories=ctx.memories,
            requests=ctx.requests,
            approvals=ctx.approvals,
            logger=ctx.logger,
            learning_reflector=ctx.learning_reflector,
            summarizer=ctx.summarizer,
            route_model_factory=create_model,
        )
    )


def main() -> None:
    app = build_app()
    port = int(os.environ.get("PORT", "3000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
