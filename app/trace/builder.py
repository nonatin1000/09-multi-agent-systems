"""Constrói o trace a partir de mensagens LangChain
(equivalente a ``src/trace/builder.ts``)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.domain.types import TraceEvent
from app.graph.stamp_node import stamp_node

_ALLOWED_PLAN_EXECUTE_TYPES = {"plan", "action", "observation", "critique", "answer"}


def _to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item))
        return " ".join(parts).strip()
    return str(content) if content is not None else ""


def build_trace_from_messages(
    messages: Sequence[BaseMessage], node: str = "react"
) -> list[TraceEvent]:
    trace: list[TraceEvent] = []

    for message in messages:
        if isinstance(message, AIMessage):
            content = _to_text(message.content).strip()
            tool_calls = message.tool_calls or []
            if tool_calls:
                if content:
                    trace.append(TraceEvent(type="thought", content=content, node=node))
                for call in tool_calls:
                    args = call.get("args") or {}
                    trace.append(
                        TraceEvent(
                            type="action",
                            content=f"{call.get('name')}({json.dumps(args)})",
                            tool=call.get("name"),
                            tool_args=args,
                            node=node,
                        )
                    )
            elif content:
                trace.append(TraceEvent(type="answer", content=content, node=node))
            continue

        if isinstance(message, ToolMessage):
            content = _to_text(message.content).strip()
            trace.append(TraceEvent(type="observation", content=content, node=node))

    if not any(event.type == "answer" for event in trace):
        trace.append(TraceEvent(type="answer", content="No answer generated.", node=node))

    return trace


def build_plan_execute_trace(
    events: list[TraceEvent], node: str = "planExecute"
) -> list[TraceEvent]:
    filtered = [e for e in events if e.type in _ALLOWED_PLAN_EXECUTE_TYPES]
    if not filtered or filtered[-1].type != "answer":
        filtered.append(
            TraceEvent(type="answer", content="No final answer generated.", node=node)
        )
    return stamp_node(node, filtered)
