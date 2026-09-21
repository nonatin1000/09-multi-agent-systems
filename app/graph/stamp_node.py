"""Carimba (ou sobrescreve) ``node`` em cada evento de trace
(equivalente a ``src/graph/stamp-node.ts``)."""

from __future__ import annotations

from app.domain.types import TraceEvent


def stamp_node(node: str, events: list[TraceEvent]) -> list[TraceEvent]:
    return [event.replace_node(node) for event in events]
