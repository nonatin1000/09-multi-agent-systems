"""Blackboard compartilhado do modo equipe (equivalente a ``src/team/blackboard.ts``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlackboardKind = Literal["facts", "plan", "execution", "error"]


@dataclass(slots=True)
class BlackboardEntry:
    """Contribuição anexada por um papel ao blackboard compartilhado do turno."""

    role: str
    #: analista→facts, planejador→plan, executor→execution; error na falha do papel.
    kind: BlackboardKind
    #: Brief do supervisor que originou essa contribuição.
    brief: str
    content: str


def render_blackboard(entries: list[BlackboardEntry]) -> str:
    """Serializa o blackboard para os prompts do supervisor/papéis."""
    if not entries:
        return "(blackboard vazio — nenhuma contribuição ainda)"
    return "\n\n".join(
        f"[{i + 1}] {e.role} ({e.kind}) — brief: {e.brief}\n{e.content}"
        for i, e in enumerate(entries)
    )
