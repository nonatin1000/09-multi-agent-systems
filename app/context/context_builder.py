"""ContextBuilder — monta o prompt com teto por seção
(equivalente a ``src/context/context-builder.ts``).

Env (defaults): ``CONTEXT_BUDGET_SUMMARY=200``, ``CONTEXT_BUDGET_HISTORY=1200``,
``CONTEXT_BUDGET_MEMORIES=300``. ``CONTEXT_BUDGET_SYSTEM`` é lido mas o corte da
seção system é sempre "never" (intocável). Mensagem atual também é intocável.
Alias legado: ``CONTEXT_BUDGET_WINDOW`` → history.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Literal

from app.context.tokens import estimate_tokens
from app.domain.types import ConversationMessage, RecalledMemory

CutRule = Literal["never", "truncate", "oldest-first", "lowest-score-first"]
SectionName = Literal["system", "summary", "history", "memories"]


@dataclass(slots=True)
class SectionBudgets:
    #: Lido do env; regra de corte é sempre "never" (system intocável).
    system: float = math.inf
    summary: int = 200
    history: int = 1200
    memories: int = 300


DEFAULT_SECTION_BUDGETS = SectionBudgets()


@dataclass(slots=True)
class BuildInput:
    system: str
    summary: str | None
    history: list[ConversationMessage]
    memories: list[RecalledMemory]
    message: str


@dataclass(slots=True)
class BuildContext:
    system: str
    message: str
    summary: str
    history: list[ConversationMessage]
    memories: list[RecalledMemory]
    enriched_message: str
    history_messages: int
    recalled_memories: int
    history_text: str
    memories_text: str
    summary_text: str


def _parse_budget(raw: str | None, fallback: float) -> float:
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return float(int(float(raw)))
    except ValueError:
        return fallback


def resolve_section_budgets(
    overrides: dict[str, float] | None = None,
    env: dict[str, str] | None = None,
) -> SectionBudgets:
    env = env if env is not None else dict(os.environ)
    # Prefere HISTORY; usa o alias legado WINDOW como fallback.
    history_raw = env.get("CONTEXT_BUDGET_HISTORY") or env.get("CONTEXT_BUDGET_WINDOW")
    budgets = SectionBudgets(
        system=_parse_budget(env.get("CONTEXT_BUDGET_SYSTEM"), DEFAULT_SECTION_BUDGETS.system),
        summary=int(
            _parse_budget(env.get("CONTEXT_BUDGET_SUMMARY"), DEFAULT_SECTION_BUDGETS.summary)
        ),
        history=int(_parse_budget(history_raw, DEFAULT_SECTION_BUDGETS.history)),
        memories=int(
            _parse_budget(env.get("CONTEXT_BUDGET_MEMORIES"), DEFAULT_SECTION_BUDGETS.memories)
        ),
    )
    if overrides:
        budgets = replace(budgets, **overrides)
    return budgets


def format_history_text(history: list[ConversationMessage]) -> str:
    if not history:
        return ""
    return "\n".join(f"{m.role}: {m.content}" for m in history)


def format_memories_text(recalled: list[RecalledMemory]) -> str:
    if not recalled:
        return ""
    return "\n".join(f"- {m.fact}" for m in recalled)


def format_memories_for_prompt(
    recalled: list[RecalledMemory], current_message: str
) -> str:
    """Injeta fatos recuperados no envelope da mensagem do usuário."""
    if not recalled:
        return current_message
    lines = "\n".join(f"- {m.fact}" for m in recalled)
    return f"Relevant memories:\n{lines}\n\nCurrent message:\n{current_message}"


def format_summary_for_prompt(summary: str | None, body: str) -> str:
    """Prefixa o resumo da conversa quando presente."""
    trimmed = (summary or "").strip()
    if not trimmed:
        return body
    return f"Conversation summary:\n{trimmed}\n\n{body}"


def _truncate_to_budget(text: str, budget: float) -> str:
    if budget <= 0:
        return ""
    max_chars = int(budget * 4)
    return text if len(text) <= max_chars else text[:max_chars]


def _shrink_until_fits(prefix_len: int, body: str, budget: float) -> str:
    if budget <= 0:
        return ""
    max_total_chars = int(budget * 4)
    max_body = max(0, max_total_chars - prefix_len)
    return body if len(body) <= max_body else body[:max_body]


def _fit_history(
    history: list[ConversationMessage], budget: float
) -> list[ConversationMessage]:
    if budget <= 0:
        return []
    result = list(history)
    while len(result) > 1 and estimate_tokens(format_history_text(result)) > budget:
        result = result[1:]
    if len(result) == 1 and estimate_tokens(format_history_text(result)) > budget:
        only = result[0]
        prefix = f"{only.role}: "
        result = [
            replace(only, content=_shrink_until_fits(len(prefix), only.content, budget))
        ]
    return result


def _fit_memories(
    memories: list[RecalledMemory], budget: float
) -> list[RecalledMemory]:
    if budget <= 0:
        return []
    indexed = list(enumerate(memories))
    while len(indexed) > 1 and estimate_tokens(
        format_memories_text([m for _, m in indexed])
    ) > budget:
        # Descarta o de menor score; empate descarta o de maior índice original.
        drop_at = 0
        for i in range(1, len(indexed)):
            cur_idx, cur = indexed[i]
            best_idx, best = indexed[drop_at]
            if cur.score < best.score or (cur.score == best.score and cur_idx > best_idx):
                drop_at = i
        indexed.pop(drop_at)
    if len(indexed) == 1 and estimate_tokens(
        format_memories_text([m for _, m in indexed])
    ) > budget:
        idx, only = indexed[0]
        indexed = [(idx, replace(only, fact=_shrink_until_fits(2, only.fact, budget)))]
    indexed.sort(key=lambda pair: pair[0])
    return [m for _, m in indexed]


@dataclass(slots=True)
class _FittedSection:
    name: SectionName
    value: object
    text: str


def _fit_to_budget(
    name: SectionName, value: object, budget: float, cut: CutRule
) -> _FittedSection:
    if name == "system":
        assert isinstance(value, str)
        return _FittedSection(name, value, value)

    if name == "summary":
        trimmed = str(value or "").strip()
        if cut == "never":
            return _FittedSection(name, trimmed, trimmed)
        fitted = _truncate_to_budget(trimmed, budget)
        return _FittedSection(name, fitted, fitted)

    if name == "history":
        history = list(value)  # type: ignore[arg-type]
        fitted_history = (
            _fit_history(history, budget)
            if cut in ("oldest-first", "truncate")
            else history
        )
        return _FittedSection(name, fitted_history, format_history_text(fitted_history))

    if name == "memories":
        memories = list(value)  # type: ignore[arg-type]
        fitted_memories = (
            _fit_memories(memories, budget)
            if cut in ("lowest-score-first", "truncate")
            else memories
        )
        return _FittedSection(
            name, fitted_memories, format_memories_text(fitted_memories)
        )

    return _FittedSection(name, value, "")


def build_context(
    input: BuildInput,
    budgets: dict[str, float] | None = None,
    env: dict[str, str] | None = None,
) -> BuildContext:
    """Monta o contexto com teto por seção. Puro: sem I/O."""
    b = resolve_section_budgets(budgets, env)

    fitted = {
        f.name: f
        for f in (
            _fit_to_budget("system", input.system, b.system, "never"),
            # Spec: resumo tem teto (default 200) → truncate.
            _fit_to_budget("summary", input.summary or "", b.summary, "truncate"),
            _fit_to_budget("history", input.history, b.history, "oldest-first"),
            _fit_to_budget(
                "memories", input.memories, b.memories, "lowest-score-first"
            ),
        )
    }

    summary = str(fitted["summary"].value or "")
    history = fitted["history"].value
    memories = fitted["memories"].value

    enriched_message = format_memories_for_prompt(memories, input.message)
    enriched_message = format_summary_for_prompt(summary or None, enriched_message)

    return BuildContext(
        system=input.system,
        message=input.message,
        summary=summary,
        history=history,
        memories=memories,
        enriched_message=enriched_message,
        history_messages=len(history),
        recalled_memories=len(memories),
        history_text=fitted["history"].text,
        memories_text=fitted["memories"].text,
        summary_text=summary,
    )
