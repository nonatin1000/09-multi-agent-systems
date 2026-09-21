"""Estimativa de custo e utilitários de estatísticas
(equivalente a ``src/obs/request-stats.ts``)."""

from __future__ import annotations

import math
import re

#: Modelos com ``:free`` (tier gratuito do OpenRouter) custam sempre 0.
_DEFAULT_USD_PER_1M_PROMPT = 0.15

_SINCE_RE = re.compile(r"^(\d+)(ms|s|m|h|d)$", re.IGNORECASE)
_UNIT_MULTIPLIER = {
    "ms": 1,
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
}


def is_free_model(model: str | None) -> bool:
    if not model:
        return False
    return ":free" in model


def estimate_prompt_cost_usd(model: str | None, prompt_tokens: int) -> float:
    if prompt_tokens <= 0:
        return 0.0
    if is_free_model(model):
        return 0.0
    return (prompt_tokens / 1_000_000) * _DEFAULT_USD_PER_1M_PROMPT


def parse_since_duration(raw: str) -> int | None:
    """Parseia ``24h``, ``7d``, ``30m``, ``90s`` → milissegundos."""
    match = _SINCE_RE.match(raw.strip())
    if not match:
        return None
    n = int(match.group(1))
    if n <= 0:
        return None
    unit = match.group(2).lower()
    return n * _UNIT_MULTIPLIER[unit]


def percentile(values: list[float], p: float) -> float | None:
    """Percentil pelo rank mais próximo, sobre cópia ordenada; None se vazio."""
    if not values:
        return None
    sorted_values = sorted(values)
    rank = math.ceil((p / 100) * len(sorted_values)) - 1
    idx = min(len(sorted_values) - 1, max(0, rank))
    return sorted_values[idx]
