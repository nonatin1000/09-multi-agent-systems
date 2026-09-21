"""Normalização de severidade (equivalente a ``src/domain/severity.ts``)."""

from __future__ import annotations

from typing import Any

from app.domain.types import Severity

SEVERITIES: tuple[Severity, ...] = ("critical", "high", "medium", "low")

#: Aliases estilo pager (sev1…sev4) e variantes de caixa → valores canônicos do banco.
#: sev1=critical, sev2=high, sev3=medium, sev4=low.
_SEV_ALIASES: dict[str, Severity] = {
    "sev1": "critical",
    "sev2": "high",
    "sev3": "medium",
    "sev4": "low",
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
}


def normalize_severity(value: Any) -> Any:
    """Mapeia aliases (sev1…sev4) e variações de caixa para o valor canônico."""
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    return _SEV_ALIASES.get(key, value)


def is_severity(value: Any) -> bool:
    return isinstance(value, str) and value in SEVERITIES
