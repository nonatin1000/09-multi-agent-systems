"""Registro de estratégias (equivalente a ``src/agents/index.ts``).

Nomeado ``registry.py`` (em vez de ``index.py``) para evitar sombrear o
``__init__.py`` do pacote — ``index`` não é um nome idiomático em Python.
"""

from __future__ import annotations

from app.domain.errors import UnknownStrategyError
from app.domain.types import ReasoningStrategy
from app.strategies.reflect import ReflectionOpts, with_reflection

StrategyRegistry = dict[str, ReasoningStrategy]


def create_registry(entries: dict[str, ReasoningStrategy]) -> StrategyRegistry:
    return dict(entries)


def resolve_strategy(
    registry: StrategyRegistry,
    name: str,
    reflect: bool,
    reflection_opts: ReflectionOpts | None = None,
) -> ReasoningStrategy:
    base = registry.get(name)
    if base is None:
        raise UnknownStrategyError(name)
    if not reflect:
        return base
    return with_reflection(base, reflection_opts or ReflectionOpts())


def list_strategies(registry: StrategyRegistry) -> list[str]:
    return list(registry.keys())
