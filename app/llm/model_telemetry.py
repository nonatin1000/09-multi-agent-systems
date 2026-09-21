"""Telemetria de modelo por turno via ``contextvars`` — equivalente ao
``AsyncLocalStorage`` do TS original (``src/llm/model-telemetry.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ModelTelemetry:
    primary_model: str
    fallback_model: str | None = None
    model_used: str | None = None
    #: True quando o modelo reserva completou com sucesso após falha do primário.
    fallback_used: bool = False
    #: Ids de modelo que reportaram sucesso (LLM end) neste turno, em ordem.
    success_order: list[str] = field(default_factory=list)


_storage: ContextVar[ModelTelemetry | None] = ContextVar("model_telemetry", default=None)


def create_empty_telemetry(
    primary_model: str, fallback_model: str | None = None
) -> ModelTelemetry:
    return ModelTelemetry(primary_model=primary_model, fallback_model=fallback_model)


async def run_with_model_telemetry(
    telemetry: ModelTelemetry, fn: Callable[[], Awaitable[T]]
) -> T:
    token: Token = _storage.set(telemetry)
    try:
        return await fn()
    finally:
        _storage.reset(token)


def get_model_telemetry() -> ModelTelemetry | None:
    return _storage.get()


def record_model_success(model_id: str) -> None:
    store = _storage.get()
    if store is None:
        return
    store.model_used = model_id
    store.success_order.append(model_id)
    if store.fallback_model and model_id == store.fallback_model:
        store.fallback_used = True


def record_fallback_used() -> None:
    store = _storage.get()
    if store is None:
        return
    store.fallback_used = True
    if store.fallback_model:
        store.model_used = store.fallback_model
