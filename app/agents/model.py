"""Fábrica única de modelo: retry no primário → reserva opcional → falha vista
pelo chamador (ladder: with_retry → with_fallbacks → ModelUnavailableError,
HTTP 503). Equivalente a ``src/agents/model.ts``."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from app.domain.errors import ModelUnavailableError
from app.llm.model_telemetry import (
    ModelTelemetry,
    create_empty_telemetry,
    record_model_success,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_PRIMARY_MODEL = "openai/gpt-4o-mini"
#: Espelha o esboço de sala de aula (``stopAfterAttempt: 2``).
MODEL_RETRY_ATTEMPTS = 2


def normalize_fallback(raw: str | None, primary_id: str) -> str | None:
    if raw is None:
        return None
    trimmed = raw.strip()
    if not trimmed or trimmed == primary_id:
        return None
    return trimmed


class _ModelTelemetryCallback(AsyncCallbackHandler):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        record_model_success(self._model_id)


def base_model(model_id: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

    return ChatOpenAI(
        api_key=api_key,
        model=model_id,
        temperature=0,
        max_retries=0,
        base_url=_OPENROUTER_BASE_URL,
        callbacks=[_ModelTelemetryCallback(model_id)],
    )


def _wrap_model_unavailable(runnable: Runnable) -> Runnable:
    async def _invoke(input: Any, config: Any = None) -> Any:
        try:
            return await runnable.ainvoke(input, config)
        except ModelUnavailableError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ModelUnavailableError(str(error) or None) from error

    return RunnableLambda(_invoke)


def compose_resilient_runnable(
    primary: ChatOpenAI, backup: ChatOpenAI | None = None
) -> Runnable:
    """Compõe primary.with_retry → with_fallbacks([backup.with_retry])."""
    primary_retry = primary.with_retry(stop_after_attempt=MODEL_RETRY_ATTEMPTS)
    if backup is None:
        return _wrap_model_unavailable(primary_retry)
    backup_retry = backup.with_retry(stop_after_attempt=MODEL_RETRY_ATTEMPTS)
    return _wrap_model_unavailable(primary_retry.with_fallbacks([backup_retry]))


class OpsResilientChatModel:
    """Fachada de chat model para que create_react_agent / with_structured_output
    continuem funcionando enquanto todo invoke usa a composição resiliente."""

    def __init__(
        self,
        primary: ChatOpenAI,
        backup: ChatOpenAI | None,
        primary_model_id: str,
        fallback_model_id: str | None = None,
    ) -> None:
        self._primary = primary
        self._backup = backup
        self.primary_model_id = primary_model_id
        self.fallback_model_id = fallback_model_id

    def create_telemetry(self) -> ModelTelemetry:
        return create_empty_telemetry(self.primary_model_id, self.fallback_model_id)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        primary_retry = self._primary.bind_tools(tools, **kwargs).with_retry(
            stop_after_attempt=MODEL_RETRY_ATTEMPTS
        )
        if self._backup is None:
            return _wrap_model_unavailable(primary_retry)
        backup_retry = self._backup.bind_tools(tools, **kwargs).with_retry(
            stop_after_attempt=MODEL_RETRY_ATTEMPTS
        )
        return _wrap_model_unavailable(primary_retry.with_fallbacks([backup_retry]))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        primary_retry = self._primary.with_structured_output(schema, **kwargs).with_retry(
            stop_after_attempt=MODEL_RETRY_ATTEMPTS
        )
        if self._backup is None:
            return _wrap_model_unavailable(primary_retry)
        backup_retry = self._backup.with_structured_output(schema, **kwargs).with_retry(
            stop_after_attempt=MODEL_RETRY_ATTEMPTS
        )
        return _wrap_model_unavailable(primary_retry.with_fallbacks([backup_retry]))

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        return await compose_resilient_runnable(self._primary, self._backup).ainvoke(
            input, **kwargs
        )

    def get_primary(self) -> ChatOpenAI:
        return self._primary

    def get_backup(self) -> ChatOpenAI | None:
        return self._backup


OpsChatModel = OpsResilientChatModel


def create_model() -> OpsChatModel:
    """Esboço de sala de aula (fallback opcional + defaults):

    ```python
    primary = base_model(OPENROUTER_MODEL).with_retry(stop_after_attempt=2)
    backup = base_model(OPENROUTER_MODEL_FALLBACK).with_retry(stop_after_attempt=2)
    return primary.with_fallbacks([backup])
    ```
    """
    primary_id = os.environ.get("OPENROUTER_MODEL", "").strip() or _DEFAULT_PRIMARY_MODEL
    fallback_id = normalize_fallback(
        os.environ.get("OPENROUTER_MODEL_FALLBACK"), primary_id
    )

    primary = base_model(primary_id)
    backup = base_model(fallback_id) if fallback_id else None
    return OpsResilientChatModel(primary, backup, primary_id, fallback_id)
