"""``userId`` do chat ativo via ``contextvars``
(equivalente a ``src/memory/chat-user-context.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")

_chat_user_id: ContextVar[str | None] = ContextVar("chat_user_id", default=None)


def run_with_chat_user(user_id: str, fn: Callable[[], T]) -> T:
    """Executa ``fn`` (síncrono) com o ``userId`` do chat atual vinculado."""
    token = _chat_user_id.set(user_id)
    try:
        return fn()
    finally:
        _chat_user_id.reset(token)


async def run_with_chat_user_async(
    user_id: str, fn: Callable[[], Awaitable[T]]
) -> T:
    """Executa a coroutine ``fn()`` com o ``userId`` do chat atual vinculado."""
    token = _chat_user_id.set(user_id)
    try:
        return await fn()
    finally:
        _chat_user_id.reset(token)


def get_chat_user_id() -> str | None:
    """``userId`` ativo do chat via contextvars, se houver."""
    return _chat_user_id.get()
