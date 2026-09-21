"""Estimativa de tokens e leitura defensiva de usage do LLM
(equivalente a ``src/context/tokens.ts``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.domain.types import ContextBreakdown

__all__ = [
    "ContextBreakdown",
    "LlmUsage",
    "estimate_tokens",
    "read_llm_usage",
    "sum_prompt_tokens_from_messages",
    "add_prompt_tokens",
    "build_context_breakdown",
]


@dataclass(slots=True)
class LlmUsage:
    prompt_tokens: int
    completion_tokens: int | None = None
    total_tokens: int | None = None


def estimate_tokens(text: str) -> int:
    """Estimativa canônica: ``len(text) // 4``."""
    return len(text) // 4


def _non_neg_int(value: object) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < 0:
        return None
    return int(value)


def _usage_from_parts(
    prompt: object, completion: object, total: object
) -> LlmUsage | None:
    prompt_tokens = _non_neg_int(prompt)
    if prompt_tokens is None:
        return None
    completion_tokens = _non_neg_int(completion)
    total_tokens = _non_neg_int(total)
    return LlmUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def read_llm_usage(source: object) -> LlmUsage | None:
    """Parse defensivo de usage estilo LangChain/OpenAI. Nunca lança."""
    try:
        root = _as_record(source)
        if root is None:
            return None

        usage_meta = _as_record(root.get("usage_metadata"))
        if usage_meta:
            from_meta = _usage_from_parts(
                usage_meta.get("input_tokens"),
                usage_meta.get("output_tokens"),
                usage_meta.get("total_tokens"),
            )
            if from_meta:
                return from_meta

        response_meta = _as_record(root.get("response_metadata"))
        token_usage = (
            _as_record(response_meta.get("tokenUsage")) if response_meta else None
        ) or (
            _as_record(response_meta.get("token_usage")) if response_meta else None
        ) or _as_record(root.get("tokenUsage")) or _as_record(root.get("token_usage"))
        if token_usage:
            from_token_usage = _usage_from_parts(
                token_usage.get("promptTokens") or token_usage.get("prompt_tokens"),
                token_usage.get("completionTokens")
                or token_usage.get("completion_tokens"),
                token_usage.get("totalTokens") or token_usage.get("total_tokens"),
            )
            if from_token_usage:
                return from_token_usage

        return _usage_from_parts(
            root.get("promptTokens")
            or root.get("prompt_tokens")
            or root.get("input_tokens"),
            root.get("completionTokens")
            or root.get("completion_tokens")
            or root.get("output_tokens"),
            root.get("totalTokens") or root.get("total_tokens"),
        )
    except Exception:
        return None


def _as_record(value: object) -> dict | None:
    if value is None or not isinstance(value, dict):
        # Objetos LangChain expõem os mesmos dados via atributo ``.model_dump()``
        # ou ``__dict__``; tenta extrair um dict razoável antes de desistir.
        if value is not None and hasattr(value, "__dict__"):
            return dict(vars(value))
        return None
    return value


def sum_prompt_tokens_from_messages(messages: Iterable[object]) -> int | None:
    """Soma prompt tokens de mensagens que expõem usage (ex.: AIMessage[])."""
    total = 0
    contributions = 0
    for message in messages:
        usage = read_llm_usage(_message_usage_source(message))
        if usage:
            total += usage.prompt_tokens
            contributions += 1
    return total if contributions > 0 else None


def _message_usage_source(message: object) -> dict | None:
    """Extrai um dict com ``usage_metadata``/``response_metadata`` de uma mensagem
    LangChain (AIMessage), sem acoplar o módulo ao pacote langchain diretamente."""
    usage_metadata = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None)
    if usage_metadata is None and response_metadata is None:
        return None
    return {
        "usage_metadata": usage_metadata,
        "response_metadata": response_metadata,
    }


def add_prompt_tokens(current: int | None, next_: int | None) -> int | None:
    """Agrega totais opcionais de prompt tokens (None = desconhecido / omitir)."""
    if next_ is None:
        return current
    if current is None:
        return next_
    return current + next_


def build_context_breakdown(
    system: str,
    history: str,
    memories: str,
    message: str,
    summary: str = "",
) -> ContextBreakdown:
    return ContextBreakdown(
        system=estimate_tokens(system),
        history=estimate_tokens(history),
        memories=estimate_tokens(memories),
        message=estimate_tokens(message),
        summary=estimate_tokens(summary),
    )
