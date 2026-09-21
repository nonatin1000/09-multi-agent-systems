"""Re-exporta a fábrica de modelo (equivalente a ``src/llm/factory.ts``)."""

from app.agents.model import (
    MODEL_RETRY_ATTEMPTS,
    OpsChatModel,
    OpsResilientChatModel,
    base_model,
    compose_resilient_runnable,
    create_model,
    normalize_fallback,
)

__all__ = [
    "MODEL_RETRY_ATTEMPTS",
    "OpsChatModel",
    "OpsResilientChatModel",
    "base_model",
    "compose_resilient_runnable",
    "create_model",
    "normalize_fallback",
]
