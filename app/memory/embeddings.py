"""Embedder de produção via sentence-transformers (all-MiniLM-L6-v2)
(equivalente a ``src/memory/embeddings.ts``, que usava ``@huggingface/transformers``)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.domain.errors import EmbeddingError
from app.domain.types import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_ID)
        except Exception as cause:  # pragma: no cover - download/env dependent
            raise EmbeddingError(f"Failed to load embedding model {MODEL_ID}") from cause
    return _model


class SentenceTransformersEmbedder:
    """Embedder assíncrono (offload da inferência síncrona para thread)."""

    async def embed(self, text: str) -> list[float]:
        try:
            model = await asyncio.to_thread(_load_model)
            vector = await asyncio.to_thread(
                model.encode, text, normalize_embeddings=True
            )
            values = [float(v) for v in vector]
        except EmbeddingError:
            raise
        except Exception as cause:
            raise EmbeddingError("Failed to embed text") from cause

        if len(values) != EMBEDDING_DIM:
            raise EmbeddingError(
                f"Unexpected embedding dimension {len(values)}; expected {EMBEDDING_DIM}"
            )
        return values


_default_embedder: Embedder | None = None


def get_default_embedder() -> Embedder:
    """Singleton preguiçoso do Embedder padrão (all-MiniLM-L6-v2)."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformersEmbedder()
    return _default_embedder


def reset_default_embedder_for_tests() -> None:
    """@internal Reseta o singleton (só para testes)."""
    global _default_embedder, _model
    _default_embedder = None
    _model = None
