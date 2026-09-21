"""Vetores unitários determinísticos para testes de store/HTTP (sem download HF)
(equivalente a ``src/memory/fake-embedder.ts``)."""

from __future__ import annotations

import math

from app.memory.embeddings import EMBEDDING_DIM


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    assert norm > 0
    return [v / norm for v in values]


class FakeEmbedder:
    def __init__(self) -> None:
        self._table: dict[str, list[float]] = {}

    def set(self, text: str, values: list[float]) -> FakeEmbedder:
        assert len(values) == EMBEDDING_DIM, "fake vector must be 384-d"
        self._table[text] = _normalize(list(values))
        return self

    def set_axis(self, text: str, axis: int) -> FakeEmbedder:
        """Estilo one-hot: vetor unitário no eixo ``axis`` (0..383)."""
        values = [1.0 if i == axis else 0.0 for i in range(EMBEDDING_DIM)]
        return self.set(text, values)

    async def embed(self, text: str) -> list[float]:
        registered = self._table.get(text)
        if registered is not None:
            return list(registered)
        # Fallback determinístico: testes de /chat sem memórias seedadas ainda
        # recuperam com segurança.
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        axis = abs(_to_signed_32(h)) % EMBEDDING_DIM
        vec = [0.0] * EMBEDDING_DIM
        vec[axis] = 1.0
        vec[(axis + 1) % EMBEDDING_DIM] = ((h >> 8) & 0xFF) / 255
        return _normalize(vec)


def _to_signed_32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value
