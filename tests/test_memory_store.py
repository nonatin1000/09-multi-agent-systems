import pytest

from app.memory.fake_embedder import FakeEmbedder
from app.memory.memory_store import SqliteMemoryStore


@pytest.mark.asyncio
async def test_remember_dedupes_near_identical_facts(
    memory_store: SqliteMemoryStore, fake_embedder: FakeEmbedder
):
    fake_embedder.set_axis("prioriza checkout", 0)
    fake_embedder.set_axis("prioriza checkout ", 0)  # mesmo eixo => quase idêntico

    first = await memory_store.remember("u1", "prioriza checkout")
    second = await memory_store.remember("u1", "prioriza checkout ")

    assert first.stored is True
    assert second.stored is False
    assert second.id == first.id


@pytest.mark.asyncio
async def test_recall_filters_below_min_score_and_sorts_desc(
    memory_store: SqliteMemoryStore, fake_embedder: FakeEmbedder
):
    fake_embedder.set_axis("fato relevante", 1)
    fake_embedder.set_axis("fato pouco relevante", 2)
    fake_embedder.set_axis("query", 1)

    await memory_store.remember("u1", "fato relevante")
    await memory_store.remember("u1", "fato pouco relevante")

    results = await memory_store.recall("u1", "query")
    assert len(results) == 1
    assert results[0].fact == "fato relevante"


@pytest.mark.asyncio
async def test_forget_removes_only_for_matching_user(
    memory_store: SqliteMemoryStore, fake_embedder: FakeEmbedder
):
    fake_embedder.set_axis("fato", 3)
    result = await memory_store.remember("u1", "fato")

    assert await memory_store.forget("u2", result.id) is False
    assert await memory_store.forget("u1", result.id) is True
    assert await memory_store.forget("u1", result.id) is False


@pytest.mark.asyncio
async def test_remember_rejects_blank_input(memory_store: SqliteMemoryStore):
    from app.domain.errors import InvalidMemoryInputError

    with pytest.raises(InvalidMemoryInputError):
        await memory_store.remember("u1", "   ")
