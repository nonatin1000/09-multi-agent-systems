"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

import pytest

from app.memory.fake_embedder import FakeEmbedder
from app.memory.memory_store import SqliteMemoryStore
from app.store.in_memory_store import InMemoryStore
from app.store.seed import mercadinho_seed, seed_ops_store
from app.store.sqlite_conversation_store import SqliteConversationStore
from app.store.sqlite_ops_store import SqliteOpsStore
from app.store.sqlite_request_store import SqliteRequestStore


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    store = InMemoryStore()
    seed_ops_store(store, mercadinho_seed)
    return store


@pytest.fixture
def sqlite_ops_store() -> SqliteOpsStore:
    store = SqliteOpsStore(":memory:")
    seed_ops_store(store, mercadinho_seed)
    return store


@pytest.fixture
def sqlite_conversation_store() -> SqliteConversationStore:
    return SqliteConversationStore(":memory:")


@pytest.fixture
def sqlite_request_store() -> SqliteRequestStore:
    return SqliteRequestStore(":memory:")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store(fake_embedder: FakeEmbedder) -> SqliteMemoryStore:
    return SqliteMemoryStore(":memory:", embedder=fake_embedder)
