import pytest

from app.domain.errors import ConversationNotFoundError
from app.store.sqlite_conversation_store import SqliteConversationStore


def test_append_and_last_messages_preserve_order(
    sqlite_conversation_store: SqliteConversationStore,
):
    cid = sqlite_conversation_store.create()
    sqlite_conversation_store.append(cid, "user", "oi")
    sqlite_conversation_store.append(cid, "assistant", "olá")
    sqlite_conversation_store.append(cid, "user", "tudo bem?")

    last = sqlite_conversation_store.last_messages(cid, 2)
    assert [m.content for m in last] == ["olá", "tudo bem?"]


def test_count_and_messages_ascending(sqlite_conversation_store: SqliteConversationStore):
    cid = sqlite_conversation_store.create()
    for i in range(5):
        sqlite_conversation_store.append(cid, "user", f"m{i}")

    assert sqlite_conversation_store.count_messages(cid) == 5
    page = sqlite_conversation_store.messages_ascending(cid, 1, 2)
    assert [m.content for m in page] == ["m1", "m2"]


def test_summary_upsert_and_get(sqlite_conversation_store: SqliteConversationStore):
    cid = sqlite_conversation_store.create()
    assert sqlite_conversation_store.get_summary(cid) is None

    sqlite_conversation_store.upsert_summary(cid, "resumo 1", 8)
    record = sqlite_conversation_store.get_summary(cid)
    assert record is not None
    assert record.text == "resumo 1"
    assert record.covered_count == 8

    sqlite_conversation_store.upsert_summary(cid, "resumo 2", 16)
    record = sqlite_conversation_store.get_summary(cid)
    assert record.text == "resumo 2"
    assert record.covered_count == 16


def test_operations_on_unknown_conversation_raise(
    sqlite_conversation_store: SqliteConversationStore,
):
    with pytest.raises(ConversationNotFoundError):
        sqlite_conversation_store.append("nope", "user", "oi")
    with pytest.raises(ConversationNotFoundError):
        sqlite_conversation_store.last_messages("nope", 5)
