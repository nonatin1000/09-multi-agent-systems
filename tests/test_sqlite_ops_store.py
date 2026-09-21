import pytest

from app.domain.errors import IncidentNotFoundError, RunbookNotFoundError
from app.store.sqlite_ops_store import SqliteOpsStore


def test_seed_is_idempotent(sqlite_ops_store: SqliteOpsStore):
    from app.store.seed import mercadinho_seed

    before = sqlite_ops_store.counts()
    sqlite_ops_store.seed(mercadinho_seed)
    assert sqlite_ops_store.counts() == before


def test_create_and_resolve_incident(sqlite_ops_store: SqliteOpsStore):
    incident = sqlite_ops_store.create_incident("DB down", "checkout", "critical")
    assert incident.status == "open"

    resolved = sqlite_ops_store.resolve_incident(incident.id, "mitigated")
    assert resolved.status == "resolved"
    assert resolved.summary == "mitigated"

    all_open = sqlite_ops_store.get_incidents("open")
    assert incident.id not in [i.id for i in all_open]


def test_resolve_unknown_incident_raises(sqlite_ops_store: SqliteOpsStore):
    with pytest.raises(IncidentNotFoundError):
        sqlite_ops_store.resolve_incident("does-not-exist")


def test_get_runbook_unknown_service_raises(sqlite_ops_store: SqliteOpsStore):
    with pytest.raises(RunbookNotFoundError):
        sqlite_ops_store.get_runbook("does-not-exist")


def test_severity_check_constraint_rejects_invalid_value(sqlite_ops_store: SqliteOpsStore):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        sqlite_ops_store.database.execute(
            "INSERT INTO alerts (id, service, description, severity, status) "
            "VALUES ('x', 'checkout', 'desc', 'invalid', 'firing')"
        )
