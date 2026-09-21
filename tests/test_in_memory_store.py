import pytest

from app.domain.errors import IncidentNotFoundError, RunbookNotFoundError
from app.store.in_memory_store import InMemoryStore


def test_seed_is_idempotent(in_memory_store: InMemoryStore):
    from app.store.seed import mercadinho_seed

    before = len(in_memory_store.get_alerts())
    in_memory_store.seed(mercadinho_seed)
    assert len(in_memory_store.get_alerts()) == before


def test_get_alerts_filters_by_status(in_memory_store: InMemoryStore):
    firing = in_memory_store.get_alerts("firing")
    resolved = in_memory_store.get_alerts("resolved")
    assert all(a.status == "firing" for a in firing)
    assert all(a.status == "resolved" for a in resolved)
    assert len(firing) + len(resolved) == len(in_memory_store.get_alerts())


def test_create_and_resolve_incident(in_memory_store: InMemoryStore):
    incident = in_memory_store.create_incident("DB down", "checkout", "critical")
    assert incident.status == "open"
    assert incident.resolved_at is None

    resolved = in_memory_store.resolve_incident(incident.id, "mitigated")
    assert resolved.status == "resolved"
    assert resolved.summary == "mitigated"
    assert resolved.resolved_at is not None


def test_resolve_unknown_incident_raises(in_memory_store: InMemoryStore):
    with pytest.raises(IncidentNotFoundError):
        in_memory_store.resolve_incident("does-not-exist")


def test_get_runbook_unknown_service_raises(in_memory_store: InMemoryStore):
    with pytest.raises(RunbookNotFoundError):
        in_memory_store.get_runbook("does-not-exist")


def test_get_runbook_known_service(in_memory_store: InMemoryStore):
    runbook = in_memory_store.get_runbook("checkout")
    assert runbook.service == "checkout"
    assert runbook.content
