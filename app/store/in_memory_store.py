"""OpsStore em memória (equivalente a ``src/store/in-memory-store.ts``)."""

from __future__ import annotations

import random
import time
from dataclasses import replace

from app.domain.errors import IncidentNotFoundError, RunbookNotFoundError
from app.domain.types import (
    Alert,
    AlertStatus,
    Incident,
    IncidentStatus,
    Runbook,
    SeedPayload,
    Service,
    Severity,
)


class InMemoryStore:
    def __init__(self) -> None:
        self._services: list[Service] = []
        self._alerts: list[Alert] = []
        self._incidents: list[Incident] = []
        self._runbooks: dict[str, Runbook] = {}

    def seed(self, data: SeedPayload) -> None:
        for service in data.services:
            if not any(s.name == service.name for s in self._services):
                self._services.append(replace(service))
        for alert in data.alerts:
            if not any(a.id == alert.id for a in self._alerts):
                self._alerts.append(replace(alert))
        for runbook in data.runbooks:
            if runbook.service not in self._runbooks:
                self._runbooks[runbook.service] = replace(runbook)

    def get_alerts(self, status: AlertStatus | None = None) -> list[Alert]:
        source = (
            [a for a in self._alerts if a.status == status] if status else self._alerts
        )
        return [replace(a) for a in source]

    def get_incidents(self, status: IncidentStatus | None = None) -> list[Incident]:
        source = (
            [i for i in self._incidents if i.status == status]
            if status
            else self._incidents
        )
        return [replace(i) for i in sorted(source, key=lambda i: i.created_at)]

    def create_incident(self, title: str, service: str, severity: Severity) -> Incident:
        incident = Incident(
            id=self._generate_incident_id(),
            title=title,
            service=service,
            severity=severity,
            status="open",
            created_at=int(time.time() * 1000),
            resolved_at=None,
            summary=None,
        )
        self._incidents.append(incident)
        return replace(incident)

    def resolve_incident(self, id: str, summary: str | None = None) -> Incident:
        for incident in self._incidents:
            if incident.id == id:
                incident.status = "resolved"
                incident.resolved_at = int(time.time() * 1000)
                incident.summary = summary
                return replace(incident)
        raise IncidentNotFoundError(id)

    def get_runbook(self, service: str) -> Runbook:
        runbook = self._runbooks.get(service)
        if not runbook:
            raise RunbookNotFoundError(service)
        return replace(runbook)

    @staticmethod
    def _generate_incident_id() -> str:
        suffix = f"{random.getrandbits(16):04x}"
        return f"inc-{int(time.time() * 1000)}-{suffix}"
