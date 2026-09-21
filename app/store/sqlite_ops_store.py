"""OpsStore com SQLite (equivalente a ``src/store/sqlite-ops-store.ts``)."""

from __future__ import annotations

import os
import random
import sqlite3
import time
from pathlib import Path

from app.domain.errors import IncidentNotFoundError, RunbookNotFoundError
from app.domain.types import (
    Alert,
    AlertStatus,
    Incident,
    IncidentStatus,
    Runbook,
    SeedPayload,
    Severity,
)


class SqliteOpsStore:
    """OpsStore com SQLite. Caminho: ``OPSPILOT_DB`` (default
    ``./data/opspilot.db``); use ``:memory:`` em testes."""

    def __init__(self, path: str | None = None) -> None:
        resolved_path = path or os.environ.get("OPSPILOT_DB", "./data/opspilot.db")

        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        self.database = sqlite3.connect(resolved_path, check_same_thread=False)
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS services (
                name TEXT PRIMARY KEY,
                tier TEXT NOT NULL CHECK (tier IN ('critical', 'high', 'standard'))
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                service TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
                status TEXT NOT NULL CHECK (status IN ('firing', 'resolved'))
            );

            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                service TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
                status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
                created_at INTEGER NOT NULL,
                resolved_at INTEGER,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS runbooks (
                service TEXT PRIMARY KEY,
                content TEXT NOT NULL
            );
            """
        )
        self.database.commit()

    def seed(self, data: SeedPayload) -> None:
        for service in data.services:
            self.database.execute(
                "INSERT OR IGNORE INTO services (name, tier) VALUES (?, ?)",
                (service.name, service.tier),
            )
        for alert in data.alerts:
            self.database.execute(
                "INSERT OR IGNORE INTO alerts (id, service, description, severity, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (alert.id, alert.service, alert.description, alert.severity, alert.status),
            )
        for runbook in data.runbooks:
            self.database.execute(
                "INSERT OR IGNORE INTO runbooks (service, content) VALUES (?, ?)",
                (runbook.service, runbook.content),
            )
        self.database.commit()

    def counts(self) -> dict[str, int]:
        """Helper para testes — contagens de serviços + runbooks após seed."""
        services = self.database.execute("SELECT COUNT(*) FROM services").fetchone()[0]
        alerts = len(self.get_alerts())
        runbooks = self.database.execute("SELECT COUNT(*) FROM runbooks").fetchone()[0]
        return {"services": services, "alerts": alerts, "runbooks": runbooks}

    def get_alerts(self, status: AlertStatus | None = None) -> list[Alert]:
        if status is None:
            rows = self.database.execute(
                "SELECT id, service, description, severity, status FROM alerts"
            ).fetchall()
        else:
            rows = self.database.execute(
                "SELECT id, service, description, severity, status FROM alerts WHERE status = ?",
                (status,),
            ).fetchall()
        return [
            Alert(id=r[0], service=r[1], description=r[2], severity=r[3], status=r[4])
            for r in rows
        ]

    def get_incidents(self, status: IncidentStatus | None = None) -> list[Incident]:
        base = (
            "SELECT id, title, service, severity, status, created_at, resolved_at, summary "
            "FROM incidents"
        )
        if status is None:
            rows = self.database.execute(f"{base} ORDER BY created_at ASC").fetchall()
        else:
            rows = self.database.execute(
                f"{base} WHERE status = ? ORDER BY created_at ASC", (status,)
            ).fetchall()
        return [self._map_incident(r) for r in rows]

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
        self.database.execute(
            "INSERT INTO incidents "
            "(id, title, service, severity, status, created_at, resolved_at, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                incident.id,
                incident.title,
                incident.service,
                incident.severity,
                incident.status,
                incident.created_at,
                incident.resolved_at,
                incident.summary,
            ),
        )
        self.database.commit()
        return incident

    def resolve_incident(self, id: str, summary: str | None = None) -> Incident:
        row = self.database.execute(
            "SELECT id, title, service, severity, status, created_at, resolved_at, summary "
            "FROM incidents WHERE id = ?",
            (id,),
        ).fetchone()
        if not row:
            raise IncidentNotFoundError(id)
        resolved_at = int(time.time() * 1000)
        self.database.execute(
            "UPDATE incidents SET status = 'resolved', resolved_at = ?, summary = ? WHERE id = ?",
            (resolved_at, summary, id),
        )
        self.database.commit()
        return self._map_incident(
            (row[0], row[1], row[2], row[3], "resolved", row[5], resolved_at, summary)
        )

    def get_runbook(self, service: str) -> Runbook:
        row = self.database.execute(
            "SELECT service, content FROM runbooks WHERE service = ?", (service,)
        ).fetchone()
        if not row:
            raise RunbookNotFoundError(service)
        return Runbook(service=row[0], content=row[1])

    @staticmethod
    def _map_incident(row: tuple) -> Incident:
        return Incident(
            id=row[0],
            title=row[1],
            service=row[2],
            severity=row[3],
            status=row[4],
            created_at=row[5],
            resolved_at=row[6],
            summary=row[7],
        )

    @staticmethod
    def _generate_incident_id() -> str:
        suffix = f"{random.getrandbits(16):04x}"
        return f"inc-{int(time.time() * 1000)}-{suffix}"
