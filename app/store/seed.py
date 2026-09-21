"""Seed idempotente para qualquer OpsStore (equivalente a ``src/store/seed.ts``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.types import Alert, OpsStore, Runbook, SeedPayload, Service


class _ServiceModel(BaseModel):
    name: str = Field(min_length=1)
    tier: Literal["critical", "high", "standard"]


class _AlertModel(BaseModel):
    id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["critical", "high", "medium", "low"]
    status: Literal["firing", "resolved"]


class _RunbookModel(BaseModel):
    service: str = Field(min_length=1)
    content: str = Field(min_length=1)


class _SeedModel(BaseModel):
    services: list[_ServiceModel] = Field(min_length=1)
    alerts: list[_AlertModel] = Field(min_length=1)
    runbooks: list[_RunbookModel] = Field(min_length=1)


def _load_mercadinho_seed() -> SeedPayload:
    raw = json.loads((Path(__file__).parent / "seed_data.json").read_text(encoding="utf-8"))
    parsed = _SeedModel(**raw)
    return SeedPayload(
        services=[Service(name=s.name, tier=s.tier) for s in parsed.services],
        alerts=[
            Alert(id=a.id, service=a.service, description=a.description,
                  severity=a.severity, status=a.status)
            for a in parsed.alerts
        ],
        runbooks=[Runbook(service=r.service, content=r.content) for r in parsed.runbooks],
    )


mercadinho_seed = _load_mercadinho_seed()


def seed_ops_store(store: OpsStore, data: SeedPayload | None = None) -> None:
    store.seed(data if data is not None else mercadinho_seed)


if __name__ == "__main__":
    from app.store.in_memory_store import InMemoryStore

    _store = InMemoryStore()
    seed_ops_store(_store)
    print(f"Seeded {len(_store.get_alerts())} alerts.")
