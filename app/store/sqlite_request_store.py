"""RequestStore de auditoria com SQLite
(equivalente a ``src/store/sqlite-request-store.ts``)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from pathlib import Path

from app.domain.types import (
    ContextBreakdown,
    ExecutionMetrics,
    RequestRecord,
    RequestStatsBucket,
    RequestStatsSummary,
    SaveRequestInput,
    TraceEvent,
)
from app.obs.request_stats import estimate_prompt_cost_usd, percentile

_TRACE_PAYLOAD_KEYS = (
    "tool",
    "tool_args",
    "round",
    "approved",
    "timestamp_ms",
    "route",
    "override",
    "reason",
    "to",
)


def _split_trace_event(event: TraceEvent) -> tuple[str, str, str, dict | None]:
    payload = {
        key: value
        for key in _TRACE_PAYLOAD_KEYS
        if (value := getattr(event, key)) is not None
    }
    return event.type, event.node, event.content, (payload or None)


def _row_to_trace_event(
    type_: str, node: str, content: str, payload_json: str | None
) -> TraceEvent:
    base = TraceEvent(type=type_, node=node, content=content)
    if not payload_json:
        return base
    payload = json.loads(payload_json)
    for key, value in payload.items():
        setattr(base, key, value)
    return base


def _row_to_request(row: tuple) -> RequestRecord:
    metrics_dict = json.loads(row[7])
    context_breakdown = metrics_dict.get("context_breakdown")
    metrics = ExecutionMetrics(
        llm_calls=metrics_dict.get("llm_calls", 0),
        latency_ms=metrics_dict.get("latency_ms", 0),
        history_messages=metrics_dict.get("history_messages"),
        recalled_memories=metrics_dict.get("recalled_memories"),
        prompt_tokens=metrics_dict.get("prompt_tokens"),
        context_breakdown=ContextBreakdown(**context_breakdown) if context_breakdown else None,
        route=metrics_dict.get("route"),
        route_reason=metrics_dict.get("route_reason"),
        model_used=metrics_dict.get("model_used"),
    )
    return RequestRecord(
        id=row[0],
        created_at=row[1],
        finished_at=row[2],
        status=row[3],
        http_status=row[4],
        conversation_id=row[5],
        user_id=row[6],
        metrics=metrics,
        latency_ms=row[8],
        llm_calls=row[9],
        route=row[10],
        model_used=row[11],
    )


class SqliteRequestStore:
    """Store SQLite de auditoria para requisições /chat + trace_events ordenado.
    Mesmo arquivo OPSPILOT_DB dos outros stores; use ``:memory:`` em testes."""

    def __init__(self, path: str | None = None) -> None:
        resolved_path = path or os.environ.get("OPSPILOT_DB", "./data/opspilot.db")

        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        self.database = sqlite3.connect(resolved_path, check_same_thread=False)
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                finished_at INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('success', 'error')),
                http_status INTEGER NOT NULL,
                conversation_id TEXT,
                user_id TEXT,
                metrics_json TEXT NOT NULL,
                latency_ms INTEGER,
                llm_calls INTEGER,
                route TEXT,
                model_used TEXT
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                node TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(id),
                UNIQUE (request_id, seq)
            );

            CREATE INDEX IF NOT EXISTS idx_trace_events_request_seq
                ON trace_events (request_id, seq);
            """
        )
        self.database.commit()

    def save(self, input: SaveRequestInput) -> None:
        metrics_dict = {k: v for k, v in asdict(input.metrics).items() if v is not None}
        metrics_json = json.dumps(metrics_dict)
        try:
            self.database.execute("BEGIN")
            self.database.execute(
                """
                INSERT INTO requests (
                    id, created_at, finished_at, status, http_status,
                    conversation_id, user_id, metrics_json,
                    latency_ms, llm_calls, route, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input.id,
                    input.created_at,
                    input.finished_at,
                    input.status,
                    input.http_status,
                    input.conversation_id,
                    input.user_id,
                    metrics_json,
                    input.metrics.latency_ms,
                    input.metrics.llm_calls,
                    input.metrics.route,
                    input.metrics.model_used,
                ),
            )
            for seq, event in enumerate(input.trace):
                type_, node, content, payload = _split_trace_event(event)
                self.database.execute(
                    "INSERT INTO trace_events "
                    "(id, request_id, seq, type, node, content, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        input.id,
                        seq,
                        type_,
                        node,
                        content,
                        json.dumps(payload) if payload else None,
                    ),
                )
            self.database.commit()
        except Exception:
            self.database.rollback()
            raise

    def get_by_id(self, id: str) -> tuple[RequestRecord, list[TraceEvent]] | None:
        row = self.database.execute(
            "SELECT id, created_at, finished_at, status, http_status, "
            "conversation_id, user_id, metrics_json, latency_ms, llm_calls, "
            "route, model_used FROM requests WHERE id = ?",
            (id,),
        ).fetchone()
        if not row:
            return None
        event_rows = self.database.execute(
            "SELECT type, node, content, payload_json FROM trace_events "
            "WHERE request_id = ? ORDER BY seq ASC",
            (id,),
        ).fetchall()
        return _row_to_request(row), [_row_to_trace_event(*r) for r in event_rows]

    def stats(self, since_ms: int) -> RequestStatsSummary:
        rows = self.database.execute(
            "SELECT id, created_at, finished_at, status, http_status, "
            "conversation_id, user_id, metrics_json, latency_ms, llm_calls, "
            "route, model_used FROM requests WHERE created_at >= ? ORDER BY created_at ASC",
            (since_ms,),
        ).fetchall()

        latencies: list[float] = []
        total = 0
        errors = 0
        tokens = 0
        cost_usd = 0.0
        by_route: dict[str, RequestStatsBucket] = {}
        by_model: dict[str, RequestStatsBucket] = {}

        def bump(
            bucket_map: dict[str, RequestStatsBucket],
            key: str,
            err: bool,
            tok: int,
            cost: float,
        ) -> None:
            bucket = bucket_map.setdefault(key, RequestStatsBucket())
            bucket.total += 1
            if err:
                bucket.errors += 1
            bucket.tokens += tok
            bucket.cost_usd += cost

        for row in rows:
            total += 1
            status, http_status = row[3], row[4]
            is_error = status == "error" or http_status >= 400
            if is_error:
                errors += 1
            latency_ms = row[8]
            if isinstance(latency_ms, int):
                latencies.append(latency_ms)

            prompt_tokens = 0
            try:
                metrics = json.loads(row[7])
                pt = metrics.get("prompt_tokens")
                if isinstance(pt, (int, float)) and pt > 0:
                    prompt_tokens = int(pt)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

            model = row[11]
            route = row[10]
            cost = estimate_prompt_cost_usd(model, prompt_tokens)
            tokens += prompt_tokens
            cost_usd += cost

            bump(by_route, route or "unknown", is_error, prompt_tokens, cost)
            bump(by_model, model or "unknown", is_error, prompt_tokens, cost)

        for bucket_map in (by_route, by_model):
            for bucket in bucket_map.values():
                bucket.cost_usd = round(bucket.cost_usd, 6)

        return RequestStatsSummary(
            total=total,
            errors=errors,
            tokens=tokens,
            cost_usd=round(cost_usd, 6),
            latency={"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
            by_route=by_route,
            by_model=by_model,
        )
