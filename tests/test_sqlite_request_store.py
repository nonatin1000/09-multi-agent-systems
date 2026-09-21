from app.domain.types import ExecutionMetrics, SaveRequestInput, TraceEvent
from app.store.sqlite_request_store import SqliteRequestStore


def _save(
    store: SqliteRequestStore,
    id: str,
    route: str,
    model: str,
    prompt_tokens: int,
    created_at: int,
):
    store.save(
        SaveRequestInput(
            id=id,
            created_at=created_at,
            finished_at=created_at + 100,
            status="success",
            http_status=200,
            metrics=ExecutionMetrics(
                llm_calls=1,
                latency_ms=100,
                prompt_tokens=prompt_tokens,
                route=route,
                model_used=model,
            ),
            trace=[
                TraceEvent(type="route", content="ok", node="roteador", route=route),
                TraceEvent(type="answer", content="resposta", node=route),
            ],
        )
    )


def test_save_and_get_by_id_round_trips_trace(sqlite_request_store: SqliteRequestStore):
    _save(sqlite_request_store, "r1", "react", "openai/gpt-4o-mini", 100, 1_000)

    found = sqlite_request_store.get_by_id("r1")
    assert found is not None
    record, trace = found
    assert record.id == "r1"
    assert record.route == "react"
    assert [e.type for e in trace] == ["route", "answer"]
    assert trace[0].route == "react"


def test_get_by_id_missing_returns_none(sqlite_request_store: SqliteRequestStore):
    assert sqlite_request_store.get_by_id("missing") is None


def test_stats_aggregates_by_route_and_model(sqlite_request_store: SqliteRequestStore):
    _save(sqlite_request_store, "r1", "react", "openai/gpt-4o-mini", 1_000_000, 1_000)
    _save(sqlite_request_store, "r2", "planExecute", "openai/gpt-4o-mini:free", 1_000_000, 2_000)

    summary = sqlite_request_store.stats(0)
    assert summary.total == 2
    assert summary.tokens == 2_000_000
    assert summary.by_route["react"].total == 1
    assert summary.by_route["planExecute"].total == 1
    assert summary.by_model["openai/gpt-4o-mini:free"].cost_usd == 0
    assert summary.cost_usd == 0.15


def test_stats_excludes_requests_before_since(sqlite_request_store: SqliteRequestStore):
    _save(sqlite_request_store, "old", "react", "m", 10, 100)
    summary = sqlite_request_store.stats(1_000)
    assert summary.total == 0
