import uuid

import pytest
from fastapi.testclient import TestClient

from app.domain.types import (
    ExecutionMetrics,
    ReasoningStrategy,
    StrategyResult,
    StrategyRunInput,
    TraceEvent,
)
from app.graph.production_graph import ProductionStrategies
from app.graph.router import RouterDecision
from app.http.server import ChatAppDeps, create_app
from app.memory.fake_embedder import FakeEmbedder
from app.memory.memory_store import SqliteMemoryStore
from app.store.sqlite_conversation_store import SqliteConversationStore
from app.store.sqlite_request_store import SqliteRequestStore

_DEFAULT_ANSWER = "**Resumo**\nok\n**Achados**\n- nada\n**Próximos passos**\nNenhum"


class _FakeStrategy(ReasoningStrategy):
    def __init__(self, name: str, answer: str = _DEFAULT_ANSWER):
        self.name = name
        self._answer = answer

    async def run(self, input: StrategyRunInput) -> StrategyResult:
        return StrategyResult(
            answer=self._answer,
            trace=[TraceEvent(type="answer", content=self._answer, node=self.name)],
            metrics=ExecutionMetrics(llm_calls=1, latency_ms=5),
        )


async def _fake_classify(message: str) -> RouterDecision:
    return RouterDecision(route="react", reason="fake router")


@pytest.fixture
def client() -> TestClient:
    conversations = SqliteConversationStore(":memory:")
    memories = SqliteMemoryStore(":memory:", embedder=FakeEmbedder())
    requests = SqliteRequestStore(":memory:")
    strategies = ProductionStrategies(
        react=_FakeStrategy("react"),
        plan_execute=_FakeStrategy("planExecute"),
        reflect=_FakeStrategy("reflect"),
        team=_FakeStrategy("team"),
    )
    app = create_app(
        ChatAppDeps(
            conversations=conversations,
            memories=memories,
            strategies=strategies,
            classify_route=_fake_classify,
            requests=requests,
        )
    )
    return TestClient(app)


def test_chat_happy_path_returns_answer_trace_metrics(client: TestClient):
    response = client.post("/chat", json={"message": "liste alertas", "userId": "u1"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["trace"]
    assert body["metrics"]["route"] == "react"
    assert "X-Request-Id" in response.headers


def test_chat_unknown_strategy_returns_422(client: TestClient):
    response = client.post(
        "/chat", json={"message": "oi", "userId": "u1", "strategy": "nope"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "unknown_strategy"


def test_chat_missing_message_returns_400(client: TestClient):
    response = client.post("/chat", json={"userId": "u1"})
    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"


def test_get_request_not_found_returns_404(client: TestClient):
    response = client.get(f"/requests/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"] == "request_not_found"


def test_stats_returns_zeroed_summary_when_empty(client: TestClient):
    response = client.get("/stats", params={"since": "24h"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["latency"] == {"p50": None, "p95": None}


def test_remember_endpoint_stores_fact(client: TestClient):
    response = client.post("/memories", json={"userId": "u1", "fact": "prioriza checkout"})
    assert response.status_code == 201
    assert response.json()["stored"] is True
