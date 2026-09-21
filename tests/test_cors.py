import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.http.cors import CorsMiddleware, resolve_cors_origins


def _make_app(origins=None):
    async def handler(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/ping", handler, methods=["GET", "OPTIONS"])])
    app.add_middleware(CorsMiddleware, origins=origins)
    return app


def test_resolve_cors_origins_defaults_to_allow_all():
    assert resolve_cors_origins(None) == "*"
    assert resolve_cors_origins("*") == "*"


def test_resolve_cors_origins_parses_allowlist():
    origins = resolve_cors_origins("http://a.com, http://b.com/")
    assert origins == ["http://a.com", "http://b.com"]


def test_cors_middleware_reflects_origin_when_allow_all():
    client = TestClient(_make_app())
    response = client.get("/ping", headers={"Origin": "http://foo.com"})
    assert response.headers["access-control-allow-origin"] == "http://foo.com"


def test_cors_middleware_restricts_to_allowlist():
    client = TestClient(_make_app(origins=["http://allowed.com"]))
    blocked = client.get("/ping", headers={"Origin": "http://blocked.com"})
    assert "access-control-allow-origin" not in blocked.headers

    allowed = client.get("/ping", headers={"Origin": "http://allowed.com"})
    assert allowed.headers["access-control-allow-origin"] == "http://allowed.com"


def test_cors_middleware_handles_preflight():
    client = TestClient(_make_app())
    response = client.options("/ping", headers={"Origin": "http://foo.com"})
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_cors_middleware_does_not_duplicate_content_length_header():
    """Regressão: Response() teria seu próprio content-length/content-type
    default, colidindo com o da resposta real e quebrando clientes HTTP reais
    (curl/httpx) com "conflicting Content-Length headers" — não pego pelo
    TestClient in-process, só em teste HTTP de verdade."""
    from app.http.cors import CorsMiddleware

    async def downstream_app(scope, receive, send):
        body = b'{"ok": true}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    middleware = CorsMiddleware(downstream_app, origins="*")
    sent_messages = []

    async def send(message):
        sent_messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/ping",
        "headers": [(b"origin", b"http://foo.com")],
    }
    await middleware(scope, receive, send)

    start = next(m for m in sent_messages if m["type"] == "http.response.start")
    content_length_headers = [v for k, v in start["headers"] if k == b"content-length"]
    assert len(content_length_headers) == 1
    content_type_headers = [v for k, v in start["headers"] if k == b"content-type"]
    assert len(content_type_headers) == 1
