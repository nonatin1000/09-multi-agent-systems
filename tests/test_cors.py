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
