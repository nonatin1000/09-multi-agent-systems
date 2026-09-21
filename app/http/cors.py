"""Middleware CORS (equivalente a ``src/http/cors.ts``)."""

from __future__ import annotations

import os
from typing import Literal

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

#: Sentinela: reflete qualquer Origin do browser (default dev/demo).
CORS_ALLOW_ALL: Literal["*"] = "*"


def _parse_origins(raw: str | None) -> list[str] | Literal["*"]:
    if raw is None or raw.strip() == "" or raw.strip() == "*":
        return CORS_ALLOW_ALL
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def resolve_cors_origins(env_value: str | None = None) -> list[str] | Literal["*"]:
    """Default: permite todas as origens (``*``). Defina ``OPSPILOT_CORS_ORIGINS``
    com uma allowlist separada por vírgula para restringir (ex.:
    ``http://localhost:5173,http://127.0.0.1:5173``)."""
    value = env_value if env_value is not None else os.environ.get("OPSPILOT_CORS_ORIGINS")
    return _parse_origins(value)


def _apply_cors_headers(response: Response, origin: str | None, allow_all: bool) -> None:
    if allow_all:
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
    elif origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        return
    if origin:
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Request-Id"
    response.headers["Access-Control-Expose-Headers"] = "X-Request-Id"
    response.headers["Access-Control-Max-Age"] = "86400"


class CorsMiddleware:
    """Middleware CORS ASGI. Default: permite todas as origens; passe uma
    allowlist para restringir (ou use ``resolve_cors_origins()`` para o
    default do env)."""

    def __init__(self, app: ASGIApp, origins: list[str] | Literal["*"] | None = None) -> None:
        self._app = app
        resolved = origins if origins is not None else resolve_cors_origins()
        self._allow_all = resolved == CORS_ALLOW_ALL
        self._allowlist = None if self._allow_all else set(resolved)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        origin_header = request.headers.get("origin")
        origin = origin_header.rstrip("/") if origin_header else None

        allowed = self._allow_all or (
            origin is not None and self._allowlist is not None and origin in self._allowlist
        )

        if request.method == "OPTIONS":
            response = Response(status_code=204)
            if allowed:
                _apply_cors_headers(response, origin, self._allow_all)
            await response(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and allowed:
                response = Response()
                _apply_cors_headers(response, origin, self._allow_all)
                message["headers"] = list(message.get("headers", [])) + [
                    (k.encode("latin-1"), v.encode("latin-1"))
                    for k, v in response.headers.items()
                ]
            await send(message)

        await self._app(scope, receive, send_wrapper)
