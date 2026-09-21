"""Consulta statuspage pública de provedores externos (sem chave)
(equivalente a ``src/tools/check-provider-status.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

import httpx

PROVIDER_URLS: dict[str, str] = {
    "github": "https://www.githubstatus.com/api/v2/status.json",
    "cloudflare": "https://www.cloudflarestatus.com/api/v2/status.json",
}

ProviderId = Literal["github", "cloudflare"]

#: Assinatura injetável para testes; default usa httpx.AsyncClient.
FetchFn = Callable[[str], Awaitable[httpx.Response]]


def format_provider_status(provider: str, indicator: str, description: str) -> str:
    return f"{provider} está {indicator} - {description}"


def _failure_message(provider: str, detail: str) -> str:
    return (
        f"não consegui consultar o status de {provider} ({detail}). "
        "Responda com base nos alertas internos e avise o plantonista da limitação"
    )


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, httpx.TimeoutException):
        return True
    if isinstance(error, httpx.TransportError):
        return True
    if str(error).startswith("upstream "):
        return True
    return not isinstance(error, ValueError)


async def _default_fetch(url: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get(url)


async def fetch_provider_status(
    provider: ProviderId, fetch: FetchFn | None = None
) -> str:
    """Consulta a statuspage pública do provedor. Sempre resolve com string
    (sucesso compacto ou erro legível) — nunca levanta por falha operacional."""
    do_fetch = fetch or _default_fetch
    url = PROVIDER_URLS[provider]
    last_detail = "unknown error"

    for attempt in range(1, 3):
        try:
            response = await do_fetch(url)

            if response.status_code >= 500:
                raise ValueError(f"upstream {response.status_code}")

            if response.status_code >= 400:
                return f"status page de {provider} respondeu HTTP {response.status_code}"

            try:
                payload = response.json()
            except Exception as error:  # noqa: BLE001
                return _failure_message(provider, f"invalid JSON: {error}")

            status = payload.get("status") if isinstance(payload, dict) else None
            indicator = status.get("indicator") if isinstance(status, dict) else None
            description = status.get("description") if isinstance(status, dict) else None
            if not isinstance(indicator, str) or not isinstance(description, str):
                return _failure_message(provider, "invalid statuspage response")

            return format_provider_status(provider, indicator, description)
        except Exception as error:  # noqa: BLE001
            last_detail = str(error)
            if attempt == 2 or not _is_retryable_error(error):
                return _failure_message(provider, last_detail)

    return _failure_message(provider, last_detail)
