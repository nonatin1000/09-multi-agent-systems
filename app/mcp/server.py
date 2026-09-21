"""Entrypoint MCP stdio. stdout é reservado para o protocolo — use logging
para stderr apenas para diagnóstico. Equivalente a ``src/mcp/server.ts``."""

from __future__ import annotations

import sys

from app.mcp.create_server import create_ops_mcp_server
from app.store.seed import seed_ops_store
from app.store.sqlite_ops_store import SqliteOpsStore


def main() -> None:
    import os

    store = SqliteOpsStore(os.environ.get("OPSPILOT_DB", "./data/opspilot.db"))
    seed_ops_store(store)

    server = create_ops_mcp_server(store)
    print("opspilot MCP server: pronto (stdio)", file=sys.stderr)
    server.run(transport="stdio")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
