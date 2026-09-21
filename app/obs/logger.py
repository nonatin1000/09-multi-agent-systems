"""Logger estruturado: uma linha JSON por evento, só metadados
(equivalente a ``src/obs/logger.ts``)."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.domain.types import LogLevel, LogMeta

_FORBIDDEN_META_KEYS = {
    "message",
    "answer",
    "trace",
    "content",
    "payload",
    "toolArgs",
    "body",
    "prompt",
}


def _sanitize_meta(meta: LogMeta | None) -> LogMeta:
    if not meta:
        return {}
    out: LogMeta = {}
    for key, value in meta.items():
        if key in _FORBIDDEN_META_KEYS:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


def _default_write(line: str) -> None:
    sys.stdout.write(line)


@dataclass(slots=True)
class Logger:
    """Logger estruturado; ``write``/``now`` injetáveis para testes."""

    write: Callable[[str], None] = _default_write
    now: Callable[[], int] = lambda: int(time.time() * 1000)

    def _emit(self, level: LogLevel, event: str, meta: LogMeta | None = None) -> None:
        line = json.dumps(
            {"ts": self.now(), "level": level, "event": event, **_sanitize_meta(meta)}
        )
        self.write(f"{line}\n")

    def info(self, event: str, meta: LogMeta | None = None) -> None:
        self._emit("info", event, meta)

    def warn(self, event: str, meta: LogMeta | None = None) -> None:
        self._emit("warn", event, meta)

    def error(self, event: str, meta: LogMeta | None = None) -> None:
        self._emit("error", event, meta)


def create_logger(
    write: Callable[[str], None] | None = None,
    now: Callable[[], int] | None = None,
) -> Logger:
    kwargs: dict[str, object] = {}
    if write is not None:
        kwargs["write"] = write
    if now is not None:
        kwargs["now"] = now
    return Logger(**kwargs)
