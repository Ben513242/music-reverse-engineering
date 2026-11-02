"""Structured logging: console + JSON-lines file with project/stage/model/device context."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONTEXT_FIELDS = ("project_id", "stage", "model", "device", "runtime", "vram", "cache_hit", "output", "error")


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for f in _CONTEXT_FIELDS:
            if hasattr(record, f):
                payload[f] = getattr(record, f)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = " ".join(f"{f}={getattr(record, f)}" for f in ("stage", "model", "device", "runtime", "cache_hit") if hasattr(record, f))
        base = f"[{record.levelname:<7}] {record.getMessage()}"
        return f"{base}  ({ctx})" if ctx else base


def configure_logging(level: str = "INFO", json_file: Path | None = None) -> logging.Logger:
    root = logging.getLogger("mrep")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)
    if json_file is not None:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_file, encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        root.addHandler(fh)
    return root


def get_logger(name: str = "mrep") -> logging.Logger:
    return logging.getLogger(name if name.startswith("mrep") else f"mrep.{name}")


class StageLogger(logging.LoggerAdapter):
    """Logger adapter injecting project/stage context into every record."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", {}) or {})
        kwargs["extra"] = extra
        return msg, kwargs
