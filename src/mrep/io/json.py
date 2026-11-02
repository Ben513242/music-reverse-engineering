"""JSON / JSONL helpers that understand numpy types, paths and domain objects."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from mrep.domain.events import Event, event_from_dict


class MrepJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


def write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=MrepJSONEncoder, indent=indent, ensure_ascii=False)
        fh.write("\n")
    return path


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: Path, items: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, cls=MrepJSONEncoder, ensure_ascii=False))
            fh.write("\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_events(path: Path, events: Iterable[Event]) -> Path:
    return write_jsonl(path, (e.to_dict() for e in events))


def read_events(path: Path) -> list[Event]:
    return [event_from_dict(d) for d in read_jsonl(path)]
