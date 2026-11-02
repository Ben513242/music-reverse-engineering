from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Context manager measuring wall-clock seconds."""

    start: float = field(default=0.0)
    elapsed: float = field(default=0.0)

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id() -> str:
    import secrets

    return f"run_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{secrets.token_hex(2)}"
