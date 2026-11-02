"""Provenance: *who* produced an inference result, with which model and parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Provenance:
    stage: str
    adapter: str
    model: str | None = None
    model_version: str | None = None
    run_id: str = ""
    input: str = ""
    parameters_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Provenance | None:
        if d is None:
            return None
        return cls(
            stage=str(d.get("stage", "")),
            adapter=str(d.get("adapter", "")),
            model=d.get("model"),
            model_version=d.get("model_version"),
            run_id=str(d.get("run_id", "")),
            input=str(d.get("input", "")),
            parameters_hash=str(d.get("parameters_hash", "")),
        )

    def derive(self, **changes: Any) -> Provenance:
        """Return a copy with some fields replaced (e.g. a different ``input``)."""
        data = self.to_dict()
        data.update(changes)
        return Provenance(**data)
