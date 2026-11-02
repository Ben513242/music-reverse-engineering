"""Project manifest: the single source of truth about a run (status, artifacts, provenance)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mrep.io.json import read_json, write_json
from mrep.utils.timing import utc_now_iso

SCHEMA_VERSION = "1.0"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

PROJECT_PENDING = "PENDING"
PROJECT_RUNNING = "RUNNING"
PROJECT_COMPLETE = "COMPLETE"
PROJECT_PARTIAL = "PARTIAL"
PROJECT_FAILED = "FAILED"


@dataclass
class ArtifactRecord:
    key: str
    path: str
    stage: str
    kind: str = "file"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterRecord:
    kind: str
    name: str
    model: str | None = None
    model_version: str | None = None
    device: str = "cpu"


@dataclass
class StageRecord:
    name: str
    status: str = STATUS_PENDING
    started_at: str | None = None
    finished_at: str | None = None
    runtime_sec: float | None = None
    run_id: str | None = None
    params_hash: str | None = None
    upstream_fingerprint: str | None = None
    cache_hit: bool = False
    adapters: list[AdapterRecord] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    vram: dict[str, float] | None = None

    @property
    def is_done(self) -> bool:
        return self.status in (STATUS_COMPLETE, STATUS_PARTIAL)


@dataclass
class Manifest:
    project_id: str = ""
    song_name: str = ""
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    status: str = PROJECT_PENDING
    source: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    profile: str = "default"
    device: str = "cpu"
    mrep_version: str = ""
    stages: dict[str, StageRecord] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = field(default=None, repr=False, compare=False)

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("path", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        d = dict(d)
        stages = {k: StageRecord(**{**v, "adapters": [AdapterRecord(**a) for a in v.get("adapters", [])]}) for k, v in d.pop("stages", {}).items()}
        artifacts = {k: ArtifactRecord(**v) for k, v in d.pop("artifacts", {}).items()}
        d.pop("path", None)
        allowed = set(cls.__dataclass_fields__)
        m = cls(**{k: v for k, v in d.items() if k in allowed})
        m.stages = stages
        m.artifacts = artifacts
        return m

    @classmethod
    def load(cls, path: Path) -> Manifest:
        m = cls.from_dict(read_json(path))
        m.path = Path(path)
        return m

    def save(self, path: Path | None = None) -> Path:
        self.updated_at = utc_now_iso()
        target = Path(path or self.path or "manifest.json")
        self.path = target
        write_json(target, self.to_dict())
        return target

    # ---- stage bookkeeping ---------------------------------------------------------
    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def invalidate(self, names: list[str]) -> None:
        for n in names:
            rec = self.stage(n)
            rec.status = STATUS_PENDING
            rec.cache_hit = False
            rec.upstream_fingerprint = None

    def add_artifact(self, key: str, path: str, stage: str, kind: str = "file", **meta: Any) -> None:
        self.artifacts[key] = ArtifactRecord(key=key, path=path, stage=stage, kind=kind, meta=dict(meta))
        rec = self.stage(stage)
        if key not in rec.artifacts:
            rec.artifacts.append(key)

    def artifact_path(self, key: str) -> str | None:
        rec = self.artifacts.get(key)
        return rec.path if rec else None

    def compute_status(self, required: set[str] | None = None) -> str:
        required = required or set()
        statuses = {name: rec.status for name, rec in self.stages.items()}
        if any(statuses.get(n) == STATUS_FAILED for n in required):
            self.status = PROJECT_FAILED
        elif any(s == STATUS_RUNNING for s in statuses.values()):
            self.status = PROJECT_RUNNING
        elif any(s in (STATUS_FAILED, STATUS_PARTIAL) for s in statuses.values()):
            self.status = PROJECT_PARTIAL
        elif statuses and all(s in (STATUS_COMPLETE, STATUS_SKIPPED) for s in statuses.values()):
            self.status = PROJECT_COMPLETE
        else:
            self.status = PROJECT_PENDING
        return self.status

    def summary(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "song": self.song_name,
            "status": self.status,
            "profile": self.profile,
            "device": self.device,
            "stages": {n: {"status": r.status, "runtime_sec": r.runtime_sec, "cache_hit": r.cache_hit, "errors": len(r.errors)} for n, r in self.stages.items()},
            "artifacts": len(self.artifacts),
        }
