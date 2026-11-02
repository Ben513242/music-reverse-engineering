"""Resumable, cache-aware stage runner with partial-failure semantics.

* ``--resume`` skips stages that are complete, whose parameter hash is unchanged and whose
  upstream stages did not re-run in this invocation.
* ``--force-stage X`` invalidates X and every stage downstream of it.
* A failing optional stage marks the project PARTIAL and downstream stages are skipped; a
  failing *required* stage (ingest / normalization) marks the project FAILED.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mrep import __version__
from mrep.adapters.base import AdapterInfo, BaseAdapter
from mrep.adapters.registry import create_adapter
from mrep.config import config_hash, stage_config, stage_enabled, stage_params_hash
from mrep.domain.project import Project
from mrep.domain.provenance import Provenance
from mrep.logging import StageLogger, get_logger
from mrep.manifest import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    AdapterRecord,
    Manifest,
)
from mrep.utils.device import gpu_memory_info, resolve_device
from mrep.utils.hashing import short_hash
from mrep.utils.timing import Timer, utc_now_iso
from mrep.utils.timing import run_id as make_run_id

STAGE_ORDER: list[str] = [
    "ingest",
    "normalization",
    "global_analysis",
    "separation",
    "classification",
    "transcription",
    "fusion",
    "scoring",
    "reconstruction",
]
STAGE_DEPS: dict[str, list[str]] = {
    "ingest": [],
    "normalization": ["ingest"],
    "global_analysis": ["normalization"],
    "separation": ["normalization"],
    "classification": ["separation"],
    "transcription": ["classification", "global_analysis"],
    "fusion": ["transcription"],
    "scoring": ["fusion", "global_analysis"],
    "reconstruction": ["scoring"],
}
REQUIRED_STAGES: set[str] = {"ingest", "normalization"}


def downstream_of(stage: str) -> list[str]:
    out: set[str] = set()
    frontier = [stage]
    while frontier:
        cur = frontier.pop()
        for name, deps in STAGE_DEPS.items():
            if cur in deps and name not in out:
                out.add(name)
                frontier.append(name)
    return [s for s in STAGE_ORDER if s in out]


def upstream_closure(stages: list[str]) -> list[str]:
    out: set[str] = set()
    frontier = list(stages)
    while frontier:
        cur = frontier.pop()
        if cur in out:
            continue
        out.add(cur)
        frontier.extend(STAGE_DEPS[cur])
    return [s for s in STAGE_ORDER if s in out]


@dataclass
class StageResult:
    artifacts: dict[str, str] = field(default_factory=dict)  # key -> relative path
    summary: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapters: list[AdapterInfo] = field(default_factory=list)
    partial: bool = False

    def add_error(self, where: str, exc: BaseException) -> None:
        self.errors.append({"where": where, "type": type(exc).__name__, "message": str(exc)})
        self.partial = True


@dataclass
class PipelineContext:
    project: Project
    config: dict[str, Any]
    manifest: Manifest
    run_id: str
    device: str
    source_path: Path | None = None
    log: Any = None
    _adapters: dict[tuple[str, str], BaseAdapter] = field(default_factory=dict, repr=False)
    cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def stage_cfg(self, stage: str) -> dict[str, Any]:
        return stage_config(self.config, stage)

    def adapter(self, kind: str, name: str, cfg: dict[str, Any] | None = None) -> BaseAdapter:
        key = (kind, name)
        if key not in self._adapters:
            self._adapters[key] = create_adapter(kind, name, cfg or {}, self.device)
        return self._adapters[key]

    def provenance(self, stage: str, adapter: BaseAdapter, input_name: str) -> Provenance:
        return adapter.provenance(stage, self.run_id, input_name, short_hash(adapter.config))

    def logger(self, stage: str) -> StageLogger:
        return StageLogger(get_logger("pipeline"), {"project_id": self.manifest.project_id, "stage": stage, "device": self.device})

    def rel(self, path: Path) -> str:
        return self.project.rel(path)


StageFn = Callable[[PipelineContext], StageResult]


class StageRunner:
    def __init__(self, ctx: PipelineContext, stages: dict[str, StageFn]):
        self.ctx = ctx
        self.stages = stages
        self._ran_this_invocation: set[str] = set()

    # -- helpers ---------------------------------------------------------------------
    def _upstream_fingerprint(self, stage: str) -> str:
        parts = []
        for dep in STAGE_DEPS[stage]:
            rec = self.ctx.manifest.stage(dep)
            parts.append((dep, rec.status, rec.run_id, rec.params_hash, rec.finished_at))
        return short_hash(parts)

    def _cache_valid(self, stage: str) -> bool:
        rec = self.ctx.manifest.stage(stage)
        if not rec.is_done:
            return False
        if rec.params_hash != stage_params_hash(self.ctx.config, stage):
            return False
        if rec.upstream_fingerprint != self._upstream_fingerprint(stage):
            return False
        if any(dep in self._ran_this_invocation for dep in STAGE_DEPS[stage]):
            return False
        # all artifacts must still exist on disk
        for key in rec.artifacts:
            art = self.ctx.manifest.artifacts.get(key)
            if art is None or not self.ctx.project.abs(art.path).exists():
                return False
        return True

    # -- main ----------------------------------------------------------------------------
    def run(self, targets: list[str] | None = None, resume: bool = False, force: list[str] | None = None) -> Manifest:
        ctx = self.ctx
        man = ctx.manifest
        targets = targets or list(STAGE_ORDER)
        plan = upstream_closure(targets)
        if force:
            invalid: list[str] = []
            for f in force:
                invalid.extend([f] + downstream_of(f))
            man.invalidate(sorted(set(invalid), key=STAGE_ORDER.index))
        man.runs.append({"run_id": ctx.run_id, "started_at": utc_now_iso(), "targets": targets, "resume": resume, "force": list(force or [])})
        man.status = "RUNNING"
        man.save(ctx.project.manifest_path)

        for stage in plan:
            rec = man.stage(stage)
            log = ctx.logger(stage)
            if not stage_enabled(ctx.config, stage):
                rec.status = STATUS_SKIPPED
                rec.summary = {"reason": "disabled in config"}
                log.info("skipped (disabled)")
                continue
            failed_deps = [d for d in STAGE_DEPS[stage] if man.stage(d).status in (STATUS_FAILED, STATUS_SKIPPED) and stage_enabled(ctx.config, d)]
            if failed_deps:
                rec.status = STATUS_SKIPPED
                rec.summary = {"reason": f"upstream failed/skipped: {failed_deps}"}
                rec.finished_at = utc_now_iso()
                log.warning("skipped: upstream %s not available", failed_deps)
                man.save()
                continue
            if resume and self._cache_valid(stage):
                rec.cache_hit = True
                log.info("cache hit, skipping", extra={"cache_hit": True})
                continue
            rec.cache_hit = False
            rec.status = STATUS_RUNNING
            rec.started_at = utc_now_iso()
            rec.run_id = ctx.run_id
            rec.device = ctx.device
            rec.errors = []
            rec.warnings = []
            rec.artifacts = []
            man.save()
            log.info("start")
            with Timer() as timer:
                try:
                    result = self.stages[stage](ctx)
                except Exception as exc:  # noqa: BLE001 - a stage crash must not kill the run
                    rec.status = STATUS_FAILED
                    rec.errors.append({"where": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                    man.errors.append({"stage": stage, "type": type(exc).__name__, "message": str(exc), "run_id": ctx.run_id})
                    log.error("failed: %s", exc, extra={"error": str(exc)})
                    result = None
            rec.runtime_sec = round(timer.elapsed, 3)
            rec.finished_at = utc_now_iso()
            rec.params_hash = stage_params_hash(ctx.config, stage)
            rec.upstream_fingerprint = self._upstream_fingerprint(stage)
            rec.vram = gpu_memory_info(ctx.device)
            self._ran_this_invocation.add(stage)
            if result is not None:
                for key, rel_path in result.artifacts.items():
                    man.add_artifact(key, rel_path, stage)
                rec.summary = result.summary
                rec.confidence = result.confidence
                rec.errors.extend(result.errors)
                rec.warnings.extend(result.warnings)
                rec.adapters = [AdapterRecord(kind=a.kind, name=a.name, model=a.model, model_version=a.model_version, device=a.device) for a in result.adapters]
                rec.status = STATUS_PARTIAL if result.partial else STATUS_COMPLETE
                if result.confidence:
                    man.confidence[stage] = result.confidence
                log.info("done", extra={"runtime": rec.runtime_sec, "model": ",".join(a.name for a in result.adapters) or None, "output": list(result.artifacts)[:6]})
            man.compute_status(REQUIRED_STAGES)
            man.save()
            if rec.status == STATUS_FAILED and stage in REQUIRED_STAGES:
                log.error("required stage failed; aborting run")
                break
        man.compute_status(REQUIRED_STAGES)
        man.runs[-1]["finished_at"] = utc_now_iso()
        man.runs[-1]["status"] = man.status
        man.save()
        return man


def build_runner(project: Project, config: dict[str, Any], source_path: Path | None = None, manifest: Manifest | None = None) -> StageRunner:
    """Wire the standard stages into a runner for ``project``."""
    from mrep.pipeline import (
        classification,
        fusion,
        global_analysis,
        ingest,
        normalization,
        reconstruction,
        scoring,
        separation,
        transcription,
    )

    project.ensure_dirs()
    device = resolve_device(config.get("device"))
    if manifest is None:
        manifest = Manifest.load(project.manifest_path) if project.manifest_path.exists() else Manifest(song_name=project.song_name)
    manifest.song_name = project.song_name
    manifest.config = config
    manifest.config_hash = config_hash(config)
    manifest.profile = str(config.get("profile", "default"))
    manifest.device = device
    manifest.mrep_version = __version__
    if source_path is not None:
        manifest.source.setdefault("path", str(Path(source_path).resolve()))
    ctx = PipelineContext(project=project, config=config, manifest=manifest, run_id=make_run_id(), device=device, source_path=source_path)
    stages: dict[str, StageFn] = {
        "ingest": ingest.run,
        "normalization": normalization.run,
        "global_analysis": global_analysis.run,
        "separation": separation.run,
        "classification": classification.run,
        "transcription": transcription.run,
        "fusion": fusion.run,
        "scoring": scoring.run,
        "reconstruction": reconstruction.run,
    }
    return StageRunner(ctx, stages)
