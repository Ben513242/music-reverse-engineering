"""Stage 6b - multi-label classification of every stem (source / role / frequency / timbre)."""

from __future__ import annotations

from mrep.adapters.base import ClassificationAdapter
from mrep.domain.labels import SoundLabel
from mrep.io.json import read_json, write_json
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.pipeline.separation import load_stem_audio, load_stems

LABELS_FILE = "labels.json"


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("classification")
    cfg = ctx.stage_cfg("classification")
    result = StageResult()
    adapter: ClassificationAdapter = ctx.adapter("classification", cfg.get("adapter", "heuristic"), cfg)  # type: ignore[assignment]
    result.adapters.append(adapter.info())
    labels: dict[str, dict] = {}
    for stem in load_stems(ctx):
        try:
            audio, sr = load_stem_audio(ctx, stem)
            prov = ctx.provenance("classification", adapter, f"stem:{stem.name}")
            cres = adapter.classify(audio, sr, stem_name=stem.name, source_hint=stem.source_hint, provenance=prov)
            problems = cres.label.validate()
            if problems:
                result.warnings.append(f"{stem.name}: {problems}")
            labels[stem.name] = cres.label.to_dict()
            log.info("%s -> %s %s %s", stem.name, cres.label.source_family, cres.label.musical_role, cres.label.frequency_role)
        except Exception as exc:  # noqa: BLE001
            result.add_error(f"classify:{stem.name}", exc)
            labels[stem.name] = SoundLabel(source_family="unknown", confidence=None, confidence_method="unavailable").to_dict()
    out = ctx.project.events_dir / LABELS_FILE
    write_json(out, labels)
    result.artifacts["labels"] = ctx.rel(out)
    # merge into instruments.json
    inst_path = ctx.project.analysis_dir / "instruments.json"
    inst = read_json(inst_path) if inst_path.exists() else {}
    inst["labels"] = {k: {kk: v[kk] for kk in ("source_family", "source_subtype", "musical_role", "frequency_role", "sound_design", "confidence", "confidence_method")} for k, v in labels.items()}
    write_json(inst_path, inst)
    result.artifacts["instruments"] = ctx.rel(inst_path)
    result.summary = {name: {"family": lab["source_family"], "roles": lab["musical_role"]} for name, lab in labels.items()}
    result.confidence = {name: lab["confidence"] for name, lab in labels.items()}
    return result


def load_labels(ctx: PipelineContext) -> dict[str, SoundLabel]:
    path = ctx.project.events_dir / LABELS_FILE
    if not path.exists():
        return {}
    return {k: SoundLabel.from_dict(v) for k, v in read_json(path).items()}
