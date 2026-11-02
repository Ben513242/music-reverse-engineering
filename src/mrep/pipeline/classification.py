"""Stage 6b - multi-label classification of every stem (source / role / frequency / timbre)."""

from __future__ import annotations

from mrep.adapters.base import AdapterUnavailable, ClassificationAdapter, SoundAnalysisAdapter
from mrep.domain.labels import SoundLabel
from mrep.io.json import read_json, write_json
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.pipeline.separation import load_stem_audio, load_stems

LABELS_FILE = "labels.json"
SOUND_DESIGN_FILE = "sound_design.json"


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("classification")
    cfg = ctx.stage_cfg("classification")
    result = StageResult()
    adapter: ClassificationAdapter = ctx.adapter("classification", cfg.get("adapter", "heuristic"), cfg)  # type: ignore[assignment]
    result.adapters.append(adapter.info())
    labels: dict[str, dict] = {}
    label_objs: dict[str, SoundLabel] = {}
    for stem in load_stems(ctx):
        try:
            audio, sr = load_stem_audio(ctx, stem)
            prov = ctx.provenance("classification", adapter, f"stem:{stem.name}")
            cres = adapter.classify(audio, sr, stem_name=stem.name, source_hint=stem.source_hint, provenance=prov)
            problems = cres.label.validate()
            if problems:
                result.warnings.append(f"{stem.name}: {problems}")
            label_objs[stem.name] = cres.label
            log.info("%s -> %s %s %s", stem.name, cres.label.source_family, cres.label.musical_role, cres.label.frequency_role)
        except Exception as exc:  # noqa: BLE001
            result.add_error(f"classify:{stem.name}", exc)
            label_objs[stem.name] = SoundLabel(source_family="unknown", confidence=None, confidence_method="unavailable")

    # ---- sound design / timbre analysis of pitched stems ----------------------------
    sa_cfg = cfg.get("sound_analysis", {}) or {}
    sound_design: dict[str, dict] = {}
    if sa_cfg.get("enabled", True):
        try:
            sa: SoundAnalysisAdapter = ctx.adapter("sound_analysis", sa_cfg.get("adapter", "dsp"), sa_cfg)  # type: ignore[assignment]
            result.adapters.append(sa.info())
            stems_by_name = {st.name: st for st in load_stems(ctx)}
            for name, label in label_objs.items():
                if label.source_family == "drums" or name.startswith("drums"):
                    continue
                if not (label.is_pitched or label.source_family == "unknown"):
                    continue
                try:
                    audio, sr = load_stem_audio(ctx, stems_by_name[name])
                    sprov = ctx.provenance("classification", sa, f"stem:{name}")
                    res = sa.analyze_sound(audio, sr, stem_name=name, label=label, provenance=sprov)
                    label.sound_design = sorted(set(label.sound_design) | set(res.sound_design))
                    label.patch_candidates = list(res.patch_candidates)
                    label.features["sound_analysis"] = res.features
                    sound_design[name] = {"features": res.features, "sound_design": res.sound_design, "patch_candidates": [c.to_dict() for c in res.patch_candidates], "confidence": res.confidence, "confidence_method": res.confidence_method, "adapter": sa.name, "provenance": sprov.to_dict()}
                except Exception as exc:  # noqa: BLE001
                    result.add_error(f"sound_analysis:{name}", exc)
            write_json(ctx.project.analysis_dir / SOUND_DESIGN_FILE, sound_design)
            result.artifacts["sound_design"] = ctx.rel(ctx.project.analysis_dir / SOUND_DESIGN_FILE)
        except AdapterUnavailable as exc:
            result.warnings.append(f"sound_analysis adapter unavailable: {exc}")

    labels = {name: lab.to_dict() for name, lab in label_objs.items()}
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
