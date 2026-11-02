"""Stage 3 - global music analysis (tempo, beats, downbeats, key, structure, loudness)."""

from __future__ import annotations

from mrep.adapters.base import AdapterUnavailable, MIRAdapter
from mrep.domain.events import BeatEvent, KeyEvent, SectionEvent, TempoEvent
from mrep.io.json import read_json, write_events, write_json
from mrep.pipeline.normalization import load_normalized
from mrep.pipeline.runner import PipelineContext, StageResult


def _pick_adapter(ctx: PipelineContext, kind: str, name: str, fallback: str | None, cfg: dict, result: StageResult):
    try:
        return ctx.adapter(kind, name, cfg)
    except AdapterUnavailable as exc:
        if fallback and fallback != name:
            result.warnings.append(f"{kind}/{name} unavailable ({exc}); falling back to {fallback}")
            return ctx.adapter(kind, fallback, cfg)
        raise


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("global_analysis")
    cfg = ctx.stage_cfg("global_analysis")
    result = StageResult()
    adapter: MIRAdapter = _pick_adapter(ctx, "mir", cfg.get("adapter", "dsp"), cfg.get("fallback_adapter"), cfg, result)  # type: ignore[assignment]
    result.adapters.append(adapter.info())
    data, sr = load_normalized(ctx)
    mir = adapter.analyze(data, sr)
    prov = ctx.provenance("global_analysis", adapter, "normalized_audio")
    out = ctx.project.analysis_dir
    duration = len(data) / sr

    tempo = {
        "tempo": mir.tempo_bpm,
        "confidence": mir.tempo_confidence,
        "confidence_method": mir.tempo_method,
        "time_signature": mir.time_signature,
        "time_signature_confidence": mir.time_signature_confidence,
        "time_signature_method": mir.time_signature_method,
        "beats_per_bar": mir.beats_per_bar,
        "time_signature_scores": mir.extra.get("time_signature_scores", {}),
        "tempo_events": [TempoEvent(time=0.0, bpm=mir.tempo_bpm or 120.0, confidence=mir.tempo_confidence, confidence_method=mir.tempo_method, provenance=prov).to_dict()],
        "provenance": prov.to_dict(),
    }
    write_json(out / "tempo.json", tempo)

    beats = []
    downbeats = set(round(b, 4) for b in mir.downbeat_times)
    pos = 0
    for i, t in enumerate(mir.beat_times):
        is_db = round(t, 4) in downbeats
        if is_db:
            pos = 1
        elif i == 0:
            pos = 1
        else:
            pos = pos % mir.beats_per_bar + 1 if pos else 1
        beats.append(BeatEvent(time=t, position=pos, is_downbeat=is_db, confidence=mir.tempo_confidence, confidence_method=mir.tempo_method, provenance=prov))
    write_json(out / "beat.json", {"beats": [b.to_dict() for b in beats], "downbeats": list(mir.downbeat_times), "beats_per_bar": mir.beats_per_bar})

    key_events = [KeyEvent(start=k["start"], end=k.get("end"), tonic=k["tonic"], mode=k["mode"], confidence=k.get("confidence"), confidence_method=k.get("confidence_method", "unavailable"), provenance=prov) for k in mir.key_events]
    write_json(out / "key.json", {"key_events": [k.to_dict() for k in key_events], "global_key": key_events[0].key if key_events else None, "scores": mir.extra.get("key_scores", {})})

    sections = [SectionEvent(start=s["start"], end=s["end"], label=s["label"], confidence=s.get("confidence"), confidence_method=s.get("confidence_method", "unavailable"), provenance=prov) for s in mir.sections]
    dyn_by_start = {round(d["start"], 3): d for d in mir.dynamics}
    for s in sections:
        d = dyn_by_start.get(round(s.start, 3))
        if d:
            s.dynamic = d.get("dynamic")
            s.loudness_db = d.get("rms_db")
    write_events(out / "structure.jsonl", sections)
    write_json(out / "structure.json", {"sections": [s.to_dict() for s in sections]})
    write_json(out / "dynamics.json", {"sections": mir.dynamics, "loudness": mir.loudness, "provenance": prov.to_dict()})
    write_json(out / "loudness.json", mir.loudness)

    artifacts = {k: ctx.rel(out / f) for k, f in (("tempo", "tempo.json"), ("beat", "beat.json"), ("key", "key.json"), ("structure", "structure.json"), ("dynamics", "dynamics.json"), ("loudness", "loudness.json"))}
    result.artifacts.update(artifacts)
    result.summary = {"tempo": mir.tempo_bpm, "beats": len(beats), "time_signature": mir.time_signature, "key": key_events[0].key if key_events else None, "key_changes": max(0, len(key_events) - 1), "sections": len(sections), "duration": duration}
    result.confidence = {"tempo": mir.tempo_confidence, "key": key_events[0].confidence if key_events else None, "time_signature": mir.time_signature_confidence}
    log.info("tempo %.2f bpm, key %s, %d sections", mir.tempo_bpm or 0.0, result.summary["key"], len(sections))
    return result


def load_global(ctx: PipelineContext) -> dict:
    """Load the global analysis outputs needed by later stages."""
    out = ctx.project.analysis_dir
    tempo = read_json(out / "tempo.json")
    beat = read_json(out / "beat.json")
    key = read_json(out / "key.json")
    structure = read_json(out / "structure.json")
    return {"tempo": tempo, "beat": beat, "key": key, "structure": structure}
