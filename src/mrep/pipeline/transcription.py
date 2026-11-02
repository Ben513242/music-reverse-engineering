"""Stages 7-8 - drum event detection, pitched AMT (per-stem path A and optional full-mix
path B, optional multi-adapter ensemble) and FX event detection.

All outputs here are RAW events (seconds). They are written under ``03_events/raw/`` and
are never overwritten by later quantisation.
"""

from __future__ import annotations

from pathlib import Path

from mrep.adapters.base import (
    AdapterUnavailable,
    DrumTranscriptionAdapter,
    FXAdapter,
    TranscriptionAdapter,
)
from mrep.domain.labels import DRUM_CLASSES, PITCHED_FAMILIES
from mrep.domain.tracks import Stem
from mrep.io.audio import write_audio
from mrep.io.json import write_events, write_json
from mrep.pipeline.classification import load_labels
from mrep.pipeline.normalization import load_normalized
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.pipeline.separation import load_stem_audio, load_stems

RAW_DIR = "raw"
COMPONENT_CLASS = {"kick": "kick", "snare": "snare", "hats": None, "clap": "clap", "percussion": None, "closed_hat": "closed_hat", "open_hat": "open_hat"}


def _is_drum_stem(stem: Stem) -> bool:
    return stem.source_hint == "drums" or stem.name.startswith("drums")


def _pitched_candidates(stems: list[Stem], labels: dict) -> list[Stem]:
    broad_with_children = {s.parent for s in stems if s.parent}
    out = []
    for s in stems:
        if _is_drum_stem(s):
            continue
        if s.name in broad_with_children:  # refined into components; transcribe the components
            continue
        label = labels.get(s.name)
        fam = label.source_family if label else s.source_hint
        if fam in PITCHED_FAMILIES or s.source_hint in ("bass", "other", "piano", "guitar", "vocals", "vocal"):
            if label is not None and label.is_fx:
                continue
            out.append(s)
    return out


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("transcription")
    cfg = ctx.stage_cfg("transcription")
    result = StageResult()
    stems = load_stems(ctx)
    labels = load_labels(ctx)
    raw_dir = ctx.project.events_dir / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[dict]] = {"drums": [], "notes": [], "fx": []}

    # ---- drums ------------------------------------------------------------------------
    drum_stems = [s for s in stems if _is_drum_stem(s)]
    if drum_stems:
        try:
            dadapter: DrumTranscriptionAdapter = ctx.adapter("drum_transcription", cfg.get("drums_adapter", "onset_bands"), cfg)  # type: ignore[assignment]
            result.adapters.append(dadapter.info())
            for stem in drum_stems:
                component = stem.name.split("/")[-1] if "/" in stem.name else None
                hint = COMPONENT_CLASS.get(component) if component else None
                if component and component not in COMPONENT_CLASS and component in DRUM_CLASSES:
                    hint = component
                try:
                    audio, sr = load_stem_audio(ctx, stem)
                    prov = ctx.provenance("transcription", dadapter, f"stem:{stem.name}")
                    events = dadapter.transcribe_drums(audio, sr, track=stem.name, class_hint=hint, provenance=prov)
                    path = raw_dir / f"drums__{stem.file_stem}.jsonl"
                    write_events(path, events)
                    index["drums"].append({"stem": stem.name, "adapter": dadapter.name, "path": ctx.rel(path), "count": len(events), "class_hint": hint, "level": stem.level, "stem_confidence_method": stem.confidence_method})
                    log.info("drums %s: %d events", stem.name, len(events))
                except Exception as exc:  # noqa: BLE001
                    result.add_error(f"drums:{stem.name}", exc)
        except AdapterUnavailable as exc:
            result.add_error("drums:adapter", exc)

    # ---- pitched AMT: path A (per stem) --------------------------------------------------
    adapter_names = list(cfg.get("pitched_adapters") or ["spectral_peaks"])
    pitched_adapters: list[TranscriptionAdapter] = []
    for name in adapter_names:
        try:
            ad = ctx.adapter("transcription", name, cfg)  # type: ignore[assignment]
            pitched_adapters.append(ad)
            result.adapters.append(ad.info())
        except AdapterUnavailable as exc:
            result.add_error(f"amt:{name}", exc)
    for stem in _pitched_candidates(stems, labels):
        label = labels.get(stem.name)
        instrument = label.display_name if label else stem.source_hint
        for ad in pitched_adapters:
            try:
                audio, sr = load_stem_audio(ctx, stem)
                prov = ctx.provenance("transcription", ad, f"stem:{stem.name}")
                notes = ad.transcribe(audio, sr, track=stem.name, instrument=instrument, provenance=prov)
                path = raw_dir / f"notes__{stem.file_stem}__{ad.name}.jsonl"
                write_events(path, notes)
                index["notes"].append({"stem": stem.name, "adapter": ad.name, "path": ctx.rel(path), "count": len(notes), "source": "stem", "instrument": instrument})
                log.info("AMT %s/%s: %d notes", stem.name, ad.name, len(notes))
            except Exception as exc:  # noqa: BLE001
                result.add_error(f"amt:{stem.name}:{ad.name}", exc)

    # ---- pitched AMT: path B (full mix) ---------------------------------------------------
    if cfg.get("full_mix_amt") and pitched_adapters:
        ad = pitched_adapters[0]
        try:
            data, sr = load_normalized(ctx)
            prov = ctx.provenance("transcription", ad, "normalized_audio")
            notes = ad.transcribe(data, sr, track="fullmix", instrument="mix", provenance=prov)
            path = raw_dir / f"notes__fullmix__{ad.name}.jsonl"
            write_events(path, notes)
            index["notes"].append({"stem": "fullmix", "adapter": ad.name, "path": ctx.rel(path), "count": len(notes), "source": "fullmix", "instrument": "mix"})
            log.info("AMT full mix: %d notes", len(notes))
        except Exception as exc:  # noqa: BLE001
            result.add_error("amt:fullmix", exc)

    # ---- FX ---------------------------------------------------------------------------
    fxcfg = cfg.get("fx", {}) or {}
    if fxcfg.get("enabled", True):
        try:
            fxad: FXAdapter = ctx.adapter("fx", fxcfg.get("adapter", "heuristic"), fxcfg)  # type: ignore[assignment]
            result.adapters.append(fxad.info())
            data, sr = load_normalized(ctx)
            prov = ctx.provenance("transcription", fxad, "normalized_audio")
            fx_events = fxad.detect(data, sr, provenance=prov)
            fx_dir = ctx.project.fx_dir
            counters: dict[str, int] = {}
            for ev in fx_events:
                counters[ev.fx_class] = counters.get(ev.fx_class, 0) + 1
                a, b = int(ev.start * sr), int(min(len(data), ev.end * sr))
                clip_path = fx_dir / f"{ev.fx_class}_{counters[ev.fx_class]:03d}.wav"
                write_audio(clip_path, data[a:b], sr)
                ev.clip_path = ctx.rel(clip_path)
            path = raw_dir / "fx.jsonl"
            write_events(path, fx_events)
            index["fx"].append({"adapter": fxad.name, "path": ctx.rel(path), "count": len(fx_events)})
            log.info("FX: %d events", len(fx_events))
        except Exception as exc:  # noqa: BLE001
            result.add_error("fx", exc)

    write_json(raw_dir / "index.json", index)
    result.artifacts["raw_index"] = ctx.rel(raw_dir / "index.json")
    for group in index.values():
        for entry in group:
            result.artifacts[f"raw:{Path(entry['path']).stem}"] = entry["path"]
    result.summary = {"drum_runs": len(index["drums"]), "note_runs": len(index["notes"]), "fx_events": sum(e["count"] for e in index["fx"]), "errors": [e["where"] for e in result.errors]}
    return result
