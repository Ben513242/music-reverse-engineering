"""Stage 9 - event fusion + musical analysis on events (chords, arpeggios).

Reads RAW events from ``03_events/raw`` and writes the fused ``notes.jsonl`` /
``drums.jsonl`` / ``fx.jsonl`` plus ``06_analysis/chords.json`` and ``arpeggios.json``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mrep.analysis.ambiguity import build_ambiguity_report
from mrep.analysis.arpeggio import detect_arpeggios
from mrep.analysis.chords import detect_chords
from mrep.analysis.fusion_rules import (
    drum_leak_filter,
    fuse_with_fullmix,
    merge_drum_adapter_runs,
    merge_drum_runs,
    merge_note_runs,
)
from mrep.domain.events import DrumEvent, FXEvent, NoteEvent
from mrep.domain.provenance import Provenance
from mrep.io.json import read_events, read_json, write_events, write_json
from mrep.pipeline.classification import LABELS_FILE, load_labels
from mrep.pipeline.global_analysis import load_global
from mrep.pipeline.normalization import load_normalized
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.pipeline.separation import load_stem_audio, load_stems
from mrep.utils.dsp import chroma_from_stft, fft_freqs, stft, to_mono


def _audio_chroma_fn(ctx: PipelineContext):
    """Chroma of the harmonic material (other + bass stems if present, else the mix)."""
    stems = {s.name: s for s in load_stems(ctx)}
    parts = []
    sr = None
    for name in ("other", "bass", "piano", "guitar", "vocals"):
        if name in stems:
            audio, sr = load_stem_audio(ctx, stems[name])
            parts.append(to_mono(audio))
    if not parts:
        data, sr = load_normalized(ctx)
        parts.append(to_mono(data))
    x = np.sum(parts, axis=0)
    n_fft, hop = 4096, 1024
    S = np.sqrt(np.abs(stft(x, n_fft, hop)))
    chroma = chroma_from_stft(S, fft_freqs(sr, n_fft), fmin=50.0)
    fps = sr / hop

    def fn(start: float, end: float):
        a, b = int(start * fps), max(int(end * fps), int(start * fps) + 1)
        if a >= chroma.shape[1]:
            return None
        return chroma[:, a:b].mean(axis=1)

    return fn


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("fusion")
    cfg = ctx.stage_cfg("fusion")
    result = StageResult()
    tol = float(cfg.get("onset_tolerance_seconds", 0.05))
    ptol = int(cfg.get("pitch_tolerance_semitones", 0))
    min_conf = float(cfg.get("min_confidence", 0.2))
    raw_dir = ctx.project.events_dir / "raw"
    index = read_json(raw_dir / "index.json")
    labels = load_labels(ctx)
    prov = Provenance(stage="fusion", adapter="fusion_rules", model="rule_based", model_version="1.0", run_id=ctx.run_id, input="03_events/raw")

    # ---- drums ----------------------------------------------------------------------
    broad_runs: dict[str, list[list[DrumEvent]]] = defaultdict(list)  # stem -> runs (one per adapter)
    comp_runs: dict[str, list[list[DrumEvent]]] = defaultdict(list)
    evidence_only = False
    for entry in index.get("drums", []):
        evs = [e for e in read_events(ctx.project.abs(entry["path"])) if isinstance(e, DrumEvent)]
        if entry.get("level") == "specialist" and entry.get("class_hint"):
            comp_runs[entry["stem"]].append(evs)
            if entry.get("stem_confidence_method") == "band_filter_heuristic":
                evidence_only = True
        elif entry.get("level") == "specialist":
            continue  # unhinted band stems (hats/percussion) are too ambiguous to add events
        else:
            broad_runs[entry["stem"]].append(evs)
    broad: list[DrumEvent] = []
    for runs in broad_runs.values():
        broad.extend(merge_drum_adapter_runs(runs, tol))
    components: list[DrumEvent] = []
    for runs in comp_runs.values():
        components.extend(merge_drum_adapter_runs(runs, tol))
    drums = merge_drum_runs(broad, components, tol, add_unmatched=not evidence_only)
    for d in drums:
        d.track = "drums"
    write_events(ctx.project.events_dir / "drums.jsonl", drums)

    # ---- notes ----------------------------------------------------------------------
    per_stem: dict[str, list[list[NoteEvent]]] = defaultdict(list)
    fullmix: list[NoteEvent] = []
    for entry in index.get("notes", []):
        notes = [n for n in read_events(ctx.project.abs(entry["path"])) if isinstance(n, NoteEvent)]
        if entry.get("source") == "fullmix":
            fullmix.extend(notes)
        else:
            per_stem[entry["stem"]].append(notes)
    fused: list[NoteEvent] = []
    stats = {}
    for stem, runs in per_stem.items():
        merged = merge_note_runs(runs, tol, ptol)
        stats[stem] = {"adapters": len(runs), "raw": sum(len(r) for r in runs), "merged": len(merged)}
        fused.extend(merged)
    fused, extras = fuse_with_fullmix(fused, fullmix, tol, ptol, float(cfg.get("fullmix_only_weight", 0.5)))
    before = len(fused) + len(extras)
    fused = [n for n in fused + extras if n.confidence is None or n.confidence >= min_conf]
    leak_cfg = cfg.get("drum_leak_filter", {}) or {}
    dropped_leak = 0
    if leak_cfg.get("enabled", True):
        fused, dropped_leak = drum_leak_filter(fused, drums, float(leak_cfg.get("max_note_seconds", 0.15)), float(leak_cfg.get("tolerance_seconds", 0.03)))
    fused.sort(key=lambda n: (n.onset, n.midi_note))
    for n in fused:
        n.provenance = n.provenance or prov
    write_events(ctx.project.events_dir / "notes.jsonl", fused)
    stats["fullmix"] = {"raw": len(fullmix), "unassigned_kept": sum(1 for n in fused if n.track == "unassigned"), "dropped_low_confidence": before - len(fused) - dropped_leak, "dropped_drum_leak": dropped_leak}
    log.info("notes: %d fused (%s)", len(fused), stats)

    # ---- fx ---------------------------------------------------------------------------
    fx: list[FXEvent] = []
    for entry in index.get("fx", []):
        fx.extend(e for e in read_events(ctx.project.abs(entry["path"])) if isinstance(e, FXEvent))
    write_events(ctx.project.events_dir / "fx.jsonl", fx)

    # ---- chords ---------------------------------------------------------------------
    chords = []
    ccfg = cfg.get("chords", {}) or {}
    if ccfg.get("enabled", True):
        try:
            glob = load_global(ctx)
            beats = [b["time"] for b in glob["beat"]["beats"]]
            duration = float(glob["tempo"].get("duration") or ctx.manifest.source.get("duration") or (beats[-1] + 1 if beats else 1.0))
            if len(beats) >= 2:
                windows = list(zip(beats[:-1], beats[1:]))
            else:
                step = 0.5
                windows = [(t, t + step) for t in np.arange(0.0, duration, step)]
            bass_notes = [n for n in fused if labels.get(n.track) is not None and labels[n.track].source_family == "bass"]
            chord_notes = [n for n in fused if n.track != "unassigned"]
            chords = detect_chords(windows, chord_notes, bass_notes, _audio_chroma_fn(ctx), list(ccfg.get("qualities", ["major", "minor"])), float(ccfg.get("min_confidence", 0.3)), prov)
            write_json(ctx.project.analysis_dir / "chords.json", {"chords": [c.to_dict() for c in chords]})
            result.artifacts["chords"] = ctx.rel(ctx.project.analysis_dir / "chords.json")
            log.info("chords: %d", len(chords))
        except Exception as exc:  # noqa: BLE001
            result.add_error("chords", exc)

    # ---- arpeggios ----------------------------------------------------------------
    arps = []
    acfg = cfg.get("arpeggio", {}) or {}
    if acfg.get("enabled", True):
        try:
            glob = load_global(ctx)
            bpm = float(glob["tempo"].get("tempo") or 120.0)
            by_track: dict[str, list[NoteEvent]] = defaultdict(list)
            for n in fused:
                by_track[n.track].append(n)
            for track, notes in by_track.items():
                segs = detect_arpeggios(notes, 60.0 / bpm, int(acfg.get("min_notes", 4)), float(acfg.get("ioi_cv_max", 0.3)))
                arps.extend(segs)
                covered = sum(s["note_count"] for s in segs)
                label = labels.get(track)
                if label is not None and notes and covered / len(notes) >= 0.5 and "arpeggio" not in label.musical_role:
                    label.musical_role.append("arpeggio")
            write_json(ctx.project.analysis_dir / "arpeggios.json", {"arpeggios": arps})
            result.artifacts["arpeggios"] = ctx.rel(ctx.project.analysis_dir / "arpeggios.json")
        except Exception as exc:  # noqa: BLE001
            result.add_error("arpeggio", exc)

    # ---- low-confidence handling --------------------------------------------------------
    ambiguity: dict = {"count": 0, "items": []}
    try:
        key_events = []
        try:
            key_events = load_global(ctx)["key"].get("key_events", [])
        except Exception:  # noqa: BLE001
            pass
        ambiguity = build_ambiguity_report(labels, chords, key_events, fx, fused, drums, cfg.get("ambiguity") or None)
        write_json(ctx.project.analysis_dir / "ambiguities.json", ambiguity)
        result.artifacts["ambiguities"] = ctx.rel(ctx.project.analysis_dir / "ambiguities.json")
        if ambiguity["count"]:
            log.info("%d ambiguous results flagged (see 06_analysis/ambiguities.json)", ambiguity["count"])
    except Exception as exc:  # noqa: BLE001
        result.add_error("ambiguity", exc)
    write_json(ctx.project.events_dir / LABELS_FILE, {k: v.to_dict() for k, v in labels.items()})

    write_json(ctx.project.events_dir / "fusion.json", {"notes": stats, "drums": {"broad": len(broad), "components": len(components), "fused": len(drums)}, "fx": len(fx), "chords": len(chords), "arpeggios": len(arps), "settings": cfg})
    result.artifacts.update({"notes": ctx.rel(ctx.project.events_dir / "notes.jsonl"), "drums": ctx.rel(ctx.project.events_dir / "drums.jsonl"), "fx": ctx.rel(ctx.project.events_dir / "fx.jsonl"), "labels": ctx.rel(ctx.project.events_dir / LABELS_FILE), "fusion_stats": ctx.rel(ctx.project.events_dir / "fusion.json")})
    result.summary = {"notes": len(fused), "drums": len(drums), "fx": len(fx), "chords": len(chords), "arpeggios": len(arps), "ambiguous": ambiguity.get("count", 0), "notes_with_pitch_bend": sum(1 for n in fused if n.pitch_bend)}
    result.confidence = {"notes_mean": round(float(np.mean([n.confidence for n in fused if n.confidence is not None])), 4) if any(n.confidence is not None for n in fused) else None, "drums_mean": round(float(np.mean([d.confidence for d in drums if d.confidence is not None])), 4) if any(d.confidence is not None for d in drums) else None}
    return result


def load_fused(ctx: PipelineContext) -> tuple[list[NoteEvent], list[DrumEvent], list[FXEvent]]:
    ev = ctx.project.events_dir
    notes = [n for n in read_events(ev / "notes.jsonl") if isinstance(n, NoteEvent)] if (ev / "notes.jsonl").exists() else []
    drums = [d for d in read_events(ev / "drums.jsonl") if isinstance(d, DrumEvent)] if (ev / "drums.jsonl").exists() else []
    fx = [f for f in read_events(ev / "fx.jsonl") if isinstance(f, FXEvent)] if (ev / "fx.jsonl").exists() else []
    return notes, drums, fx
