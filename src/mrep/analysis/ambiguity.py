"""Low-confidence handling: flag ambiguous results and suggest what to try next."""

from __future__ import annotations

from typing import Any

import numpy as np

from mrep.domain.events import ChordEvent, DrumEvent, FXEvent, NoteEvent
from mrep.domain.labels import SoundLabel

DEFAULT_THRESHOLDS = {"label": 0.5, "chord": 0.5, "key": 0.3, "fx": 0.5, "notes": 0.3, "drums": 0.5}


def build_ambiguity_report(labels: dict[str, SoundLabel], chords: list[ChordEvent], key_events: list[dict[str, Any]], fx: list[FXEvent], notes: list[NoteEvent], drums: list[DrumEvent], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    items: list[dict[str, Any]] = []
    for stem, label in labels.items():
        c = label.confidence
        if c is None or c < thr["label"]:
            label.ambiguous = True
            cands = [f"{k.label} ({k.probability})" for k in label.source_candidates[:3]]
            items.append({"kind": "label", "id": stem, "confidence": c, "threshold": thr["label"], "value": label.source_family, "candidates": cands, "suggestion": "re-run with a learned classification adapter, add the source to stages.separation.force_specialists, or compare several adapters"})
        else:
            label.ambiguous = False
    for ch in chords:
        if ch.confidence is None or ch.confidence < thr["chord"]:
            items.append({"kind": "chord", "id": ch.event_id, "start": ch.start, "end": ch.end, "confidence": ch.confidence, "threshold": thr["chord"], "value": ch.symbol, "suggestion": "ambiguous chord window; check the alternatives in chords.json or widen the analysis window"})
    for k in key_events:
        if k.get("confidence") is None or k["confidence"] < thr["key"]:
            items.append({"kind": "key", "id": f"key@{k['start']}", "confidence": k.get("confidence"), "threshold": thr["key"], "value": f"{k['tonic']} {k['mode']}", "suggestion": "key is uncertain (modal / relative-key ambiguity); see key.json scores for the runner-up"})
    for e in fx:
        if e.confidence is None or e.confidence < thr["fx"]:
            items.append({"kind": "fx", "id": e.event_id, "start": e.start, "end": e.end, "confidence": e.confidence, "threshold": thr["fx"], "value": e.fx_class, "suggestion": "listen to the clip in 07_fx before trusting the class"})
    by_track: dict[str, list[float]] = {}
    for n in notes:
        if n.confidence is not None:
            by_track.setdefault(n.track, []).append(n.confidence)
    for track, confs in by_track.items():
        mean = float(np.mean(confs))
        if mean < thr["notes"]:
            items.append({"kind": "notes", "id": track, "confidence": round(mean, 4), "threshold": thr["notes"], "value": f"{len(confs)} notes", "suggestion": "low mean note confidence; enable full_mix_amt or add a second AMT adapter for consensus"})
    dconf = [d.confidence for d in drums if d.confidence is not None]
    if dconf and float(np.mean(dconf)) < thr["drums"]:
        items.append({"kind": "drums", "id": "drums", "confidence": round(float(np.mean(dconf)), 4), "threshold": thr["drums"], "value": f"{len(dconf)} events", "suggestion": "drum class confidence is low; try a drum transcription model adapter"})
    return {"thresholds": thr, "count": len(items), "items": items}
