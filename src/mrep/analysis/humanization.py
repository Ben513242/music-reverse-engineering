"""Humanisation metadata: articulation per note and per-track timing / velocity statistics.

Everything here is derived from RAW event timing so that re-performance can keep the feel
(micro timing, velocity variation, legato / staccato) instead of a grid-aligned, velocity-100 MIDI.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mrep.domain.events import NoteEvent, ScoreNoteEvent


def classify_articulations(score_notes: list[ScoreNoteEvent], raw_by_id: dict[str, NoteEvent], beat_seconds: float, legato_gap_s: float = 0.02, staccato_ratio: float = 0.5) -> None:
    """Set ``articulation`` and ``duration_ratio`` on each score note (in place)."""
    by_track: dict[str, list[ScoreNoteEvent]] = defaultdict(list)
    for s in score_notes:
        by_track[s.track].append(s)
    for notes in by_track.values():
        notes.sort(key=lambda s: (raw_by_id[s.raw_event_id].onset if s.raw_event_id in raw_by_id else s.beat_onset, s.midi_note))
        onsets = [raw_by_id[s.raw_event_id].onset if s.raw_event_id in raw_by_id else s.beat_onset * beat_seconds for s in notes]
        for i, s in enumerate(notes):
            raw = raw_by_id.get(s.raw_event_id)
            onset = onsets[i]
            offset = raw.offset if raw else onset + s.beat_duration * beat_seconds
            nxt = next((onsets[j] for j in range(i + 1, len(notes)) if onsets[j] > onset + 1e-3), None)
            if nxt is None:
                s.duration_ratio = None
                s.articulation = "normal"
                continue
            ioi = nxt - onset
            s.duration_ratio = round((offset - onset) / ioi, 4) if ioi > 0 else None
            if offset >= nxt - legato_gap_s:
                s.articulation = "legato"
            elif s.duration_ratio is not None and s.duration_ratio < staccato_ratio and (offset - onset) < 0.75 * beat_seconds:
                s.articulation = "staccato"
            else:
                s.articulation = "normal"


def humanization_stats(score_notes: list[ScoreNoteEvent], raw_by_id: dict[str, NoteEvent], beat_seconds: float) -> dict[str, dict[str, Any]]:
    by_track: dict[str, list[ScoreNoteEvent]] = defaultdict(list)
    for s in score_notes:
        by_track[s.track].append(s)
    out: dict[str, dict[str, Any]] = {}
    for track, notes in by_track.items():
        offsets = np.array([s.timing_offset_beats for s in notes])
        vel = np.array([s.velocity for s in notes])
        arts = [s.articulation for s in notes]
        ratios = [s.duration_ratio for s in notes if s.duration_ratio is not None]
        vib = [raw_by_id[s.raw_event_id].features.get("vibrato") for s in notes if s.raw_event_id in raw_by_id and raw_by_id[s.raw_event_id].features.get("vibrato")]
        bends = sum(1 for s in notes if s.raw_event_id in raw_by_id and raw_by_id[s.raw_event_id].pitch_bend)
        out[track] = {
            "note_count": len(notes),
            "timing_offset_mean_beats": round(float(offsets.mean()), 5),
            "timing_offset_std_beats": round(float(offsets.std()), 5),
            "timing_offset_std_ms": round(float(offsets.std() * beat_seconds * 1000), 2),
            "velocity_mean": round(float(vel.mean()), 2),
            "velocity_std": round(float(vel.std()), 2),
            "velocity_min": int(vel.min()),
            "velocity_max": int(vel.max()),
            "legato_ratio": round(arts.count("legato") / len(arts), 4),
            "staccato_ratio": round(arts.count("staccato") / len(arts), 4),
            "mean_duration_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
            "notes_with_pitch_bend": bends,
            "notes_with_vibrato": len(vib),
            "vibrato_rate_hz_mean": round(float(np.mean([v["rate_hz"] for v in vib])), 2) if vib else None,
            "vibrato_depth_cents_mean": round(float(np.mean([v["depth_cents"] for v in vib])), 1) if vib else None,
        }
    return out
