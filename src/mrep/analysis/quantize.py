"""Score quantisation (grid, triplets, swing). Raw events are never modified: this produces
separate ``ScoreNoteEvent`` objects that reference the raw event id."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mrep.domain.events import DrumEvent, NoteEvent, ScoreNoteEvent
from mrep.io.midi import TempoMap
from mrep.utils.music import velocity_to_dynamic

GRID_BEATS = {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125}


@dataclass
class QuantizeSettings:
    grid: str = "1/16"
    triplets: bool = True
    swing: float = 0.0  # 0 = straight, 1 = full triplet swing on off-beat eighths
    beats_per_bar: int = 4

    @property
    def grid_beats(self) -> float:
        return GRID_BEATS.get(self.grid, 0.25)

    @property
    def triplet_beats(self) -> float:
        # triplet subdivision at the same level: 1/8 grid -> triplet 8ths (1/3 beat)
        return self.grid_beats * 4.0 / 3.0 / 2.0 if self.grid != "1/4" else 1.0 / 3.0


def quantize_position(pos: float, settings: QuantizeSettings) -> tuple[float, str]:
    """Snap a beat position to the nearest straight (with swing) or triplet grid point."""
    g = settings.grid_beats
    q_straight = round(pos / g) * g
    if settings.swing and g <= 0.5:
        # off-beat eighths are delayed: 0.5 -> 0.5 + swing/6 (swing=1 gives 2/3)
        frac = q_straight % 1.0
        if abs(frac - 0.5) < 1e-9:
            swung = math.floor(q_straight) + 0.5 + settings.swing / 6.0
            if abs(pos - swung) < abs(pos - q_straight):
                return round(swung, 6), "swing_8"
    best, name = q_straight, settings.grid
    if settings.triplets:
        tg = settings.triplet_beats
        q_trip = round(pos / tg) * tg
        if abs(pos - q_trip) < abs(pos - q_straight) - 1e-9:
            best, name = q_trip, "triplet"
    return round(best, 6), name


def quantize_duration(dur: float, settings: QuantizeSettings, triplet: bool = False) -> float:
    unit = settings.triplet_beats if triplet else settings.grid_beats
    return max(unit, round(dur / unit) * unit)


def quantize_notes(notes: list[NoteEvent], tempo_map: TempoMap, settings: QuantizeSettings) -> list[ScoreNoteEvent]:
    out: list[ScoreNoteEvent] = []
    bpb = settings.beats_per_bar
    for n in notes:
        raw_on = tempo_map.beats_at(n.onset)
        raw_off = tempo_map.beats_at(n.offset)
        q_on, grid = quantize_position(max(raw_on, 0.0), settings)
        q_dur = quantize_duration(max(raw_off - raw_on, 0.0), settings, triplet=(grid == "triplet"))
        measure = int(math.floor(q_on / bpb)) + 1
        out.append(
            ScoreNoteEvent(
                raw_event_id=n.event_id,
                midi_note=n.midi_note,
                beat_onset=q_on,
                beat_duration=round(q_dur, 6),
                measure=measure,
                beat_in_measure=round(q_on - (measure - 1) * bpb, 6),
                grid=grid,
                velocity=n.velocity,
                dynamic=velocity_to_dynamic(n.velocity),
                track=n.track,
                instrument=n.instrument,
                timing_offset_beats=round(raw_on - q_on, 6),
                confidence=n.confidence,
                confidence_method=n.confidence_method,
                provenance=n.provenance,
            )
        )
    out.sort(key=lambda s: (s.beat_onset, s.midi_note))
    return out


def quantize_drums(events: list[DrumEvent], tempo_map: TempoMap, settings: QuantizeSettings) -> list[dict[str, Any]]:
    out = []
    bpb = settings.beats_per_bar
    for e in events:
        raw = tempo_map.beats_at(e.time)
        q, grid = quantize_position(max(raw, 0.0), settings)
        measure = int(math.floor(q / bpb)) + 1
        out.append({"raw_event_id": e.event_id, "class": e.drum_class, "beat_onset": q, "beat_duration": settings.grid_beats, "measure": measure, "beat_in_measure": round(q - (measure - 1) * bpb, 6), "grid": grid, "velocity": e.velocity, "dynamic": velocity_to_dynamic(e.velocity), "timing_offset_beats": round(raw - q, 6), "confidence": e.confidence, "confidence_method": e.confidence_method, "track": e.track})
    out.sort(key=lambda d: (d["beat_onset"], d["class"]))
    return out
