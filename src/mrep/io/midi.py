"""Standard MIDI file writing (mido) with a beat-grid tempo map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo


class TempoMap:
    """Maps seconds <-> beats using detected beat times (piecewise linear) or a constant bpm."""

    def __init__(self, bpm: float, beat_times: list[float] | None = None, offset: float = 0.0):
        self.bpm = float(bpm) if bpm and bpm > 0 else 120.0
        self.offset = float(offset)
        bt = np.asarray(beat_times or [], dtype=np.float64)
        bt = bt[np.concatenate([[True], np.diff(bt) > 1e-3])] if bt.size else bt
        self.beat_times = bt if bt.size >= 2 else None

    def beats_at(self, t: float) -> float:
        if self.beat_times is None:
            return (t - self.offset) * self.bpm / 60.0
        bt = self.beat_times
        if t <= bt[0]:
            return (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]:
            return (len(bt) - 1) + (t - bt[-1]) / (bt[-1] - bt[-2])
        return float(np.interp(t, bt, np.arange(len(bt))))

    def seconds_at(self, beat: float) -> float:
        if self.beat_times is None:
            return self.offset + beat * 60.0 / self.bpm
        bt = self.beat_times
        if beat <= 0:
            return bt[0] + beat * (bt[1] - bt[0])
        if beat >= len(bt) - 1:
            return bt[-1] + (beat - (len(bt) - 1)) * (bt[-1] - bt[-2])
        return float(np.interp(beat, np.arange(len(bt)), bt))

    def tempo_events(self, min_change_bpm: float = 0.5) -> list[tuple[float, float]]:
        """(beat, bpm) pairs describing the tempo curve, thinned to meaningful changes."""
        if self.beat_times is None:
            return [(0.0, self.bpm)]
        intervals = np.diff(self.beat_times)
        bpms = 60.0 / np.maximum(intervals, 1e-3)
        events: list[tuple[float, float]] = []
        last = None
        for i, b in enumerate(bpms):
            if last is None or abs(b - last) >= min_change_bpm:
                events.append((float(i), float(b)))
                last = b
        if not events:
            events.append((0.0, self.bpm))
        # make sure the map starts at beat 0 (the first beat might be after t=0)
        if events[0][0] > 0:
            events.insert(0, (0.0, events[0][1]))
        return events


@dataclass
class MidiNote:
    beat_on: float
    beat_off: float
    note: int
    velocity: int
    bends: list[tuple[float, float]] = field(default_factory=list)  # (absolute beat, semitones)


@dataclass
class MidiTrackSpec:
    name: str
    notes: list[MidiNote] = field(default_factory=list)
    channel: int = 0
    program: int | None = 0
    is_drum: bool = False


def build_midi(
    tracks: list[MidiTrackSpec],
    tempo_map: TempoMap,
    ticks_per_beat: int = 480,
    time_signature: tuple[int, int] = (4, 4),
    key_signature: str | None = None,
    title: str = "MREP",
    pitch_bend_range: float = 2.0,
) -> MidiFile:
    mf = MidiFile(ticks_per_beat=ticks_per_beat)
    conductor = MidiTrack()
    mf.tracks.append(conductor)
    conductor.append(MetaMessage("track_name", name=title, time=0))
    conductor.append(MetaMessage("time_signature", numerator=time_signature[0], denominator=time_signature[1], time=0))
    if key_signature:
        try:
            conductor.append(MetaMessage("key_signature", key=key_signature, time=0))
        except ValueError:
            pass
    last_tick = 0
    for beat, bpm in tempo_map.tempo_events():
        tick = int(round(max(beat, 0.0) * ticks_per_beat))
        conductor.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=tick - last_tick))
        last_tick = tick
    conductor.append(MetaMessage("end_of_track", time=0))

    for spec in tracks:
        track = MidiTrack()
        mf.tracks.append(track)
        channel = 9 if spec.is_drum else spec.channel
        track.append(MetaMessage("track_name", name=spec.name, time=0))
        if not spec.is_drum and spec.program is not None:
            track.append(Message("program_change", program=int(spec.program) % 128, channel=channel, time=0))
        # (tick, order, kind, a, b); order: note_off=0 < bend reset=1 < bend=2 < note_on=3
        events: list[tuple[int, int, str, int, int]] = []
        for n in spec.notes:
            on = int(round(max(n.beat_on, 0.0) * ticks_per_beat))
            off = int(round(max(n.beat_off, 0.0) * ticks_per_beat))
            if off <= on:
                off = on + max(1, ticks_per_beat // 8)
            events.append((on, 3, "on", int(n.note), int(max(1, min(127, n.velocity)))))
            events.append((off, 0, "off", int(n.note), 0))
            if n.bends and not spec.is_drum:
                for beat, semis in n.bends:
                    tick = int(round(max(beat, 0.0) * ticks_per_beat))
                    if on <= tick <= off:
                        value = int(round(max(-1.0, min(1.0, semis / pitch_bend_range)) * 8191))
                        events.append((tick, 2, "bend", value, 0))
                events.append((off, 1, "bend", 0, 0))
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for tick, _order, kind, a, b in events:
            delta = tick - last
            last = tick
            if kind == "on":
                track.append(Message("note_on", note=a, velocity=b, channel=channel, time=delta))
            elif kind == "off":
                track.append(Message("note_off", note=a, velocity=0, channel=channel, time=delta))
            else:
                track.append(Message("pitchwheel", pitch=a, channel=channel, time=delta))
        track.append(MetaMessage("end_of_track", time=0))
    return mf


def write_midi(path, mf: MidiFile) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mf.save(str(path))


def read_midi_notes(path, include_bends: bool = False) -> list[dict]:
    """Return notes as dicts with absolute seconds (for tests / verification)."""
    mf = MidiFile(str(path))
    notes: list[dict] = []
    open_notes: dict[tuple[int, int], tuple[float, int]] = {}
    t = 0.0
    for msg in mf:  # mido converts delta ticks to seconds honouring tempo changes
        t += msg.time
        if include_bends and msg.type == "pitchwheel":
            notes.append({"pitchwheel": msg.pitch, "channel": msg.channel, "time": t})
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in open_notes:
                start, vel = open_notes.pop(key)
                notes.append({"note": msg.note, "channel": msg.channel, "onset": start, "offset": t, "velocity": vel})
    return notes
