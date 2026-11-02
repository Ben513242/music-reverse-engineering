from mrep.analysis.quantize import (
    QuantizeSettings,
    quantize_duration,
    quantize_notes,
    quantize_position,
)
from mrep.domain.events import NoteEvent
from mrep.io.midi import TempoMap


def test_grid_quantisation_straight():
    s = QuantizeSettings(grid="1/16", triplets=False)
    assert quantize_position(0.503, s) == (0.5, "1/16")
    assert quantize_position(1.13, s) == (1.25, "1/16")
    assert quantize_duration(0.52, s) == 0.5
    assert quantize_duration(0.01, s) == 0.25  # never shorter than one grid unit


def test_triplets_are_used_when_closer():
    s = QuantizeSettings(grid="1/8", triplets=True)
    pos, grid = quantize_position(0.34, s)
    assert grid == "triplet" and abs(pos - 1 / 3) < 1e-6
    s2 = QuantizeSettings(grid="1/8", triplets=False)
    assert quantize_position(0.34, s2) == (0.5, "1/8")


def test_swing_delays_offbeat_eighths():
    s = QuantizeSettings(grid="1/8", triplets=False, swing=1.0)
    pos, grid = quantize_position(0.66, s)
    assert grid == "swing_8" and abs(pos - (0.5 + 1 / 6)) < 1e-6


def test_raw_events_are_preserved_and_referenced():
    raw = [NoteEvent(midi_note=60, onset=0.503, offset=0.998, velocity=90, confidence=0.9, confidence_method="m"), NoteEvent(midi_note=64, onset=0.998, offset=1.496, velocity=70)]
    tm = TempoMap(120.0)
    score = quantize_notes(raw, tm, QuantizeSettings(grid="1/8", triplets=False, beats_per_bar=4))
    assert raw[0].onset == 0.503  # untouched
    assert score[0].raw_event_id == raw[0].event_id
    assert score[0].beat_onset == 1.0 and score[0].beat_duration == 1.0
    assert abs(score[0].timing_offset_beats - (0.503 * 2 - 1.0)) < 1e-6
    assert score[0].measure == 1 and score[0].beat_in_measure == 1.0
    assert score[0].dynamic == "f" and score[1].dynamic == "mf"


def test_tempo_map_follows_detected_beats():
    beats = [0.25, 0.75, 1.3, 1.8]  # slightly irregular
    tm = TempoMap(120.0, beats)
    assert abs(tm.beats_at(0.75) - 1.0) < 1e-9
    assert abs(tm.beats_at(1.025) - 1.5) < 1e-6  # midway between beats 1 (0.75) and 2 (1.3)
    assert abs(tm.seconds_at(2.0) - 1.3) < 1e-9
    assert tm.beats_at(0.0) < 0  # before the first beat extrapolates
