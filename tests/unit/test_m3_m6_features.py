"""Unit tests for time signature, sound analysis, humanisation, pitch bend, ambiguity and ensembles."""

from __future__ import annotations

import numpy as np

from mrep.adapters import create_adapter
from mrep.analysis.ambiguity import build_ambiguity_report
from mrep.analysis.fusion_rules import merge_drum_adapter_runs
from mrep.analysis.humanization import classify_articulations, humanization_stats
from mrep.analysis.pitch_bend import estimate_note_bends, f0_track
from mrep.analysis.quantize import QuantizeSettings, quantize_notes
from mrep.analysis.time_signature import estimate_time_signature
from mrep.domain.events import ChordEvent, DrumEvent, NoteEvent
from mrep.domain.labels import Candidate, SoundLabel
from mrep.domain.provenance import Provenance
from mrep.io.midi import MidiNote, MidiTrackSpec, TempoMap, build_midi, read_midi_notes, write_midi
from mrep.io.musicxml import (
    DIVISIONS,
    ScoreSpec,
    XmlNote,
    XmlPart,
    read_musicxml_summary,
    write_musicxml,
)
from mrep.utils import synthetic as syn
from mrep.utils.music import midi_to_hz

PROV = Provenance(stage="test", adapter="x")


def test_time_signature_three_vs_four():
    rng = np.random.default_rng(0)
    waltz = np.tile([1.0, 0.3, 0.35], 16) + rng.normal(0, 0.03, 48)
    L, phase, conf, method, scores = estimate_time_signature(waltz)
    assert (L, phase) == (3, 0) and conf is not None and conf > 0.05 and method == "accent_periodicity"
    common = np.tile([1.0, 0.3, 0.6, 0.3], 12) + rng.normal(0, 0.03, 48)
    L4, phase4, conf4, _, _ = estimate_time_signature(np.roll(common, 1))
    assert L4 == 4 and phase4 == 1 and conf4 > 0.1
    assert estimate_time_signature(np.array([1.0, 0.2]))[3] == "assumed"


def test_f0_track_and_pitch_bend_vibrato():
    sr = 22050
    dur = 1.0
    t = np.arange(int(dur * sr)) / sr
    f0 = midi_to_hz(57) * 2 ** ((0.5 * np.sin(2 * np.pi * 5.5 * t)) / 12)  # A3 with +-50 cent vibrato at 5.5 Hz
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = (0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase)).astype(np.float32)
    times, est = f0_track(x, sr)
    voiced = ~np.isnan(est)
    assert voiced.mean() > 0.8
    assert abs(np.nanmedian(est) - midi_to_hz(57)) / midi_to_hz(57) < 0.03
    note = NoteEvent(midi_note=57, onset=0.0, offset=dur, track="lead")
    n_bent = estimate_note_bends([note], times, est)
    assert n_bent == 1 and note.pitch_bend and max(abs(b[1]) for b in note.pitch_bend) > 0.25
    assert "vibrato" in note.features and 4.5 < note.features["vibrato"]["rate_hz"] < 6.5
    assert note.features["vibrato"]["depth_cents"] > 20
    # a steady tone has no bend
    steady = (0.6 * np.sin(2 * np.pi * midi_to_hz(57) * t)).astype(np.float32)
    times2, est2 = f0_track(steady, sr)
    n2 = NoteEvent(midi_note=57, onset=0.0, offset=dur)
    assert estimate_note_bends([n2], times2, est2) == 0 and n2.pitch_bend is None


def test_sound_analysis_descriptors_pluck_vs_pad():
    sr = 22050
    pluck = np.concatenate([syn.tone(72, 0.25, sr, amp=0.5, attack=0.003, decay=0.08, sustain=0.1, release=0.03) for _ in range(8)])
    pad = syn.chord([60, 64, 67], 3.0, sr, amp=0.4, attack=0.3, decay=0.3, sustain=0.9, release=0.6)
    sa = create_adapter("sound_analysis", "dsp")
    rp = sa.analyze_sound(pluck, sr, stem_name="x", label=None, provenance=PROV)
    rd = sa.analyze_sound(pad, sr, stem_name="y", label=None, provenance=PROV)
    assert "short_attack" in rp.sound_design and rp.features["attack_ms"] < 20
    assert "slow_attack" in rd.sound_design and rd.features["attack_ms"] > 80
    assert rp.features["harmonicity"] > 0.5 and rd.features["noise_ratio"] < 0.5
    assert rp.patch_candidates and rp.confidence is not None and rp.confidence_method == "descriptor_jaccard"
    assert all(0 <= c.probability <= 1 for c in rp.patch_candidates)
    label = SoundLabel(source_family="synthesizer", sound_design=rp.sound_design, patch_candidates=rp.patch_candidates)
    assert label.validate() == []
    back = SoundLabel.from_dict(label.to_dict())
    assert back.patch_candidates[0].label == rp.patch_candidates[0].label and back.ambiguous is False
    noise = syn.white_noise(sr, 2.0, 0.3)
    rn = sa.analyze_sound(noise, sr, stem_name="n", label=None, provenance=PROV)
    assert "noise_based" in rn.sound_design


def _raw(midi, on, off, track="lead", vel=80):
    return NoteEvent(midi_note=midi, onset=on, offset=off, velocity=vel, track=track, confidence=0.9, confidence_method="t")


def test_articulation_and_humanization_stats():
    raw = [_raw(60, 0.0, 0.2), _raw(62, 0.5, 1.0), _raw(64, 1.0, 1.48), _raw(65, 1.5, 1.9), _raw(67, 2.0, 2.4)]
    tm = TempoMap(120.0)
    score = quantize_notes(raw, tm, QuantizeSettings(grid="1/16", triplets=False))
    raw_by_id = {n.event_id: n for n in raw}
    classify_articulations(score, raw_by_id, 0.5)
    arts = {s.midi_note: s.articulation for s in score}
    assert arts[60] == "staccato" and arts[62] == "legato" and arts[64] == "legato" and arts[65] == "normal"
    stats = humanization_stats(score, raw_by_id, 0.5)["lead"]
    assert stats["note_count"] == 5 and stats["legato_ratio"] == 0.4 and stats["staccato_ratio"] == 0.2
    assert stats["velocity_std"] == 0.0 and stats["timing_offset_std_ms"] >= 0


def test_musicxml_staccato_and_slur_marks(tmp_path):
    q = DIVISIONS
    part = XmlPart(part_id="P1", name="Lead", notes=[XmlNote(0, q // 2, 60, articulation="staccato"), XmlNote(q, q, 62, articulation="legato", slur="start"), XmlNote(2 * q, q, 64, articulation="legato"), XmlNote(3 * q, q, 65, slur="stop")])
    path = write_musicxml(tmp_path / "a.musicxml", ScoreSpec(parts=[part]))
    summary = read_musicxml_summary(path)
    assert summary["staccatos"] == 1 and summary["slurs"] == 2


def test_midi_pitch_bend_messages(tmp_path):
    tm = TempoMap(120.0)
    spec = MidiTrackSpec(name="Bass", notes=[MidiNote(0.0, 1.0, 40, 90, bends=[(0.0, 0.0), (0.5, 1.0)])], program=33)
    path = tmp_path / "b.mid"
    write_midi(path, build_midi([spec], tm, pitch_bend_range=2.0))
    msgs = read_midi_notes(path, include_bends=True)
    wheels = [m for m in msgs if "pitchwheel" in m]
    assert len(wheels) == 3 and max(w["pitchwheel"] for w in wheels) == 4096  # +1 semitone of a +-2 range
    assert wheels[-1]["pitchwheel"] == 0  # reset at note off


def test_drum_adapter_consensus():
    a = [DrumEvent(time=1.0, drum_class="kick", confidence=0.6, confidence_method="m"), DrumEvent(time=2.0, drum_class="snare", confidence=0.6, confidence_method="m")]
    b = [DrumEvent(time=1.01, drum_class="kick", confidence=0.7, confidence_method="m"), DrumEvent(time=3.0, drum_class="closed_hat", confidence=0.5, confidence_method="m")]
    merged = merge_drum_adapter_runs([a, b], tol=0.03)
    by_class = {e.drum_class: e for e in merged}
    assert len(merged) == 3 and by_class["kick"].confidence > 0.7 and abs(by_class["snare"].confidence - 0.48) < 1e-6


def test_ambiguity_report_flags_low_confidence():
    labels = {"other": SoundLabel(source_family="harmonic", confidence=0.3, confidence_method="h", source_candidates=[Candidate("strings", 0.3), Candidate("piano", 0.28)]), "bass": SoundLabel(source_family="bass", confidence=0.95, confidence_method="h")}
    chords = [ChordEvent(start=0, end=1, root="C", quality="major", symbol="C", confidence=0.9, confidence_method="t"), ChordEvent(start=1, end=2, root="F", quality="add9", symbol="Fadd9", confidence=0.35, confidence_method="t")]
    keys = [{"start": 0.0, "tonic": "C", "mode": "major", "confidence": 0.2}]
    report = build_ambiguity_report(labels, chords, keys, [], [_raw(60, 0, 1, "other"), _raw(62, 1, 2, "other")], [])
    kinds = sorted(i["kind"] for i in report["items"])
    assert kinds == ["chord", "key", "label"]
    assert labels["other"].ambiguous is True and labels["bass"].ambiguous is False
    assert all("suggestion" in i for i in report["items"])
