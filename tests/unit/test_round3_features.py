"""Unit tests: Viterbi key segments, 6/8 metre, tempo marks, humanised rendering, DAW export, evaluation."""

from __future__ import annotations

import numpy as np

from mrep.adapters.render.additive import render_note
from mrep.analysis.key_segments import segments_from_path, viterbi_key_path
from mrep.analysis.pitch_bend import f0_track
from mrep.analysis.time_signature import estimate_time_signature, metre_name
from mrep.io.daw import write_reaper_rpp, write_session_json
from mrep.io.musicxml import (
    DIVISIONS,
    ScoreSpec,
    XmlNote,
    XmlPart,
    read_musicxml_summary,
    write_musicxml,
)
from mrep.utils.music import midi_to_hz
from mrep.utils.synthetic import demo_song


def _scores(best: str, second: str | None = None, margin: float = 0.3, noise: float = 0.0, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    keys = [f"{t} {m}" for m in ("major", "minor") for t in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")]
    out = {k: float(rng.normal(0.0, noise)) for k in keys}
    out[best] = 0.8
    if second:
        out[second] = 0.8 - margin
    return out


def test_viterbi_smooths_single_window_flips_but_keeps_real_changes():
    windows = [_scores("D major", "B minor", seed=i) for i in range(4)] + [_scores("B minor", "D major", margin=0.05, seed=10)] + [_scores("D major", "B minor", seed=20)] + [_scores("E major", "C# minor", seed=i) for i in range(30, 35)]
    path = viterbi_key_path(windows, switch_penalty=0.25)
    assert path[:6] == ["D major"] * 6  # the one-window relative-minor flip is smoothed away
    assert path[6:] == ["E major"] * 5  # the real modulation survives
    bounds = [(i * 10.0, (i + 1) * 10.0) for i in range(len(windows))]
    segs = segments_from_path(path, bounds, [0.5] * len(windows))
    assert [(s["key"], s["start"]) for s in segs] == [("D major", 0.0), ("E major", 60.0)]
    assert segs[0]["tonic"] == "D" and segs[0]["mode"] == "major" and segs[0]["confidence"] == 0.5


def test_time_signature_six_eight_and_names():
    rng = np.random.default_rng(1)
    six = np.tile([1.0, 0.3, 0.3, 0.6, 0.3, 0.3], 10) + rng.normal(0, 0.02, 60)
    L, phase, conf, _method, scores = estimate_time_signature(six)
    assert L == 6 and phase == 0 and scores["6/8"] > scores["3/4"]
    waltz = np.tile([1.0, 0.3, 0.35], 20) + rng.normal(0, 0.02, 60)
    assert estimate_time_signature(waltz)[0] == 3
    common = np.tile([1.0, 0.3, 0.6, 0.3], 15) + rng.normal(0, 0.02, 60)
    assert estimate_time_signature(common)[0] == 4
    assert metre_name(6) == "6/8" and metre_name(4) == "4/4" and metre_name(5) == "5/4"


def test_musicxml_tempo_changes(tmp_path):
    q = DIVISIONS
    part = XmlPart(part_id="P1", name="X", notes=[XmlNote(i * q, q, 60 + i % 5) for i in range(12)])
    spec = ScoreSpec(parts=[part], bpm=120, tempo_changes=[(2, 126.0), (3, 132.0)])
    summary = read_musicxml_summary(write_musicxml(tmp_path / "t.musicxml", spec))
    assert summary["tempo_marks"] == 3 and summary["measures"] == 3


def test_render_note_applies_vibrato_bend_and_articulation():
    sr = 22050
    plain = render_note(57, 1.0, sr, harmonics=((1, 1.0), (2, 0.4)), adsr=(0.01, 0.1, 0.8, 0.1), amp=0.5)
    vib = render_note(57, 1.0, sr, harmonics=((1, 1.0), (2, 0.4)), adsr=(0.01, 0.1, 0.8, 0.1), amp=0.5, vibrato={"rate_hz": 5.0, "depth_cents": 60.0})
    _t, f_plain = f0_track(plain, sr)
    _t, f_vib = f0_track(vib, sr)
    dev_plain = np.nanstd(12 * np.log2(f_plain / midi_to_hz(57))) * 100
    dev_vib = np.nanstd(12 * np.log2(f_vib / midi_to_hz(57))) * 100
    assert dev_vib > 15 and dev_plain < 5
    bent = render_note(57, 1.0, sr, harmonics=((1, 1.0),), adsr=(0.01, 0.1, 0.8, 0.1), amp=0.5, pitch_bend=[[0.0, 0.0], [1.0, 2.0]])
    _t, f_b = f0_track(bent, sr)
    assert np.nanmedian(f_b[-40:]) > np.nanmedian(f_b[:40]) * 1.05
    stacc = render_note(60, 1.0, sr, harmonics=((1, 1.0),), adsr=(0.01, 0.1, 0.8, 0.3), amp=0.5, articulation="staccato")
    legato = render_note(60, 1.0, sr, harmonics=((1, 1.0),), adsr=(0.01, 0.1, 0.8, 0.3), amp=0.5, articulation="legato")
    assert len(stacc) < len(plain) < len(legato) + sr  # staccato is shorter, legato slightly longer than nominal


def test_daw_export_files(tmp_path):
    rpp = write_reaper_rpp(tmp_path / "p.rpp", bpm=120.0, time_signature=(4, 4), duration=17.0, audio_tracks=[{"name": "Bass", "file": "../02_stems/bass.wav"}, {"name": "Drums", "file": "../02_stems/drums.wav", "mute": True}], markers=[(0.25, "A"), (8.25, "B")], tempo_changes=[(0.0, 120.0), (8.0, 124.0)])
    text = rpp.read_text()
    assert text.startswith("<REAPER_PROJECT") and text.count("<TRACK") == 2 and "MARKER 2 8.250000" in text
    assert 'FILE "../02_stems/bass.wav"' in text and "TEMPO 120.000 4 4" in text and "<TEMPOENVEX" in text
    session = write_session_json(tmp_path / "session.json", {"format": "mrep-session/1", "audio_tracks": []})
    assert session.exists()


def test_demo_song_stems_sum_to_mix_and_truth_has_beats():
    mix, truth, stems = demo_song(sr=22050, bars=2, return_stems=True)
    total = sum(stems.values())
    assert np.allclose(total, mix, atol=1e-5)
    assert set(stems) == {"drums", "bass", "other", "fx"}
    assert len(truth["beats"]) == 8 and len(truth["downbeats"]) == 2 and abs(truth["beats"][1] - truth["beats"][0] - 0.5) < 1e-9
