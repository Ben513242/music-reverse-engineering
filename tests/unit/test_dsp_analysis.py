import numpy as np

from mrep.analysis.arpeggio import detect_arpeggios
from mrep.analysis.chords import detect_chords
from mrep.domain.events import NoteEvent
from mrep.metrics import drum_metrics, note_metrics, si_sdr, tempo_error
from mrep.utils import synthetic as syn
from mrep.utils.dsp import (
    chroma_from_stft,
    estimate_key,
    estimate_tempo,
    fft_freqs,
    onset_strength,
    stft,
    track_beats,
)
from mrep.utils.music import (
    chord_symbol,
    hz_to_midi,
    key_fifths,
    midi_to_hz,
    midi_to_name,
    name_to_midi,
    spell_pitch,
)


def test_pitch_helpers():
    assert midi_to_name(66) == "F#4" and name_to_midi("F#4") == 66 and name_to_midi("Bb2") == 46
    assert abs(midi_to_hz(69) - 440.0) < 1e-9 and abs(hz_to_midi(440.0) - 69) < 1e-9
    assert spell_pitch(61, flats=False) == ("C", 1, 4) and spell_pitch(61, flats=True) == ("D", -1, 4)
    assert key_fifths(2, "major") == 2 and key_fifths(9, "minor") == 0 and key_fifths(5, "major") == -1
    assert chord_symbol("D", "min7", "F") == "Dm7/F" and chord_symbol("D", "major") == "D"


def test_tempo_and_beats_on_click_track():
    sr = 22050
    bpm = 100.0
    dur = 12.0
    x = np.zeros(int(dur * sr), dtype=np.float32)
    k = syn.kick(sr)
    for i in range(int(dur / (60 / bpm))):
        t = int(i * 60 / bpm * sr)
        x[t : t + len(k)] += k[: max(0, len(x) - t)]
    S = np.abs(stft(x, 2048, 512))
    env = onset_strength(S, freqs=fft_freqs(sr, 2048), n_bands=48)
    est, conf = estimate_tempo(env, sr / 512, 60, 200)
    assert tempo_error(bpm, est)["within_4_percent"]
    assert conf > 0.3
    beats = track_beats(env, sr / 512, est) * 512 / sr
    expected = np.arange(0, dur, 60 / bpm)
    hits = sum(1 for e in expected[1:] if np.min(np.abs(beats - e)) < 0.07)
    assert hits / (len(expected) - 1) > 0.8


def test_key_detection_on_synthetic_chords():
    sr = 22050
    x = np.concatenate([syn.chord(m, 1.0, sr) for m in ([62, 66, 69], [67, 71, 74], [69, 73, 76], [62, 66, 69])])  # D G A D
    S = np.sqrt(np.abs(stft(x, 2048, 512)))
    tonic, mode, conf, _ = estimate_key(chroma_from_stft(S, fft_freqs(sr, 2048)).mean(axis=1))
    assert (tonic, mode) == (2, "major") and conf > 0.2


def _n(m, on, off, tr="arp"):
    return NoteEvent(midi_note=m, onset=on, offset=off, velocity=90, confidence=0.9, confidence_method="t", track=tr)


def test_arpeggio_pattern_detection():
    pattern = [62, 69, 66, 69] * 2  # D A F# A -> 1-5-3-5
    notes = [_n(m, i * 0.25, i * 0.25 + 0.2) for i, m in enumerate(pattern)]
    segs = detect_arpeggios(notes, beat_seconds=0.5)
    assert len(segs) == 1
    s = segs[0]
    assert s["chord_root"] == 2 and s["chord_quality"] == "major" and s["pattern"] == "1-5-3-5" and s["rhythm"] == "1/8"


def test_chord_detection_from_notes_and_bass():
    notes = [_n(m, 0.0, 2.0, "piano") for m in (62, 66, 69)] + [_n(m, 2.0, 4.0, "piano") for m in (67, 71, 74)]
    bass = [_n(42, 0.0, 2.0, "bass"), _n(47, 2.0, 4.0, "bass")]  # F#2 under D (first inversion), B2 under G
    chords = detect_chords([(0, 1), (1, 2), (2, 3), (3, 4)], notes + bass, bass, None, ["major", "minor", "dom7"], 0.3, None)
    assert [c.symbol for c in chords] == ["D/F#", "G/B"]
    assert chords[0].inversion == 1 and chords[0].confidence > 0.5


def test_metrics():
    ref = [{"midi_note": 60, "onset": 1.0, "offset": 1.5}, {"midi_note": 64, "onset": 2.0, "offset": 2.5}]
    est = [{"midi_note": 60, "onset": 1.02, "offset": 1.4}, {"midi_note": 65, "onset": 2.0, "offset": 2.5}, {"midi_note": 70, "onset": 5.0, "offset": 5.5}]
    m = note_metrics(ref, est)
    assert m["onset_pitch"]["hits"] == 1 and m["onset_only"]["hits"] == 2 and 0 < m["onset_pitch"]["f1"] < 1
    d = drum_metrics([{"time": 1.0, "class": "kick"}, {"time": 1.5, "class": "snare"}], [{"time": 1.01, "class": "kick"}, {"time": 1.5, "class": "clap"}])
    assert d["confusion"]["snare->clap"] == 1 and d["per_class"]["kick"]["f1"] == 1.0
    x = np.random.default_rng(0).standard_normal(4000)
    assert si_sdr(x, x * 0.5) > 50 and si_sdr(x, x + 0.5 * np.random.default_rng(1).standard_normal(4000)) < 10
    assert tempo_error(120, 60)["octave_equivalent"] and tempo_error(120, 60)["abs_error_bpm"] == 0
