import pytest

from mrep.domain.events import DrumEvent, FXEvent, NoteEvent, ScoreNoteEvent, event_from_dict
from mrep.domain.labels import DRUM_CLASSES, FX_CLASSES, SOURCE_SUBTYPES, Candidate, SoundLabel
from mrep.domain.provenance import Provenance


def test_multilabel_sound_label_roundtrip():
    prov = Provenance(stage="classification", adapter="x", model="m", model_version="1", run_id="r", input="stem:bass")
    label = SoundLabel(source_family="synthesizer", source_subtype="synth_bass", musical_role=["bassline", "pluck"], frequency_role=["sub_bass"], sound_design=["short_decay", "analog_style"], confidence=0.8, confidence_method="model", provenance=prov, source_candidates=[Candidate("synth_bass", 0.8), Candidate("808", 0.2)])
    assert label.validate() == []
    d = label.to_dict()
    assert d["musical_role"] == ["bassline", "pluck"]
    back = SoundLabel.from_dict(d)
    assert back == label
    assert back.is_pitched and not back.is_fx


def test_label_validation_reports_problems():
    label = SoundLabel(source_family="synthesizer", source_subtype="kick", musical_role=["nonsense"], frequency_role=["mid"], sound_design=["bogus"])
    problems = label.validate()
    assert any("subtype" in p for p in problems)
    assert any("musical_role" in p for p in problems)
    assert any("sound_design" in p for p in problems)


def test_taxonomies_cover_spec_minimums():
    for cls in ("kick", "snare", "closed_hat", "open_hat", "clap", "rimshot", "crash", "ride", "shaker", "tambourine", "cowbell", "conga"):
        assert cls in DRUM_CLASSES
    for fx in ("riser", "downlifter", "white_noise", "impact", "glitch", "granular_texture"):
        assert fx in FX_CLASSES
    assert {"synth_lead", "synth_pad", "synth_arp", "synth_pluck", "synth_drone"} <= SOURCE_SUBTYPES["synthesizer"]
    assert {"electric_bass", "synth_bass", "808"} <= SOURCE_SUBTYPES["bass"]


def test_confidence_must_be_unit_interval_or_none():
    with pytest.raises(ValueError):
        NoteEvent(midi_note=60, onset=0.0, offset=1.0, confidence=1.5)
    n = NoteEvent(midi_note=66, onset=12.5, offset=13.125, velocity=81)
    assert n.confidence is None and n.confidence_method == "unavailable"
    assert n.pitch == "F#4"


def test_event_serialisation_roundtrip():
    prov = Provenance(stage="transcription", adapter="amt_x", model="ckpt", model_version="abc", run_id="run_001", input="stem_piano_01", parameters_hash="p")
    n = NoteEvent(midi_note=66, onset=12.5, offset=13.125, velocity=81, confidence=0.92, confidence_method="model", provenance=prov, track="piano")
    d = n.to_dict()
    assert d["type"] == "note" and d["provenance"]["adapter"] == "amt_x"
    back = event_from_dict(d)
    assert isinstance(back, NoteEvent) and back == n
    drum = DrumEvent(time=12.482, drum_class="snare", velocity=96, confidence=0.94, confidence_method="model")
    dd = drum.to_dict()
    assert dd["class"] == "snare" and "drum_class" not in dd
    assert event_from_dict(dd) == drum
    fx = FXEvent(fx_class="riser", start=28.1, end=31.95, confidence=0.86, confidence_method="model")
    assert event_from_dict(fx.to_dict()).fx_class == "riser"
    s = ScoreNoteEvent(raw_event_id=n.event_id, midi_note=66, beat_onset=1.5, beat_duration=0.5, measure=1, beat_in_measure=1.5)
    assert event_from_dict(s.to_dict()).raw_event_id == n.event_id


def test_velocity_is_clamped():
    assert NoteEvent(midi_note=60, onset=0, offset=1, velocity=300).velocity == 127
    assert DrumEvent(time=0, drum_class="kick", velocity=0).velocity == 1
