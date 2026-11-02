from mrep.analysis.fusion_rules import (
    combine_confidence,
    drum_leak_filter,
    fuse_with_fullmix,
    merge_drum_runs,
    merge_note_runs,
)
from mrep.domain.events import DrumEvent, NoteEvent


def n(midi, onset, offset=None, conf=0.8, track="piano"):
    return NoteEvent(midi_note=midi, onset=onset, offset=offset or onset + 0.5, confidence=conf, confidence_method="m", track=track)


def test_combine_confidence_noisy_or():
    assert combine_confidence([0.88, 0.79])[0] > 0.88
    assert combine_confidence([None, None]) == (None, "unavailable")
    assert combine_confidence([0.5, None])[0] == 0.5


def test_stem_and_fullmix_agreement_boosts_confidence_and_averages_onset():
    stem = [n(66, 23.51, conf=0.88)]
    full = [n(66, 23.50, conf=0.79, track="fullmix"), n(40, 10.0, conf=0.9, track="fullmix")]
    fused, extras = fuse_with_fullmix(stem, full, onset_tol=0.05, fullmix_only_weight=0.5)
    assert len(fused) == 1 and fused[0].confidence > 0.88
    assert 23.50 < fused[0].onset < 23.51
    assert fused[0].confidence_method == "stem_fullmix_agreement"
    assert len(extras) == 1 and extras[0].track == "unassigned" and abs(extras[0].confidence - 0.45) < 1e-6


def test_consensus_between_adapters_scales_unmatched_notes():
    a = [n(60, 0.0, conf=0.6), n(62, 1.0, conf=0.6)]
    b = [n(60, 0.02, conf=0.7)]
    merged = merge_note_runs([a, b], onset_tol=0.05)
    assert len(merged) == 2
    by_pitch = {m.midi_note: m for m in merged}
    assert by_pitch[60].confidence > 0.7 and len(by_pitch[60].sources) == 2
    assert abs(by_pitch[62].confidence - 0.48) < 1e-6


def test_drum_component_evidence_relabels_but_does_not_add():
    broad = [DrumEvent(time=1.0, drum_class="snare", confidence=0.5, confidence_method="m", class_scores={"snare": 0.5, "kick": 0.4}), DrumEvent(time=2.0, drum_class="closed_hat", confidence=0.9, confidence_method="m", class_scores={"hat": 0.9})]
    comp = [DrumEvent(time=1.01, drum_class="kick", confidence=0.6, confidence_method="hint"), DrumEvent(time=3.0, drum_class="kick", confidence=0.6, confidence_method="hint")]
    merged = merge_drum_runs(broad, comp, tol=0.03, add_unmatched=False)
    assert [e.drum_class for e in merged] == ["kick", "closed_hat"]
    merged2 = merge_drum_runs(broad, comp, tol=0.03, add_unmatched=True)
    assert len(merged2) == 3


def test_drum_leak_filter_drops_short_notes_on_kicks():
    notes = [n(28, 1.0, 1.08, track="bass"), n(36, 1.0, 1.8, track="bass"), n(28, 1.5, 1.58, track="bass")]
    drums = [DrumEvent(time=1.01, drum_class="kick")]
    kept, dropped = drum_leak_filter(notes, drums)
    assert dropped == 1 and len(kept) == 2
