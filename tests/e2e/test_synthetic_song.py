"""End-to-end: the real baseline (DSP) adapters on a synthetic song with known ground truth."""

from __future__ import annotations

import json

import pytest

from mrep.cli import main
from mrep.io.json import read_json, read_jsonl
from mrep.io.midi import read_midi_notes
from mrep.io.musicxml import read_musicxml_summary
from mrep.metrics import drum_metrics, note_metrics, tempo_error

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def project(tmp_path_factory, demo_wav_44k):
    out = tmp_path_factory.mktemp("e2e")
    rc = main(["run", str(demo_wav_44k), "--profile", "light", "-o", str(out), "--set", "stages.reconstruction.mp3=false"])
    assert rc == 0
    return out / "demo_song"


def test_tempo_key_and_time_signature(project, demo_audio_44k):
    _, truth, _ = demo_audio_44k
    tempo = read_json(project / "06_analysis/tempo.json")
    assert tempo_error(truth["bpm"], tempo["tempo"])["within_4_percent"]
    assert tempo["time_signature"] == "4/4" and tempo["time_signature_method"] == "accent_periodicity"
    assert tempo["time_signature_confidence"] is not None and tempo["time_signature_confidence"] > 0.05
    key = read_json(project / "06_analysis/key.json")
    assert key["global_key"] == truth["key"]
    beat = read_json(project / "06_analysis/beat.json")
    first_db = beat["downbeats"][0]
    bar = 4 * 60.0 / truth["bpm"]
    assert min((first_db - truth["lead_in"]) % bar, bar - (first_db - truth["lead_in"]) % bar) < 0.1


def test_drums_detected_with_core_classes(project, demo_audio_44k):
    _, truth, _ = demo_audio_44k
    drums = read_jsonl(project / "03_events/drums.jsonl")
    m = drum_metrics([{"time": d["time"], "class": d["class"]} for d in truth["drums"]], [{"time": d["time"], "class": d["class"]} for d in drums], tol=0.05)
    assert m["onset_f1"]["recall"] > 0.6, m["onset_f1"]  # coincident kick+hat count as two reference events
    assert m["per_class"]["kick"]["recall"] > 0.75 and m["per_class"]["snare"]["recall"] > 0.75, m["per_class"]
    hats_ref = [d["time"] for d in truth["drums"] if d["class"] in ("closed_hat", "open_hat")]
    hats_est = [d["time"] for d in drums if d["class"] in ("closed_hat", "open_hat")]
    from mrep.metrics import onset_f1

    assert onset_f1(hats_ref, hats_est, 0.05)["recall"] > 0.3  # open/closed split is heuristic; hat family must be found
    assert all(0 <= d["velocity"] <= 127 for d in drums) and len({d["velocity"] for d in drums}) > 3
    assert all(d["provenance"]["adapter"] == "onset_bands" for d in drums)


def test_pitched_notes_recall(project, demo_audio_44k):
    _, truth, _ = demo_audio_44k
    notes = read_jsonl(project / "03_events/notes.jsonl")
    ref = [n for n in truth["notes"] if n["track"] != "bass"]
    est = [n for n in notes if n["track"] != "unassigned"]
    m = note_metrics(ref, est, onset_tol=0.08)
    assert m["onset_pitch"]["recall"] > 0.6, m
    bass_ref = [n for n in truth["notes"] if n["track"] == "bass"]
    bass_est = [n for n in notes if n["track"] == "bass"]
    assert note_metrics(bass_ref, bass_est, onset_tol=0.1)["onset_pitch"]["recall"] > 0.4


def test_fx_riser_detected_and_kept_out_of_notation(project, demo_audio_44k):
    _, truth, _ = demo_audio_44k
    fx = read_jsonl(project / "03_events/fx.jsonl")
    risers = [e for e in fx if e["class"] == "riser"]
    assert risers, fx
    t = truth["fx"][0]
    assert abs(risers[0]["end"] - t["end"]) < 0.6 and risers[0]["start"] >= t["start"] - 0.3
    assert risers[0]["clip_path"] and (project / risers[0]["clip_path"]).exists()
    trig = read_json(project / "07_fx/trigger_map.json")
    assert any(v["class"] == "riser" for v in trig["triggers"].values())


def test_outputs_midi_musicxml_reconstruction(project):
    notes = read_midi_notes(project / "04_midi/FULL_SONG.mid")
    assert len(notes) > 20 and any(n["channel"] == 9 for n in notes)
    assert len({n["velocity"] for n in notes}) > 5  # dynamics, not fixed velocity
    summary = read_musicxml_summary(project / "05_score/Full_Score.musicxml")
    assert "Drums" in summary["part_names"] and summary["measures"] >= 4 and summary["notes"] > 20
    recon = read_json(project / "08_reconstruction/reconstruction.json")
    assert recon["sources"]["Drums"] == "midi_rendered"
    assert any(v == "original_audio_clip" for v in recon["sources"].values())
    man = json.loads((project / "manifest.json").read_text())
    assert man["status"] == "COMPLETE"
    assert man["stages"]["separation"]["summary"]["stems"][:3] == ["drums", "bass", "other"]
    labels = read_json(project / "03_events/labels.json")
    assert labels["bass"]["source_family"] == "bass" and "bassline" in labels["bass"]["musical_role"]
    assert "sub_bass" in labels["bass"]["frequency_role"]
    assert labels["drums"]["source_family"] == "drums"
    sd = read_json(project / "06_analysis/sound_design.json")
    assert "bass" in sd and "other" in sd and sd["bass"]["features"]["attack_ms"] >= 0
    assert "drums" not in sd
    human = read_json(project / "05_score/humanization.json")["tracks"]
    assert human and all(0 <= h["legato_ratio"] <= 1 for h in human.values())
    amb = read_json(project / "06_analysis/ambiguities.json")
    assert isinstance(amb["items"], list)
