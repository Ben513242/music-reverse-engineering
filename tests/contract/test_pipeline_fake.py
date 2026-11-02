"""Contract tests: the pipeline with fake adapters (no DSP models, deterministic)."""

from __future__ import annotations

import json
from pathlib import Path

from mrep.domain.project import Project
from mrep.io.json import read_json, read_jsonl
from mrep.manifest import (
    PROJECT_COMPLETE,
    PROJECT_PARTIAL,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    Manifest,
)
from mrep.pipeline.runner import STAGE_ORDER, build_runner, downstream_of, upstream_closure


def _run(demo_wav: Path, cfg: dict, out: Path, **kw):
    project = Project.for_audio(demo_wav, out)
    runner = build_runner(project, cfg, demo_wav)
    return project, runner.run(**kw)


def test_dependency_helpers():
    assert downstream_of("transcription") == ["fusion", "scoring", "reconstruction"]
    assert upstream_closure(["scoring"]) == ["ingest", "normalization", "global_analysis", "separation", "classification", "transcription", "fusion", "scoring"]
    assert STAGE_ORDER[0] == "ingest" and STAGE_ORDER[-1] == "reconstruction"


def test_full_pipeline_with_fake_adapters(demo_wav, fake_config, tmp_path):
    project, man = _run(demo_wav, fake_config, tmp_path / "out")
    assert man.status == PROJECT_COMPLETE, man.to_dict()["stages"]
    assert all(man.stages[s].status == STATUS_COMPLETE for s in STAGE_ORDER)
    root = project.root
    for rel in (
        "manifest.json",
        "01_normalized/audio.wav",
        "01_normalized/analysis_mono.wav",
        "02_stems/drums.wav",
        "02_stems/bass.wav",
        "02_stems/other.wav",
        "02_stems/drums/kick.wav",
        "02_stems/stems.json",
        "03_events/notes.jsonl",
        "03_events/drums.jsonl",
        "03_events/fx.jsonl",
        "03_events/labels.json",
        "03_events/raw/index.json",
        "04_midi/FULL_SONG.mid",
        "04_midi/Drums.mid",
        "04_midi/quantized/FULL_SONG_quantized.mid",
        "05_score/Full_Score.musicxml",
        "05_score/Drums.musicxml",
        "05_score/score_events.jsonl",
        "05_score/humanization.json",
        "06_analysis/sound_design.json",
        "06_analysis/ambiguities.json",
        "06_analysis/tempo.json",
        "06_analysis/beat.json",
        "06_analysis/key.json",
        "06_analysis/chords.json",
        "06_analysis/structure.json",
        "06_analysis/dynamics.json",
        "06_analysis/instruments.json",
        "07_fx/riser_001.wav",
        "07_fx/trigger_map.json",
        "08_reconstruction/reconstructed.wav",
        "08_reconstruction/reconstruction.json",
        "08_reconstruction/daw/session.json",
        "08_reconstruction/daw/demo_song.rpp",
    ):
        assert (root / rel).exists(), rel
    # manifest content
    m = read_json(root / "manifest.json")
    assert m["project_id"].startswith("mrep_") and len(m["source"]["sha256"]) == 64
    assert m["config"]["stages"]["separation"]["broad_adapter"] == "fake"
    assert m["stages"]["separation"]["adapters"][1]["name"] == "fake"
    assert m["stages"]["transcription"]["params_hash"]
    # provenance + confidence on every fused event
    for line in read_jsonl(root / "03_events/notes.jsonl"):
        assert line["provenance"]["adapter"] and line["provenance"]["run_id"]
        assert line["confidence"] is None or 0 <= line["confidence"] <= 1
    drums = read_jsonl(root / "03_events/drums.jsonl")
    assert {d["class"] for d in drums} >= {"kick", "snare", "closed_hat"}
    # raw and score events are separate; raw untouched by quantisation
    raw = read_jsonl(root / "03_events/raw/notes__bass__fake.jsonl")
    score = read_jsonl(root / "05_score/score_events.jsonl")
    assert raw[0]["onset"] == 0.0 and score[0]["raw_event_id"]
    assert all(s["articulation"] in ("normal", "staccato", "legato") for s in score)
    human = read_json(root / "05_score/humanization.json")["tracks"]
    assert "bass" in human and human["bass"]["note_count"] > 0
    sd = read_json(root / "06_analysis/sound_design.json")
    assert "bass" in sd and sd["bass"]["patch_candidates"][0]["label"] == "digital_pluck" and "drums" not in sd
    amb = read_json(root / "06_analysis/ambiguities.json")
    assert "items" in amb and amb["thresholds"]["label"] == 0.5
    session = read_json(root / "08_reconstruction/daw/session.json")
    assert session["format"] == "mrep-session/1" and len(session["audio_tracks"]) >= 4 and session["midi_files"]
    assert (root / "08_reconstruction/daw" / session["audio_tracks"][0]["file"]).resolve().exists()
    rpp = (root / "08_reconstruction/daw/demo_song.rpp").read_text()
    assert rpp.count("<TRACK") == len(session["audio_tracks"]) and "MARKER" in rpp
    assert read_json(root / "06_analysis/tempo.json")["time_signature_method"] in ("fake_constant", "assumed", "accent_periodicity", "configured")
    # FX kept as clips, mapped to trigger notes, never notated as pitched notes
    trig = read_json(root / "07_fx/trigger_map.json")
    assert trig["triggers"]["24"]["class"] == "riser"
    recon = read_json(root / "08_reconstruction/reconstruction.json")
    assert any(v == "original_audio_clip" for v in recon["sources"].values())
    assert "midi_rendered" in recon["sources"].values()
    # labels are multi-label
    labels = read_json(root / "03_events/labels.json")
    assert "bassline" in labels["bass"]["musical_role"] and labels["bass"]["frequency_role"] == ["sub_bass", "low_bass"]
    assert "arpeggio" in labels["bass"]["musical_role"]  # the fake AMT emits an arpeggio -> fusion adds the role


def test_resume_skips_completed_stages_and_force_reruns_downstream(demo_wav, fake_config, tmp_path):
    out = tmp_path / "out"
    project, man = _run(demo_wav, fake_config, out)
    first_run = man.stages["transcription"].run_id
    # resume: everything cached
    _, man2 = _run(demo_wav, fake_config, out, resume=True)
    assert all(man2.stages[s].cache_hit for s in STAGE_ORDER)
    assert man2.stages["transcription"].run_id == first_run
    assert man2.status == PROJECT_COMPLETE
    # force transcription: it and downstream re-run, upstream cached
    _, man3 = _run(demo_wav, fake_config, out, resume=True, force=["transcription"])
    for s in ("ingest", "normalization", "global_analysis", "separation", "classification"):
        assert man3.stages[s].cache_hit, s
    for s in ("transcription", "fusion", "scoring", "reconstruction"):
        assert not man3.stages[s].cache_hit and man3.stages[s].run_id != first_run, s
    # changing a parameter invalidates the cache of that stage (+ downstream)
    cfg = json.loads(json.dumps(fake_config))
    cfg["stages"]["scoring"]["quantize"]["grid"] = "1/8"
    _, man4 = _run(demo_wav, cfg, out, resume=True)
    assert man4.stages["fusion"].cache_hit and not man4.stages["scoring"].cache_hit and not man4.stages["reconstruction"].cache_hit
    # partial targets: `mrep analyze` style
    _, man5 = _run(demo_wav, cfg, out, targets=["global_analysis"], resume=True)
    assert man5.stages["global_analysis"].cache_hit


def test_specialist_failure_gives_partial_success(demo_wav, fake_config, tmp_path):
    fake_config["stages"]["separation"]["specialists"] = {"drums": "fake_components", "piano": "failing"}
    fake_config["stages"]["separation"]["force_specialists"] = ["piano"]
    project, man = _run(demo_wav, fake_config, tmp_path / "out")
    sep = man.stages["separation"]
    assert sep.status == STATUS_PARTIAL
    assert any("failing" in e["where"] for e in sep.errors)
    assert man.status == PROJECT_PARTIAL
    # everything else still ran and produced outputs
    for s in ("classification", "transcription", "fusion", "scoring", "reconstruction"):
        assert man.stages[s].status == STATUS_COMPLETE, s
    assert (project.root / "02_stems/drums/kick.wav").exists()
    assert (project.root / "04_midi/FULL_SONG.mid").exists()


def test_transcription_adapter_failure_is_isolated(demo_wav, fake_config, tmp_path):
    fake_config["stages"]["transcription"]["pitched_adapters"] = ["fake", "failing"]
    project, man = _run(demo_wav, fake_config, tmp_path / "out")
    tr = man.stages["transcription"]
    assert tr.status == STATUS_PARTIAL and any("failing" in e["where"] for e in tr.errors)
    assert man.stages["fusion"].status == STATUS_COMPLETE
    assert (project.root / "03_events/notes.jsonl").stat().st_size > 0
    assert man.status == PROJECT_PARTIAL


def test_required_stage_failure_fails_project(fake_config, tmp_path):
    bad = tmp_path / "nope.wav"
    bad.write_bytes(b"not audio at all")
    project = Project.for_audio(bad, tmp_path / "out")
    man = build_runner(project, fake_config, bad).run()
    assert man.stages["ingest"].status == STATUS_FAILED
    assert man.status == "FAILED"
    assert "normalization" not in man.stages or man.stages["normalization"].status != STATUS_COMPLETE


def test_disabled_stage_is_skipped_and_downstream_skipped(demo_wav, fake_config, tmp_path):
    fake_config["stages"]["reconstruction"]["enabled"] = False
    fake_config["stages"]["scoring"]["musicxml"]["enabled"] = False
    project, man = _run(demo_wav, fake_config, tmp_path / "out")
    assert man.stages["reconstruction"].status == STATUS_SKIPPED
    assert man.status == PROJECT_COMPLETE
    assert not (project.root / "05_score/Full_Score.musicxml").exists()
    assert (project.root / "04_midi/FULL_SONG.mid").exists()


def test_manifest_reload_matches_runner_state(demo_wav, fake_config, tmp_path):
    project, man = _run(demo_wav, fake_config, tmp_path / "out")
    loaded = Manifest.load(project.manifest_path)
    assert loaded.project_id == man.project_id
    assert set(loaded.artifacts) == set(man.artifacts)
    assert loaded.runs[-1]["status"] == PROJECT_COMPLETE
