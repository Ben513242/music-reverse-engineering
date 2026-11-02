from pathlib import Path

from mrep.config import deep_merge, load_config, parse_set, stage_params_hash
from mrep.manifest import (
    PROJECT_COMPLETE,
    PROJECT_FAILED,
    PROJECT_PARTIAL,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PARTIAL,
    Manifest,
)


def test_profiles_merge_over_default():
    base = load_config("default")
    light = load_config("light")
    assert light["profile"] == "light" and light["device"] == "cpu"
    assert light["stages"]["separation"]["chunk_seconds"] == 30
    assert light["stages"]["scoring"] == base["stages"]["scoring"]  # untouched sections survive
    mx = load_config("max_quality")
    assert mx["stages"]["transcription"]["full_mix_amt"] is True


def test_set_overrides_and_hash_changes():
    cfg = load_config("standard", sets=["stages.transcription.polyphony=7", "stages.fusion.chords.enabled=false"])
    assert cfg["stages"]["transcription"]["polyphony"] == 7
    assert cfg["stages"]["fusion"]["chords"]["enabled"] is False
    assert parse_set("a.b=3") == (["a", "b"], 3)
    h1 = stage_params_hash(load_config("standard"), "transcription")
    h2 = stage_params_hash(cfg, "transcription")
    assert h1 != h2
    assert stage_params_hash(load_config("standard"), "scoring") == stage_params_hash(cfg, "scoring")


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = deep_merge(a, {"x": {"y": 5}})
    assert a["x"]["y"] == 1 and b["x"] == {"y": 5, "z": 2}


def test_manifest_roundtrip_and_status(tmp_path: Path):
    m = Manifest(project_id="p", song_name="s")
    m.stage("ingest").status = STATUS_COMPLETE
    m.stage("separation").status = STATUS_PARTIAL
    m.add_artifact("stem:drums", "02_stems/drums.wav", "separation")
    assert m.compute_status({"ingest"}) == PROJECT_PARTIAL
    path = m.save(tmp_path / "manifest.json")
    back = Manifest.load(path)
    assert back.stages["separation"].status == STATUS_PARTIAL
    assert back.artifacts["stem:drums"].path == "02_stems/drums.wav"
    assert "stem:drums" in back.stages["separation"].artifacts
    back.stage("separation").status = STATUS_COMPLETE
    assert back.compute_status({"ingest"}) == PROJECT_COMPLETE
    back.stage("ingest").status = STATUS_FAILED
    assert back.compute_status({"ingest"}) == PROJECT_FAILED
