from __future__ import annotations

import json
from pathlib import Path

import yaml

from mrep.cli import main


def _write_fake_profile(tmp_path: Path, fake_config: dict) -> Path:
    p = tmp_path / "fake_profile.yaml"
    p.write_text(yaml.safe_dump(fake_config))
    return p


def test_cli_run_info_and_staged_commands(demo_wav, fake_config, tmp_path, capsys):
    profile = _write_fake_profile(tmp_path, fake_config)
    out = tmp_path / "out"
    rc = main(["analyze", str(demo_wav), "--profile", str(profile), "-o", str(out)])
    assert rc == 0
    project = out / "demo_song"
    assert (project / "06_analysis/tempo.json").exists() and not (project / "02_stems/stems.json").exists()
    rc = main(["separate", str(demo_wav), "--profile", str(profile), "-o", str(out), "--resume"])
    assert rc == 0 and (project / "02_stems/stems.json").exists()
    rc = main(["transcribe", str(project)])
    assert rc == 0 and (project / "03_events/notes.jsonl").exists()
    rc = main(["score", str(project)])
    assert rc == 0 and (project / "04_midi/FULL_SONG.mid").exists()
    rc = main(["reconstruct", str(project)])
    assert rc == 0 and (project / "08_reconstruction/reconstructed.wav").exists()
    capsys.readouterr()
    rc = main(["info", str(project), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "COMPLETE" and all(data["stages"][s]["status"] == "complete" for s in data["stages"])


def test_cli_run_with_set_override_and_force(demo_wav, fake_config, tmp_path, capsys):
    profile = _write_fake_profile(tmp_path, fake_config)
    out = tmp_path / "out"
    rc = main(["run", str(demo_wav), "--profile", str(profile), "-o", str(out), "--set", "stages.scoring.quantize.grid=1/8", "--device", "cpu"])
    assert rc == 0
    man = json.loads((out / "demo_song/manifest.json").read_text())
    assert man["config"]["stages"]["scoring"]["quantize"]["grid"] == "1/8" and man["device"] == "cpu"
    rc = main(["run", str(demo_wav), "--profile", str(profile), "-o", str(out), "--resume", "--force-stage", "scoring"])
    assert rc == 0
    man = json.loads((out / "demo_song/manifest.json").read_text())
    assert man["stages"]["fusion"]["cache_hit"] and not man["stages"]["scoring"]["cache_hit"]


def test_cli_partial_exit_code(demo_wav, fake_config, tmp_path):
    fake_config["stages"]["transcription"]["pitched_adapters"] = ["fake", "failing"]
    profile = _write_fake_profile(tmp_path, fake_config)
    rc = main(["run", str(demo_wav), "--profile", str(profile), "-o", str(tmp_path / "out")])
    assert rc == 2


def test_cli_synth_adapters_profiles(tmp_path, capsys):
    out = tmp_path / "s.wav"
    assert main(["synth", str(out), "--bars", "2", "--sr", "22050"]) == 0
    assert out.exists() and out.with_suffix(".wav.truth.json").exists()
    assert main(["adapters", "--kind", "separation"]) == 0
    assert "hpss" in capsys.readouterr().out
    assert main(["profiles"]) == 0
    assert "max_quality" in capsys.readouterr().out
    assert main(["models", "download", "separation", "hpss"]) == 0
