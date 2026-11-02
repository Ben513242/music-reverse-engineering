"""Extra input formats (via ffmpeg), 48 kHz input, and the evaluate command."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from mrep.cli import main
from mrep.io.audio import audio_info, read_audio, write_audio
from mrep.io.json import read_json
from mrep.utils.synthetic import demo_song


def _profile(tmp_path: Path, fake_config: dict) -> Path:
    p = tmp_path / "fake.yaml"
    p.write_text(yaml.safe_dump(fake_config))
    return p


@pytest.mark.parametrize("ext,codec", [("ogg", "libvorbis"), ("m4a", "aac"), ("flac", "flac")])
def test_other_input_formats_are_ingested(demo_wav, fake_config, tmp_path, ext, codec):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    src = tmp_path / f"demo_song.{ext}"
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(demo_wav), "-c:a", codec, str(src)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {codec}: {proc.stderr.strip()[:200]}")
    info = audio_info(src)
    assert info["duration"] > 5 and info["channels"] == 2
    data, sr = read_audio(src)
    assert data.shape[1] == 2 and abs(len(data) / sr - info["duration"]) < 0.2
    rc = main(["analyze", str(src), "--profile", str(_profile(tmp_path, fake_config)), "-o", str(tmp_path / "out")])
    assert rc == 0
    man = read_json(tmp_path / "out" / "demo_song" / "manifest.json")
    assert man["source"]["format"] == ext and man["source"]["decoder"] in ("libsndfile", "ffprobe")
    assert man["source"]["first_class_format"] == (ext in ("wav", "flac", "mp3"))


def test_48k_input_is_resampled_and_tempo_detected(tmp_path):
    sr = 48000
    audio, truth = demo_song(sr=sr, bars=4)
    path = tmp_path / "demo48k.wav"
    write_audio(path, audio, sr)
    rc = main(["analyze", str(path), "--profile", "light", "-o", str(tmp_path / "out")])
    assert rc == 0
    project = tmp_path / "out" / "demo48k"
    norm = read_json(project / "01_normalized" / "normalization.json")
    assert norm["source_sample_rate"] == 48000 and norm["sample_rate"] == 44100
    tempo = read_json(project / "06_analysis" / "tempo.json")
    assert abs(tempo["tempo"] - truth["bpm"]) / truth["bpm"] < 0.04


def test_evaluate_command_on_fake_project(demo_wav, fake_config, tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["run", str(demo_wav), "--profile", str(_profile(tmp_path, fake_config)), "-o", str(out)])
    assert rc == 0
    audio, truth, stems = demo_song(sr=22050, bars=4, return_stems=True)
    stems_dir = tmp_path / "stems"
    for name, stem in stems.items():
        write_audio(stems_dir / f"{name}.wav", stem, 22050)
    truth_path = tmp_path / "truth.json"
    from mrep.io.json import write_json

    write_json(truth_path, truth)
    rc = main(["evaluate", str(out / "demo_song"), "--truth", str(truth_path), "--stems", str(stems_dir), "--json"])
    assert rc == 0
    report = read_json(out / "demo_song" / "06_analysis" / "evaluation.json")
    assert set(report["sections"]) >= {"tempo", "beats", "notes", "drums", "fx", "chords", "separation", "key", "time_signature"}
    assert report["sections"]["tempo"]["within_4_percent"] is True  # the fake MIR returns 120 bpm
    assert "si_sdr_db" in report["sections"]["separation"]["drums"]
    assert np.isfinite(report["sections"]["separation"]["bass"]["sdr_db"])
    assert "drum_onset_f1" in report["summary"]
