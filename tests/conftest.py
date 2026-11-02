from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrep.io.audio import write_audio  # noqa: E402
from mrep.utils.synthetic import demo_song  # noqa: E402


@pytest.fixture(scope="session")
def demo_audio() -> tuple[np.ndarray, dict, int]:
    sr = 22050
    audio, truth = demo_song(sr=sr, bpm=120.0, bars=4)
    return audio, truth, sr


@pytest.fixture(scope="session")
def demo_wav(tmp_path_factory, demo_audio) -> Path:
    audio, _truth, sr = demo_audio
    path = tmp_path_factory.mktemp("audio") / "demo_song.wav"
    write_audio(path, audio, sr)
    return path


@pytest.fixture(scope="session")
def demo_audio_44k() -> tuple[np.ndarray, dict, int]:
    """Full-rate, 8-bar synthetic song for the end-to-end tests of the baseline DSP adapters."""
    sr = 44100
    audio, truth = demo_song(sr=sr, bpm=120.0, bars=8)
    return audio, truth, sr


@pytest.fixture(scope="session")
def demo_wav_44k(tmp_path_factory, demo_audio_44k) -> Path:
    audio, _truth, sr = demo_audio_44k
    path = tmp_path_factory.mktemp("audio44k") / "demo_song.wav"
    write_audio(path, audio, sr)
    return path


@pytest.fixture
def fake_config() -> dict:
    """Pipeline configuration that uses only fake adapters (no DSP, no models)."""
    from mrep.config import load_config

    cfg = load_config("default")
    cfg["audio"]["target_sample_rate"] = 22050
    st = cfg["stages"]
    st["global_analysis"]["adapter"] = "fake"
    st["separation"].update({"discovery_adapter": "fake", "broad_adapter": "fake", "specialists": {"drums": "fake_components"}, "chunk_seconds": 0})
    st["classification"]["adapter"] = "fake"
    st["classification"]["sound_analysis"]["adapter"] = "fake"
    st["transcription"].update({"drums_adapters": ["fake"], "pitched_adapters": ["fake"], "full_mix_amt": True})
    st["transcription"]["fx"]["adapter"] = "fake"
    st["reconstruction"].update({"adapter": "fake", "mp3": False})
    return cfg
