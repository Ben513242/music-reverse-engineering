"""Integration tests for optional ML adapters. Skipped when the heavy dependencies are absent."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mrep.adapters import create_adapter
from mrep.adapters.base import AdapterUnavailable
from mrep.domain.provenance import Provenance
from mrep.utils.synthetic import demo_song

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_demucs_adapter_separates_demo():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("demucs") is None:
        pytest.skip("torch/demucs not installed")
    adapter = create_adapter("separation", "demucs", {"allow_download": False}, "cpu")
    try:
        adapter.ensure_models(download=False)
    except AdapterUnavailable as exc:
        pytest.skip(f"demucs weights not present: {exc}")
    audio, _truth = demo_song(sr=44100, bars=2)
    res = adapter.separate(audio, 44100)
    assert {"drums", "bass", "other", "vocals"} <= set(res.stems)
    assert all(v.shape == audio.shape for v in res.stems.values())
    assert np.sqrt(np.mean(res.stems["drums"] ** 2)) > 0


def test_basic_pitch_adapter_transcribes_tone():
    if importlib.util.find_spec("basic_pitch") is None:
        pytest.skip("basic-pitch not installed")
    from mrep.utils.synthetic import tone

    sr = 22050
    x = np.concatenate([tone(m, 0.5, sr) for m in (60, 64, 67)])
    adapter = create_adapter("transcription", "basic_pitch", {}, "cpu")
    notes = adapter.transcribe(x, sr, track="t", instrument="piano", provenance=Provenance(stage="t", adapter="basic_pitch"))
    assert {n.midi_note for n in notes} >= {60, 64, 67}
