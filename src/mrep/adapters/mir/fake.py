from __future__ import annotations

import numpy as np

from mrep.adapters.base import MIRAdapter, MIRResult
from mrep.adapters.registry import register


@register("mir", "fake")
class FakeMIR(MIRAdapter):
    model = "fake"
    model_version = "0"

    def analyze(self, audio: np.ndarray, sr: int) -> MIRResult:
        duration = len(audio) / sr
        bpm = float(self.config.get("bpm", 120.0))
        beat = 60.0 / bpm
        beats = [round(i * beat, 4) for i in range(int(duration / beat) + 1)]
        half = duration / 2
        return MIRResult(
            tempo_bpm=bpm,
            tempo_confidence=0.95,
            tempo_method="fake_constant",
            beat_times=beats,
            downbeat_times=beats[::4],
            beats_per_bar=4,
            time_signature="4/4",
            time_signature_confidence=None,
            time_signature_method="assumed",
            key_events=[{"start": 0.0, "end": duration, "tonic": "C", "mode": "major", "confidence": 0.9, "confidence_method": "fake_constant"}],
            sections=[{"start": 0.0, "end": half, "label": "A", "confidence": 0.7}, {"start": half, "end": duration, "label": "B", "confidence": 0.7}],
            loudness={"integrated_rms_db": -18.0, "peak_db": -1.0},
            dynamics=[{"start": 0.0, "end": half, "rms_db": -20.0, "dynamic": "mp"}, {"start": half, "end": duration, "rms_db": -14.0, "dynamic": "mf"}],
        )
