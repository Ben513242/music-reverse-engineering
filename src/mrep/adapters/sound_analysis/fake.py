from __future__ import annotations

import numpy as np

from mrep.adapters.base import SoundAnalysisAdapter, SoundAnalysisResult
from mrep.adapters.registry import register
from mrep.domain.labels import Candidate, SoundLabel
from mrep.domain.provenance import Provenance


@register("sound_analysis", "fake")
class FakeSoundAnalysis(SoundAnalysisAdapter):
    model = "fake"
    model_version = "0"

    def analyze_sound(self, audio: np.ndarray, sr: int, *, stem_name: str, label: SoundLabel | None, provenance: Provenance) -> SoundAnalysisResult:
        return SoundAnalysisResult(
            features={"attack_ms": 12.0, "release_ms": 420.0, "centroid_hz": 1500.0, "stereo_width": 0.3, "fake": True},
            sound_design=["short_attack", "warm"],
            patch_candidates=[Candidate("digital_pluck", 0.6, 0.6), Candidate("analog_style_pad", 0.4, 0.4)],
            confidence=0.6,
            confidence_method="fake_constant",
        )
