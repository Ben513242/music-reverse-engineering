"""Baseline drum *specialist*: band-splits a drum stem into kick / snare / hats / percussion.

This is a crude frequency-band approximation (not real component separation); labels
derived from it carry ``confidence_method = "band_filter_heuristic"``.
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import SeparationAdapter, SeparationResult
from mrep.adapters.registry import register
from mrep.utils.dsp import butter_filter, ensure_2d


@register("separation", "drum_bandsplit")
class DrumBandSplit(SeparationAdapter):
    model = "band_filter"
    model_version = "1.0"
    outputs = ("kick", "snare", "hats", "percussion")

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        x = ensure_2d(audio)
        kick_hz = float(self.config.get("kick_cutoff_hz", 150.0))
        snare_hz = (kick_hz, float(self.config.get("snare_upper_hz", 2500.0)))
        hat_hz = float(self.config.get("hat_cutoff_hz", 5000.0))
        kick = butter_filter(x, sr, kick_hz, "low")
        snare = butter_filter(x, sr, list(snare_hz), "band")
        hats = butter_filter(x, sr, hat_hz, "high")
        perc = (x - kick - snare - hats).astype(np.float32)
        stems = {"kick": kick, "snare": snare, "hats": hats, "percussion": perc}
        conf = {k: 0.3 for k in stems}
        return SeparationResult(stems=stems, sample_rate=sr, confidence=conf, confidence_method="band_filter_heuristic")
