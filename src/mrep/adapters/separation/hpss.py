"""Baseline broad separation: median-filter HPSS + low-pass bass split (no model needed).

Produces ``drums`` (percussive), ``bass`` (harmonic < cutoff) and ``other`` (remaining
harmonic content). Quality is far below neural separators; it exists so that the whole
pipeline runs anywhere and so that better adapters can be swapped in via config.
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import SeparationAdapter, SeparationResult
from mrep.adapters.registry import register
from mrep.utils.dsp import butter_filter, ensure_2d, hpss, istft, stft


@register("separation", "hpss")
class HPSSSeparation(SeparationAdapter):
    model = "median_filter_hpss"
    model_version = "1.0"
    outputs = ("drums", "bass", "other")

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        x = ensure_2d(audio)
        n_fft = int(self.config.get("n_fft", 2048))
        hop = n_fft // 4
        kh = int(self.config.get("kernel_harmonic", 31))
        kp = int(self.config.get("kernel_percussive", 31))
        cutoff = float(self.config.get("bass_cutoff_hz", 200.0))
        drums = np.zeros_like(x)
        harm = np.zeros_like(x)
        for ch in range(x.shape[1]):
            S = stft(x[:, ch], n_fft, hop)
            H, P = hpss(S, kh, kp)
            harm[:, ch] = istft(H, n_fft, hop, len(x))
            drums[:, ch] = istft(P, n_fft, hop, len(x))
        bass = butter_filter(harm, sr, cutoff, "low", order=4)
        other = (harm - bass).astype(np.float32)
        stems = {"drums": drums.astype(np.float32), "bass": bass.astype(np.float32), "other": other}
        return SeparationResult(stems=stems, sample_rate=sr, confidence={k: None for k in stems}, confidence_method="unavailable")
