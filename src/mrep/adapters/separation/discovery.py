"""Baseline source discovery from energy ratios. Only drums / bass / harmonic are estimated;
instrument families that cannot be told apart without a model are reported as ``null``."""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import DiscoveryAdapter, DiscoveryResult
from mrep.adapters.registry import register
from mrep.utils.dsp import fft_freqs, hpss, onset_strength, pick_peaks, stft, to_mono

UNKNOWN_FAMILIES = ("piano", "synthesizer", "strings", "guitar", "vocal", "brass", "woodwinds", "organ", "harp")


@register("discovery", "energy_heuristic")
class EnergyHeuristicDiscovery(DiscoveryAdapter):
    model = "energy_ratio_rules"
    model_version = "1.0"

    def discover(self, audio: np.ndarray, sr: int) -> DiscoveryResult:
        x = to_mono(audio)
        n_fft, hop = 2048, 512
        S = stft(x, n_fft, hop)
        freqs = fft_freqs(sr, n_fft)
        H, P = hpss(S)
        Hm, Pm = np.abs(H), np.abs(P)
        e_tot = float(np.sum(np.abs(S) ** 2)) + 1e-12
        e_h = float(np.sum(Hm**2))
        duration = max(len(x) / sr, 1e-3)
        onsets = pick_peaks(onset_strength(Pm), delta=0.1)
        onset_rate = len(onsets) / duration
        hi = freqs >= 150.0  # ignore the sub/low band so a loud sub bass does not hide the drums
        perc_ratio = float(np.sum(Pm[hi] ** 2)) / (float(np.sum(Pm[hi] ** 2)) + float(np.sum(Hm[hi] ** 2)) + 1e-12)
        low = (freqs >= 30) & (freqs < 200)
        bass_ratio = float(np.sum(Hm[low] ** 2)) / (e_h + 1e-12)
        harm_high_ratio = float(np.sum(Hm[~low] ** 2)) / e_tot
        probs: dict[str, float | None] = {
            "drums": float(np.clip(1.6 * perc_ratio + 0.3 * min(onset_rate / 4.0, 1.0), 0.0, 1.0)),
            "bass": float(np.clip(bass_ratio * 4.0, 0.0, 1.0)),
            "harmonic": float(np.clip(harm_high_ratio * 2.0, 0.0, 1.0)),
        }
        for fam in UNKNOWN_FAMILIES:
            probs[fam] = None
        return DiscoveryResult(probabilities=probs, confidence_method="energy_ratio_heuristic")
