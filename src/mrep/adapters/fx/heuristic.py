"""Baseline FX detection (riser / downlifter / white_noise / impact).

Sustained noise is found on the *stationary* spectrum (a running minimum over time removes
drum transients); the class comes from the spectral-centroid trend. FX are never forced
into notation: they become ``FXEvent``s with audio clips and features.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, minimum_filter

from mrep.adapters.base import FXAdapter
from mrep.adapters.registry import register
from mrep.domain.events import FXEvent
from mrep.domain.provenance import Provenance
from mrep.utils.dsp import (
    EPS,
    fft_freqs,
    onset_strength,
    pick_peaks,
    spectral_flatness,
    stft,
    to_mono,
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


@register("fx", "heuristic")
class HeuristicFX(FXAdapter):
    model = "stationary_noise_rules"
    model_version = "1.1"

    def detect(self, audio: np.ndarray, sr: int, *, provenance: Provenance) -> list[FXEvent]:
        x = to_mono(audio)
        if x.size == 0 or float(np.max(np.abs(x))) < 1e-5:
            return []
        cfg = self.config
        n_fft, hop = 2048, 512
        fps = sr / hop
        S = np.abs(stft(x, n_fft, hop))
        freqs = fft_freqs(sr, n_fft)
        win = max(3, int(float(cfg.get("stationary_seconds", 0.3)) * fps))
        Smin = minimum_filter(S, size=(1, win), mode="nearest")  # removes drum transients
        band = freqs >= float(cfg.get("band_low_hz", 2000.0))
        Sb = Smin[band]
        e_db = median_filter(10 * np.log10((Sb**2).sum(axis=0) + EPS), size=win, mode="nearest")
        sb_db = 20 * np.log10(Sb + 1e-9)
        density = (sb_db > (sb_db.max(axis=0)[None, :] - 15.0)).mean(axis=0)
        density = median_filter(density, size=win, mode="nearest")
        floor_db = max(float(np.percentile(e_db, 20)), -80.0)
        active = e_db > floor_db + float(cfg.get("level_rise_db", 10.0))
        min_frames = int(float(cfg.get("min_seconds", 0.8)) * fps)
        slope_thr = float(cfg.get("slope_db_per_s", 8.0))
        rise_thr = float(cfg.get("min_rise_db", 12.0))
        mid = freqs >= 300.0
        cent = (freqs[mid][:, None] * Smin[mid]).sum(axis=0) / (Smin[mid].sum(axis=0) + EPS)
        events: list[FXEvent] = []
        noisy = np.zeros(len(e_db), dtype=bool)
        for a, b in _runs(active):
            if b - a < min_frames:
                continue
            seg = e_db[a:b]
            t = np.arange(b - a) / fps
            slope = float(np.polyfit(t, seg, 1)[0])
            rise = float(np.percentile(seg, 90) - np.percentile(seg, 10))
            steps = np.diff(median_filter(seg, size=max(3, int(0.2 * fps)), mode="nearest"))
            frac_up = float(np.mean(steps > 0)) if steps.size else 0.0
            frac_down = float(np.mean(steps < 0)) if steps.size else 0.0
            mono = float(cfg.get("monotonic_fraction", 0.45))
            dens = float(np.median(density[a:b]))
            k = max(1, (b - a) // 5)
            c0 = float(np.median(cent[a : a + k]))
            c1 = float(np.median(cent[b - k : b]))
            cent_ratio = c1 / (c0 + EPS)
            if slope > slope_thr and rise >= rise_thr and cent_ratio > 1.5 and frac_up >= mono:
                cls = "riser"  # steadily rising level + rising centroid
            elif slope < -slope_thr and rise >= rise_thr and cent_ratio < 1 / 1.5 and frac_down >= mono:
                cls = "downlifter"
            elif dens > float(cfg.get("noise_density", 0.15)):
                cls = "white_noise"
            else:
                continue  # sustained but sparse/tonal high content (e.g. strings) - not an FX
            noisy[a:b] = True
            conf = float(np.clip(0.3 + 0.4 * min(rise / 20.0, 1.0) + 0.3 * min(dens / 0.3, 1.0), 0.0, 1.0))
            events.append(
                FXEvent(
                    fx_class=cls,
                    start=round(a / fps, 3),
                    end=round(b / fps, 3),
                    confidence=round(conf, 4),
                    confidence_method="stationary_noise_rules",
                    provenance=provenance,
                    features={"slope_db_per_s": round(slope, 2), "rise_db": round(rise, 1), "monotonic_fraction": round(max(frac_up, frac_down), 3), "density": round(dens, 4), "centroid_start_hz": round(c0, 1), "centroid_end_hz": round(c1, 1)},
                )
            )
        # impacts: very loud broadband onsets with a long decay, outside noise runs
        env = onset_strength(S)
        peaks = pick_peaks(env, pre_avg=int(0.2 * fps), post_avg=int(0.2 * fps), delta=0.2, wait=int(0.3 * fps))
        if peaks.size:
            e_full = 10 * np.log10((S**2).sum(axis=0) + EPS)
            flat_full = spectral_flatness(S)
            strong = e_full[peaks] >= np.percentile(e_full[peaks], 90)
            for p, s in zip(peaks, strong):
                if not s or noisy[p]:
                    continue
                seg = e_full[p : p + int(1.5 * fps)]
                below = np.where(seg < seg[0] - 20)[0]
                decay = (below[0] / fps) if below.size else len(seg) / fps
                low = float((S[freqs < 150, p] ** 2).sum() / ((S[:, p] ** 2).sum() + EPS))
                if decay > float(cfg.get("impact_min_decay_s", 0.4)) and flat_full[p] > 0.2 and low > 0.25:
                    events.append(FXEvent(fx_class="impact", start=round(p / fps, 3), end=round(p / fps + decay, 3), confidence=0.4, confidence_method="stationary_noise_rules", provenance=provenance, features={"decay_s": round(decay, 3), "low_ratio": round(low, 3)}))
        events.sort(key=lambda e: e.start)
        return events
