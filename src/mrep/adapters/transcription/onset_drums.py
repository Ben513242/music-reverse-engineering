"""Baseline drum event detection: onset picking + band-energy class rules.

Minimum classes: kick / snare / closed_hat / open_hat (+ clap, low_tom, other_percussion).
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import DrumTranscriptionAdapter
from mrep.adapters.registry import register
from mrep.domain.events import DrumEvent
from mrep.domain.provenance import Provenance
from mrep.utils.dsp import (
    EPS,
    band_energy_frames,
    fft_freqs,
    onset_strength,
    pick_peaks,
    spectral_flatness,
    stft,
    to_mono,
)
from mrep.utils.music import amplitude_to_velocity

BANDS = {"low": (20.0, 150.0), "lowmid": (150.0, 500.0), "mid": (500.0, 2500.0), "high": (2500.0, 7000.0), "vhigh": (7000.0, 22050.0)}


@register("drum_transcription", "onset_bands")
class OnsetBandDrums(DrumTranscriptionAdapter):
    model = "onset_band_rules"
    model_version = "1.0"

    def transcribe_drums(self, audio: np.ndarray, sr: int, *, track: str, class_hint: str | None, provenance: Provenance) -> list[DrumEvent]:
        x = to_mono(audio)
        if x.size == 0 or float(np.max(np.abs(x))) < 1e-5:
            return []
        n_fft, hop = 1024, 256
        fps = sr / hop
        S = np.abs(stft(x, n_fft, hop))
        freqs = fft_freqs(sr, n_fft)
        env = onset_strength(S)
        delta = float(self.config.get("onset_delta", 0.1))
        peaks = pick_peaks(env, pre_max=2, post_max=2, pre_avg=int(0.1 * fps), post_avg=int(0.1 * fps), delta=delta, wait=int(0.04 * fps))
        if peaks.size == 0:
            return []
        be = band_energy_frames(S, freqs, BANDS)
        flat = spectral_flatness(S)
        win = max(1, int(0.04 * fps))
        pre = max(1, int(0.03 * fps))
        T = S.shape[1]
        rows: list[dict] = []
        for p in peaks:
            a, b = int(p), min(T, int(p) + win)
            a0 = max(0, a - pre - 1)
            # energy that is *new* at this onset: window energy minus the pre-onset tail of earlier hits
            e = {k: float(max(v[a:b].sum() - v[a0 : a - 1].sum() * (b - a) / max(1, (a - 1 - a0)), 0.0)) for k, v in be.items()}
            tot = sum(e.values()) + EPS
            r = {k: v / tot for k, v in e.items()}
            hf = be["vhigh"]
            seg_total = hf[a : min(T, a + int(0.5 * fps))]  # decay of the very-high band (hats live there)
            pk = int(np.argmax(seg_total)) if seg_total.size else 0
            peak_e = float(seg_total[pk]) if seg_total.size else 0.0
            below = np.where(seg_total[pk:] < peak_e * 0.1)[0]
            decay_s = ((pk + below[0]) / fps) if below.size else len(seg_total) / fps
            multi = bool(np.sum(env[a : a + int(0.03 * fps) + 1] > 0.5 * env[a]) >= 2) if a + 1 < T else False
            fl = float(np.median(flat[a:b]))
            scores = {
                "kick": 1.4 * r["low"] + 0.3 * r["lowmid"],
                "snare": 0.8 * r["lowmid"] + 1.0 * r["mid"] + (0.15 if fl > 0.08 else 0.0),
                "hat": 1.5 * (r["high"] + r["vhigh"]) - r["low"],
                "clap": 0.6 * r["mid"] + 0.6 * r["high"] + (0.5 if multi else 0.0) - 0.5,
                "low_tom": 0.6 * r["low"] + 0.9 * r["lowmid"] - r["high"] - (0.6 if fl > 0.05 else 0.0),
            }
            rows.append({"frame": a, "ratios": r, "scores": scores, "decay": decay_s, "amp": float(np.sqrt(tot)), "flat": fl})
        ref = float(np.percentile([r["amp"] for r in rows], 95)) + EPS
        events: list[DrumEvent] = []
        tonal_gate = float(self.config.get("tonal_flatness_gate", 0.02))
        for r in rows:
            sc = r["scores"]
            vals = np.array(list(sc.values()))
            keys = list(sc.keys())
            ex = np.exp((vals - vals.max()) * 4.0)
            probs = ex / ex.sum()
            best_i = int(np.argmax(probs))
            cls = keys[best_i]
            conf = float(probs[best_i])
            method = "band_energy_softmax"
            rt = r["ratios"]
            if cls == "hat":
                cls = "open_hat" if r["decay"] > float(self.config.get("open_hat_decay_s", 0.15)) else "closed_hat"
            # a purely tonal transient with no low end is most likely a pitched-instrument leak
            if class_hint is None and r["flat"] < tonal_gate and rt["low"] < 0.2 and rt["vhigh"] < 0.1:
                cls, conf, method = "other_percussion", min(conf, 0.3), "tonal_leak_suspect"
            if class_hint:
                cls = class_hint
                conf = max(conf, 0.6)
                method = "stem_hint"
            class_scores = {k: round(float(p), 4) for k, p in zip(keys, probs)}
            vel = amplitude_to_velocity(r["amp"] / ref)
            t = round(r["frame"] / fps, 4)
            extra: list[tuple[str, float]] = []
            if class_hint is None:
                # coincident hits (multi-label): kick/snare with a hat on top, hat with a snare underneath
                if cls in ("kick", "snare", "low_tom", "clap") and rt["vhigh"] >= float(self.config.get("coincident_hat_ratio", 0.06)):
                    extra.append(("closed_hat", rt["vhigh"]))
                if cls in ("closed_hat", "open_hat") and rt["lowmid"] + rt["mid"] >= float(self.config.get("coincident_snare_ratio", 0.4)) and r["flat"] >= tonal_gate:
                    extra.append(("snare", rt["lowmid"] + rt["mid"]))
                if cls in ("closed_hat", "open_hat", "snare") and rt["low"] >= float(self.config.get("coincident_kick_ratio", 0.4)):
                    extra.append(("kick", rt["low"]))
            for ecls, score in extra:
                events.append(DrumEvent(time=t, drum_class=ecls, velocity=max(1, int(vel * 0.7)), duration=0.05, track=track, class_scores={ecls: round(float(score), 4)}, confidence=0.4, confidence_method="band_energy_coincident", provenance=provenance))
            events.append(
                DrumEvent(
                    time=t,
                    drum_class=cls,
                    velocity=vel,
                    duration=round(r["decay"], 3),
                    track=track,
                    class_scores=class_scores,
                    confidence=round(conf, 4),
                    confidence_method=method,
                    provenance=provenance,
                )
            )
        events.sort(key=lambda e: (e.time, e.drum_class))
        return events
