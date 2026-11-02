"""Baseline multi-label stem classification from spectral / envelope features.

It is deliberately honest: families that cannot be distinguished without a learned model
are reported as *candidates with probabilities* and the family falls back to ``harmonic``.
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import ClassificationAdapter, ClassificationResult
from mrep.adapters.registry import register
from mrep.domain.labels import FREQUENCY_BANDS_HZ, Candidate, SoundLabel
from mrep.domain.provenance import Provenance
from mrep.utils.dsp import (
    EPS,
    band_energy,
    chroma_from_stft,
    ensure_2d,
    fft_freqs,
    onset_strength,
    pick_peaks,
    rms,
    rms_frames,
    spectral_centroid,
    spectral_flatness,
    stft,
    to_mono,
)

DRUM_COMPONENT_SUBTYPE = {"kick": "kick", "snare": "snare", "hats": "closed_hat", "percussion": "other_percussion", "clap": "clap", "closed_hat": "closed_hat", "open_hat": "open_hat"}


def _attack_release(env: np.ndarray, onsets: np.ndarray, fps: float) -> tuple[float, float]:
    """Mean attack (onset->peak) and release (peak -> -20 dB) in milliseconds."""
    attacks, releases = [], []
    n = len(env)
    for i, p in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else n
        seg = env[p:nxt]
        if seg.size < 2:
            continue
        k = int(np.argmax(seg))
        attacks.append(k / fps)
        peak = seg[k]
        below = np.where(seg[k:] < peak * 0.1)[0]
        releases.append((below[0] if below.size else len(seg) - k) / fps)
    a = float(np.mean(attacks)) * 1000 if attacks else 0.0
    r = float(np.mean(releases)) * 1000 if releases else 0.0
    return a, r


def extract_features(audio: np.ndarray, sr: int) -> dict:
    x2 = ensure_2d(audio)
    x = to_mono(x2)
    duration = max(len(x) / sr, 1e-3)
    n_fft, hop = 2048, 512
    fps = sr / hop
    S = np.abs(stft(x, n_fft, hop))
    freqs = fft_freqs(sr, n_fft)
    frame_e = S.sum(axis=0) + EPS
    active = frame_e > frame_e.max() * 0.05
    bands = band_energy(S, freqs, FREQUENCY_BANDS_HZ)
    centroid = float(np.average(spectral_centroid(S, freqs), weights=frame_e))
    flat_frames = spectral_flatness(S)
    flatness = float(np.median(flat_frames[active])) if active.any() else float(np.median(flat_frames))
    env = rms_frames(x, n_fft, hop)
    onsets = pick_peaks(onset_strength(S), pre_avg=int(0.1 * fps), post_avg=int(0.1 * fps), delta=0.08, wait=int(0.05 * fps))
    onset_rate = len(onsets) / duration
    attack_ms, release_ms = _attack_release(env, onsets, fps)
    act = env > env.max() * 0.1 if env.max() > 0 else np.zeros_like(env, dtype=bool)
    # mean length of active regions (sustain)
    runs, cur = [], 0
    for a in act:
        if a:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    sustain_s = float(np.mean(runs)) / fps if runs else 0.0
    width = 0.0
    if x2.shape[1] >= 2 and rms(x2[:, 0]) > 1e-5 and rms(x2[:, 1]) > 1e-5:
        corr = float(np.corrcoef(x2[:, 0], x2[:, 1])[0, 1])
        width = float(np.clip(1.0 - corr, 0.0, 2.0)) if np.isfinite(corr) else 0.0
    chroma = chroma_from_stft(S, freqs)
    cm = chroma[:, active].mean(axis=1) if active.any() else chroma.mean(axis=1)
    p = cm / (cm.sum() + EPS)
    entropy = float(-(p * np.log(p + EPS)).sum() / np.log(12))
    strong_pcs = int(np.sum(cm > 0.5 * cm.max())) if cm.max() > 0 else 0
    return {
        "duration": duration,
        "rms": rms(x),
        "bands": bands,
        "centroid_hz": centroid,
        "flatness": flatness,
        "onset_rate": onset_rate,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "sustain_s": sustain_s,
        "stereo_width": width,
        "chroma_entropy": entropy,
        "strong_pitch_classes": strong_pcs,
        "channels": int(x2.shape[1]),
    }


@register("classification", "heuristic")
class HeuristicClassification(ClassificationAdapter):
    model = "spectral_rules"
    model_version = "1.0"

    def classify(self, audio: np.ndarray, sr: int, *, stem_name: str, source_hint: str, provenance: Provenance) -> ClassificationResult:
        if rms(to_mono(audio)) < 1e-5:
            label = SoundLabel(source_family="unknown", confidence=None, confidence_method="unavailable", provenance=provenance, features={"silent": True})
            return ClassificationResult(label=label, features={"silent": True})
        f = extract_features(audio, sr)
        bands = f["bands"]
        freq_roles = [name for name, frac in bands.items() if frac >= 0.2]
        if not freq_roles:
            freq_roles = [max(bands, key=bands.get)]
        design: list[str] = []
        if f["centroid_hz"] > 3000:
            design.append("bright")
        elif f["centroid_hz"] < 800:
            design.append("dark")
        elif f["flatness"] < 0.1:
            design.append("warm")
        if f["flatness"] > 0.3:
            design.append("noise_based")
        if 0 < f["attack_ms"] < 20:
            design.append("short_attack")
        elif f["attack_ms"] > 80:
            design.append("slow_attack")
        if f["release_ms"] > 500:
            design.append("long_release")
        elif 0 < f["release_ms"] < 120:
            design.append("short_decay")
        if f["channels"] >= 2:
            design.append("wide_stereo" if f["stereo_width"] > 0.4 else "mono" if f["stereo_width"] < 0.05 else "")
        design = [d for d in design if d]

        component = stem_name.split("/")[-1] if "/" in stem_name else None
        is_drum = source_hint == "drums" or stem_name.startswith("drums")
        low_frac = bands["sub_bass"] + bands["low_bass"]

        if is_drum:
            subtype = DRUM_COMPONENT_SUBTYPE.get(component or "", None)
            label = SoundLabel(
                source_family="drums",
                source_subtype=subtype,
                musical_role=["rhythm", "percussion"],
                frequency_role=freq_roles,
                sound_design=design,
                confidence=0.5 if subtype else 0.6,
                confidence_method="stem_origin_heuristic",
                provenance=provenance,
                features=f,
            )
        elif source_hint == "bass" or low_frac > 0.6:
            roles = ["bassline"]
            fr = [r for r in ("sub_bass", "low_bass") if bands[r] >= 0.15] or ["low_bass"]
            label = SoundLabel(
                source_family="bass",
                source_subtype=None,
                source_candidates=[Candidate("synth_bass", None), Candidate("electric_bass", None), Candidate("808", None)],
                musical_role=roles,
                frequency_role=fr,
                sound_design=design,
                confidence=float(np.clip(low_frac, 0.0, 1.0)),
                confidence_method="low_band_energy_heuristic",
                provenance=provenance,
                features=f,
            )
        else:
            scores = {
                "piano": (f["attack_ms"] < 40) * 1.0 + (120 < f["release_ms"] < 1500) * 0.6 + (1.0 < f["onset_rate"] < 8.0) * 0.5 + (f["strong_pitch_classes"] >= 3) * 0.4,
                "strings": (f["attack_ms"] > 60) * 1.0 + (f["sustain_s"] > 1.5) * 0.8 + (f["onset_rate"] < 1.0) * 0.6,
                "synthesizer": (f["stereo_width"] > 0.4) * 0.7 + (f["flatness"] > 0.15) * 0.6 + (f["centroid_hz"] > 2500) * 0.4 + (f["onset_rate"] > 4.0) * 0.4 + 0.3,
                "guitar": (f["attack_ms"] < 30) * 0.7 + (0.2 < f["sustain_s"] < 1.5) * 0.5 + (1000 < f["centroid_hz"] < 3000) * 0.5,
            }
            total = sum(scores.values()) + EPS
            cands = sorted(((k, v / total) for k, v in scores.items()), key=lambda kv: kv[1], reverse=True)
            best, prob = cands[0]
            family = best if prob >= 0.45 else "harmonic"
            roles: list[str] = []
            if f["onset_rate"] < 0.6 and f["sustain_s"] > 1.5:
                roles.append("pad")
                if f["strong_pitch_classes"] <= 2:
                    roles.append("drone")
            if f["strong_pitch_classes"] >= 3:
                roles.append("chord")
            if f["onset_rate"] > 2.0 and 0 < f["release_ms"] < 150:
                roles.append("pluck")
            if f["strong_pitch_classes"] <= 2 and f["centroid_hz"] > 800 and 1.0 <= f["onset_rate"] <= 8.0:
                roles.append("lead")
            if not roles:
                roles.append("melody" if f["strong_pitch_classes"] <= 2 else "chord")
            label = SoundLabel(
                source_family=family,
                source_subtype=None,
                source_candidates=[Candidate(k, round(v, 3)) for k, v in cands],
                musical_role=roles,
                frequency_role=freq_roles,
                sound_design=design,
                confidence=float(np.clip(prob, 0.0, 1.0)),
                confidence_method="spectral_rules_heuristic",
                provenance=provenance,
                features=f,
            )
        return ClassificationResult(label=label, features=f)
