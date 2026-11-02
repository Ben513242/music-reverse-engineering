"""Baseline timbre analysis: ADSR, spectral shape, harmonicity / noise, inharmonicity,
modulation (vibrato / tremolo / filter movement), stereo width -> sound-design descriptors and
*approximate* patch candidates (never a claim about a specific preset)."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter

from mrep.adapters.base import SoundAnalysisAdapter, SoundAnalysisResult
from mrep.adapters.registry import register
from mrep.analysis.pitch_bend import f0_track
from mrep.domain.labels import Candidate, SoundLabel
from mrep.domain.provenance import Provenance
from mrep.utils.dsp import (
    EPS,
    ensure_2d,
    fft_freqs,
    onset_strength,
    peak,
    pick_peaks,
    rms,
    rms_frames,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
    stft,
    to_mono,
)

PATCH_TEMPLATES: dict[str, set[str]] = {
    "analog_style_pad": {"slow_attack", "long_release", "warm", "wide_stereo"},
    "supersaw_lead": {"bright", "wide_stereo", "supersaw_like", "short_attack"},
    "fm_bell": {"bell_like", "long_release", "bright", "short_attack"},
    "digital_pluck": {"pluck", "short_attack", "short_decay", "bright"},
    "subtractive_synth_bass": {"dark", "short_attack", "subtractive", "mono"},
    "sub_sine_bass": {"dark", "mono", "short_attack", "pure_tone"},
    "string_ensemble": {"slow_attack", "long_release", "warm", "vibrato"},
    "electric_piano": {"short_attack", "warm", "bell_like", "long_release"},
    "acoustic_piano": {"short_attack", "warm", "long_release", "pluck"},
    "noise_texture": {"noise_based", "slow_attack", "long_release"},
}


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


def _mod_peak(signal_dev: np.ndarray, rate: float, band: tuple[float, float]) -> tuple[float | None, float]:
    """Return (frequency of the dominant modulation in ``band`` or None, prominence)."""
    if signal_dev.size < 8:
        return None, 0.0
    spec = np.abs(np.fft.rfft(signal_dev * np.hanning(signal_dev.size)))
    freqs = np.fft.rfftfreq(signal_dev.size, 1.0 / rate)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any() or spec[1:].sum() <= 0:
        return None, 0.0
    k = int(np.argmax(np.where(sel, spec, 0.0)))
    baseline = float(np.median(spec[1:]) + EPS)
    prominence = float(spec[k] / baseline)
    return float(freqs[k]), prominence


@register("sound_analysis", "dsp")
class DSPSoundAnalysis(SoundAnalysisAdapter):
    model = "descriptor_rules"
    model_version = "1.0"

    def analyze_sound(self, audio: np.ndarray, sr: int, *, stem_name: str, label: SoundLabel | None, provenance: Provenance) -> SoundAnalysisResult:
        cfg = self.config
        x2 = ensure_2d(audio)
        x = to_mono(x2)
        if peak(x) < 1e-5:
            return SoundAnalysisResult(features={"silent": True}, confidence=None, confidence_method="unavailable")
        n_fft, hop = 2048, 256
        fps = sr / hop
        S = np.abs(stft(x, n_fft, hop))
        freqs = fft_freqs(sr, n_fft)
        # fine envelope (own hop) so that attack times get ~5-10 ms resolution
        env_hop = 128
        efps = sr / env_hop
        env = rms_frames(x, 256, env_hop)
        env_db = 20 * np.log10(env + EPS)
        active = env_db > env_db.max() - float(cfg.get("active_range_db", 30.0))
        # note/phrase segments = validated onset-to-onset windows. An onset candidate (spectral
        # flux peak) only counts when the level really jumps (>= 6 dB over the preceding dip), so a
        # slow pad swell is one segment, while repeated plucks are many.
        cand = pick_peaks(onset_strength(S), pre_avg=int(0.1 * fps), post_avg=int(0.1 * fps), delta=0.1, wait=int(0.08 * fps))
        onsets: list[int] = []
        for c in cand:
            ei = int(round(c * hop / env_hop))
            if ei >= len(env):
                continue
            after = env[ei : min(len(env), ei + int(0.05 * efps) + 1)].max()
            before = env[max(0, ei - int(0.1 * efps)) : ei + 1].min() if ei > 0 else 0.0
            if not onsets or after >= 2.0 * before + EPS:
                onsets.append(ei)
        T = len(env)
        if onsets:
            bounds = onsets + [T]
            segs = [(int(a), int(min(b, a + int(2.0 * efps)))) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= int(0.05 * efps)]
        else:
            segs = [(a, b) for a, b in _runs(active) if b - a >= int(0.05 * efps)]

        # ---- ADSR ---------------------------------------------------------------------
        attacks, decays, sustains, releases = [], [], [], []
        for a, b in segs:
            e = env[a:b]
            look = e[: max(2, int(0.5 * efps))]
            pk = int(np.argmax(look))
            peak_v = e[pk] + EPS
            # rise time: from the last point at 10 % of the rise above the pre-onset floor up to the
            # peak (independent of onset-frame jitter and of the previous note's tail)
            floor_v = float(e[: pk + 1].min())
            thr = floor_v + 0.1 * (peak_v - floor_v)
            below = np.where(e[: pk + 1] <= thr)[0]
            start = int(below[-1]) if below.size else 0
            attacks.append((pk - start) / efps * 1000)
            tail = e[pk:]
            if (b - a) / efps > 0.3:
                sus = float(np.median(tail[len(tail) // 2 :]) / peak_v)
            else:
                sus = None
            target = (sus * 1.1 if sus is not None else 0.5) * peak_v
            below = np.where(tail < target)[0]
            decays.append((below[0] / efps * 1000) if below.size else len(tail) / efps * 1000)
            if sus is not None:
                sustains.append(min(sus, 1.0))
            level = (sus if sus is not None else 1.0) * peak_v * 0.5
            above = np.where(e > level)[0]
            if above.size and e[-1] < level:  # only when the segment really ends with a release
                releases.append((len(e) - above[-1]) / efps * 1000)
        attack_ms = float(np.median(attacks)) if attacks else 0.0
        decay_ms = float(np.median(decays)) if decays else 0.0
        sustain_level = float(np.median(sustains)) if sustains else 0.0
        release_ms = float(np.median(releases)) if releases else 0.0

        # ---- spectral shape -------------------------------------------------------------
        frame_e = S.sum(axis=0) + EPS
        act = frame_e > frame_e.max() * 0.05
        cent = spectral_centroid(S, freqs)
        centroid = float(np.average(cent[act], weights=frame_e[act])) if act.any() else float(cent.mean())
        rolloff = float(np.median(spectral_rolloff(S, freqs)[act])) if act.any() else 0.0
        flat = float(np.median(spectral_flatness(S)[act])) if act.any() else 0.0
        floor = median_filter(S, size=(31, 1), mode="nearest")
        peaks = S > 2.5 * floor
        harmonicity = float(np.sum((S * peaks) ** 2) / (np.sum(S**2) + EPS))
        noise_ratio = 1.0 - harmonicity

        # ---- f0 / inharmonicity / vibrato ---------------------------------------------------
        times, f0 = f0_track(x, sr, frame=2048, hop=hop)
        voiced = ~np.isnan(f0)
        stable = voiced.copy()
        stable[1:] &= voiced[:-1] & (np.abs(np.nan_to_num(f0[1:]) - np.nan_to_num(f0[:-1])) < 0.01 * np.nan_to_num(f0[1:], nan=1.0))
        inharm = []
        bin_hz = sr / n_fft
        for t in np.where(stable)[0][:: max(1, int(stable.sum() // 200) or 1)]:
            base = f0[t]
            fund_bin = int(round(base / bin_hz))
            if fund_bin >= len(freqs):
                continue
            fund_amp = S[max(0, fund_bin - 1) : fund_bin + 2, t].max()
            for k in range(2, 7):
                target = k * base
                if target >= sr / 2 * 0.95:
                    break
                lo, hi = int(target * 0.97 / bin_hz), int(target * 1.03 / bin_hz) + 1
                if hi <= lo or hi >= len(freqs):
                    continue
                j = lo + int(np.argmax(S[lo:hi, t]))
                if S[j, t] > 0.2 * fund_amp:
                    inharm.append(abs(freqs[j] / target - 1.0))
        inharmonicity = float(np.median(inharm)) if len(inharm) >= 10 else 0.0
        pure_tone = harmonicity > 0.8 and len(inharm) < 10 and voiced.mean() > 0.3

        vib_rate, vib_depth, trem_rate, trem_depth, filter_mov, filter_slope = None, 0.0, None, 0.0, 0.0, 0.0
        long_segs = [(a, b) for a, b in segs if (b - a) / efps >= 0.5]
        if long_segs:
            ea, eb = max(long_segs, key=lambda ab: ab[1] - ab[0])
            a, b = int(ea * env_hop / hop), max(int(eb * env_hop / hop), int(ea * env_hop / hop) + 2)
            seg_f0 = f0[a:b]
            ok = ~np.isnan(seg_f0)
            if ok.sum() >= 8:
                semis = 12 * np.log2(seg_f0[ok] / np.median(seg_f0[ok]))
                tt = np.arange(b - a)[ok] / fps
                detr = semis - np.polyval(np.polyfit(tt, semis, 1), tt)
                rate, prom = _mod_peak(detr, fps, (3.0, 10.0))
                depth = float(np.sqrt(2) * np.std(detr) * 100)
                if rate is not None and prom > 4.0 and float(cfg.get("vibrato_min_cents", 8.0)) <= depth <= float(cfg.get("vibrato_max_cents", 100.0)) and ok.mean() > 0.5:
                    vib_rate, vib_depth = rate, depth
            e = env[ea:eb]
            dev = e / (e.mean() + EPS) - 1.0
            dev = dev - np.polyval(np.polyfit(np.arange(len(dev)), dev, 1), np.arange(len(dev)))
            rate, prom = _mod_peak(dev, efps, (2.0, 10.0))
            depth = float(np.sqrt(2) * np.std(dev))
            if rate is not None and prom > 4.0 and depth >= float(cfg.get("tremolo_min_depth", 0.05)):
                trem_rate, trem_depth = rate, depth
            c = cent[a:b]
            filter_mov = float(np.std(c) / (np.mean(c) + EPS))
            filter_slope = float(np.polyfit(np.arange(len(c)) / fps, c, 1)[0]) if len(c) > 2 else 0.0

        width = 0.0
        if x2.shape[1] >= 2 and rms(x2[:, 0]) > 1e-5 and rms(x2[:, 1]) > 1e-5:
            corr = float(np.corrcoef(x2[:, 0], x2[:, 1])[0, 1])
            width = float(np.clip(1.0 - corr, 0.0, 2.0)) if math.isfinite(corr) else 0.0

        features = {
            "attack_ms": round(attack_ms, 1),
            "decay_ms": round(decay_ms, 1),
            "sustain_level": round(sustain_level, 3),
            "release_ms": round(release_ms, 1),
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "spectral_flatness": round(flat, 4),
            "harmonicity": round(harmonicity, 4),
            "noise_ratio": round(noise_ratio, 4),
            "inharmonicity": round(inharmonicity, 4),
            "voiced_ratio": round(float(voiced.mean()), 3),
            "stereo_width": round(width, 3),
            "pitch_modulation": {"rate_hz": round(vib_rate, 2), "depth_cents": round(vib_depth, 1)} if vib_rate else None,
            "amplitude_modulation": {"rate_hz": round(trem_rate, 2), "depth": round(trem_depth, 3)} if trem_rate else None,
            "filter_movement": round(filter_mov, 4),
            "filter_slope_hz_per_s": round(filter_slope, 1),
            "segments": len(segs),
        }

        # ---- descriptors --------------------------------------------------------------------
        d: list[str] = []
        if centroid > float(cfg.get("bright_hz", 3000)):
            d.append("bright")
        elif centroid < float(cfg.get("dark_hz", 800)):
            d.append("dark")
        elif harmonicity > 0.6:
            d.append("warm")
        if noise_ratio > float(cfg.get("noise_ratio_threshold", 0.5)):
            d.append("noise_based")
        if attack_ms < 20:
            d.append("short_attack")
        elif attack_ms > 80:
            d.append("slow_attack")
        if release_ms > 500:
            d.append("long_release")
        elif 0 < release_ms < 150 and decay_ms < 300:
            d.append("short_decay")
        if inharmonicity > float(cfg.get("inharmonicity_threshold", 0.02)):
            d.append("bell_like" if release_ms > 400 and centroid > 1500 else "fm_like")
        if x2.shape[1] >= 2:
            if width > 0.4:
                d.append("wide_stereo")
            elif width < 0.05:
                d.append("mono")
        if "wide_stereo" in d and 0.15 < noise_ratio < 0.6 and centroid > 2000 and sustain_level > 0.5:
            d.append("supersaw_like")
        if filter_mov > float(cfg.get("filter_movement_threshold", 0.25)) and harmonicity > 0.5:
            d.append("subtractive")
        if attack_ms < 20 and decay_ms < 300 and sustain_level < 0.3:
            d.append("pluck")
        if pure_tone:
            d.append("pure_tone")
        if vib_rate:
            d.append("vibrato")
        detected = set(d)
        sims = []
        for name, tpl in PATCH_TEMPLATES.items():
            inter = len(detected & tpl)
            union = len(detected | tpl)
            sims.append((name, inter / union if union else 0.0))
        sims.sort(key=lambda kv: kv[1], reverse=True)
        top = [kv for kv in sims if kv[1] > 0][:3]
        if top:
            w = np.exp(np.array([s for _, s in top]) * 4.0)
            probs = w / w.sum()
            candidates = [Candidate(name, round(float(p), 3), round(sim, 3)) for (name, sim), p in zip(top, probs)]
            confidence = round(float(top[0][1]), 4)
        else:
            candidates, confidence = [], None
        # keep only taxonomy descriptors in sound_design; helper descriptors stay in features
        from mrep.domain.labels import SOUND_DESIGN

        features["descriptors"] = d
        sound_design = [x for x in d if x in SOUND_DESIGN]
        return SoundAnalysisResult(features=features, sound_design=sound_design, patch_candidates=candidates, confidence=confidence, confidence_method="descriptor_jaccard" if candidates else "unavailable")
