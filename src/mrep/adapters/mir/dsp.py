"""Baseline global music analysis (tempo, beats, downbeats, key, sections, loudness)."""

from __future__ import annotations

import string

import numpy as np
from scipy.ndimage import median_filter

from mrep.adapters.base import MIRAdapter, MIRResult
from mrep.adapters.registry import register
from mrep.analysis.key_segments import segments_from_path, viterbi_key_path
from mrep.analysis.time_signature import estimate_time_signature, metre_name
from mrep.utils.dsp import (
    EPS,
    choose_downbeat_phase,
    chroma_from_stft,
    db,
    estimate_key,
    estimate_tempo,
    fft_freqs,
    novelty_curve,
    onset_strength,
    rms,
    rms_frames,
    segment_boundaries,
    spectral_centroid,
    stft,
    to_mono,
    track_beats,
)
from mrep.utils.music import loudness_to_dynamic, pitch_class_name


@register("mir", "dsp")
class DSPMIR(MIRAdapter):
    model = "dsp_mir"
    model_version = "1.0"

    def analyze(self, audio: np.ndarray, sr: int) -> MIRResult:
        cfg = self.config
        x = to_mono(audio)
        duration = len(x) / sr
        n_fft, hop = 2048, 512
        fps = sr / hop
        S = np.abs(stft(x, n_fft, hop))
        freqs = fft_freqs(sr, n_fft)
        T = S.shape[1]
        frame_e = S.sum(axis=0) + EPS

        # ---- tempo / beats ------------------------------------------------------------
        tcfg = cfg.get("tempo", {})
        env = onset_strength(S, smooth=2, freqs=freqs, n_bands=64)
        bpm, tconf = estimate_tempo(env, fps, float(tcfg.get("min_bpm", 60)), float(tcfg.get("max_bpm", 200)), float(tcfg.get("prior_bpm", 120)))
        beat_frames = track_beats(env, fps, bpm)
        beat_times = (beat_frames * hop / sr).tolist()
        if len(beat_frames) >= 4:
            med = float(np.median(np.diff(beat_frames))) / fps
            refined = 60.0 / med if med > 0 else bpm
            if abs(refined - bpm) / bpm < 0.12:
                bpm = refined
        bpb = int(cfg.get("beats", {}).get("beats_per_bar", 4))
        low_env = onset_strength(S[freqs < 150.0]) if T > 1 else env
        chroma_db = chroma_from_stft(np.sqrt(S), freqs, fmin=30.0)
        chroma_bass = chroma_from_stft(np.sqrt(S), freqs, fmin=30.0, fmax=200.0)  # bass-note changes mark bars
        weight = np.zeros(T)

        def _change(c: np.ndarray, a: int, b: int, c2: int) -> float:
            prev = c[:, a:b].mean(axis=1)
            nxt = c[:, b:c2].mean(axis=1)
            return 1.0 - float(np.dot(prev, nxt) / (np.linalg.norm(prev) * np.linalg.norm(nxt) + EPS))

        ctx_beats = 2  # compare the two beats before with the two beats after: bar-level changes stand out
        for i, bf in enumerate(beat_frames):
            lo_i, hi_i = max(0, int(bf) - 2), min(T, int(bf) + 3)
            weight[bf] = float(low_env[lo_i:hi_i].max() + 0.5 * env[lo_i:hi_i].max())
            if ctx_beats <= i < len(beat_frames) - ctx_beats:  # harmonic / bass-note change (chords change on downbeats)
                a, b, c2 = int(beat_frames[i - ctx_beats]), int(bf), int(beat_frames[i + ctx_beats])
                weight[bf] += _change(chroma_db, a, b, c2) + 1.5 * _change(chroma_bass, a, b, c2)
        ts_cfg = cfg.get("time_signature", {}) or {}
        ts_conf: float | None = None
        ts_method = "assumed"
        ts_scores: dict[str, float] = {}
        if str(ts_cfg.get("mode", "auto")) == "auto" and len(beat_frames) >= 8:
            accents = weight[beat_frames]
            L, phase, ts_conf, ts_method, ts_scores = estimate_time_signature(accents, tuple(int(c) for c in ts_cfg.get("candidates", [3, 4])))
            if ts_conf is None:
                phase = choose_downbeat_phase(beat_frames, weight, bpb)
            else:
                bpb = L
        else:
            ts_method = "configured" if str(ts_cfg.get("mode", "auto")) != "auto" else "assumed"
            phase = choose_downbeat_phase(beat_frames, weight, bpb) if len(beat_frames) else 0
        downbeats = beat_times[phase::bpb]

        # ---- key ----------------------------------------------------------------------
        kcfg = cfg.get("key", {})
        chroma = chroma_from_stft(np.sqrt(S), freqs)  # sqrt compression: loud bass does not dominate
        active = frame_e > frame_e.max() * 0.02
        weighted = chroma[:, active].mean(axis=1) if active.any() else chroma.mean(axis=1)
        g_tonic, g_mode, g_conf, g_scores = estimate_key(weighted)
        seg_s = float(kcfg.get("segment_seconds", 20.0))
        hop_s = float(kcfg.get("hop_seconds", 10.0))
        switch_penalty = float(kcfg.get("switch_penalty", 0.25))
        window_scores: list[dict[str, float]] = []
        window_conf: list[float] = []
        bounds: list[tuple[float, float]] = []
        t = 0.0
        while t < duration:
            a, b = int(t * fps), int(min(duration, t + seg_s) * fps)
            if b - a > 2:
                act = active[a:b]
                cw = chroma[:, a:b][:, act].mean(axis=1) if act.any() else chroma[:, a:b].mean(axis=1)
                _tonic, _mode, conf, scores = estimate_key(cw)
                window_scores.append(scores)
                window_conf.append(conf)
                # windows overlap: a window "owns" the span from its start to the next start
                bounds.append((t, min(duration, t + hop_s)))
            t += hop_s
        if bounds:
            bounds[-1] = (bounds[-1][0], duration)
        if window_scores:
            path = viterbi_key_path(window_scores, switch_penalty=switch_penalty)
            key_events = [
                {"start": seg["start"], "end": seg["end"], "tonic": seg["tonic"], "mode": seg["mode"], "confidence": seg["confidence"], "confidence_method": "ks_correlation_margin_viterbi"}
                for seg in segments_from_path(path, bounds, window_conf)
            ]
        else:
            key_events = [{"start": 0.0, "end": duration, "tonic": pitch_class_name(g_tonic), "mode": g_mode, "confidence": g_conf, "confidence_method": "ks_correlation_margin"}]

        # ---- tempo curve (variable tempo) -------------------------------------------------------
        tempo_curve: list[dict] = []
        tempo_stability = None
        if len(beat_times) >= 9:
            local = 60.0 / np.maximum(np.diff(np.asarray(beat_times)), 1e-3)
            smooth = median_filter(local, size=8, mode="nearest")
            tempo_stability = round(float(np.std(smooth) / (np.mean(smooth) + EPS)), 4)
            last = None
            for i, b in enumerate(smooth):
                if last is None or abs(b - last) >= float(tcfg.get("curve_min_change_bpm", 2.0)):
                    tempo_curve.append({"time": round(float(beat_times[i]), 4), "bpm": round(float(b), 2)})
                    last = b

        # ---- sections -----------------------------------------------------------------
        scfg = cfg.get("sections", {})
        sections: list[dict] = []
        if len(beat_frames) >= 8:
            cent = spectral_centroid(S, freqs)
            feats = []
            for i in range(len(beat_frames) - 1):
                a, b = int(beat_frames[i]), max(int(beat_frames[i + 1]), int(beat_frames[i]) + 1)
                feats.append(np.concatenate([chroma[:, a:b].mean(axis=1), [frame_e[a:b].mean() / frame_e.max(), cent[a:b].mean() / (sr / 2)]]))
            feats_arr = np.asarray(feats)
            nov = novelty_curve(feats_arr, kernel=int(scfg.get("kernel_beats", 8)))
            beat_len = 60.0 / bpm
            min_beats = max(2, int(float(scfg.get("min_section_seconds", 8.0)) / beat_len))
            bounds = segment_boundaries(nov, min_distance=min_beats)
            edges = [0] + [int(b) for b in bounds if 0 < b < len(feats_arr)] + [len(feats_arr)]
            protos: list[np.ndarray] = []
            labels: list[str] = []
            letters = string.ascii_uppercase
            for s, e in zip(edges[:-1], edges[1:]):
                if e <= s:
                    continue
                proto = feats_arr[s:e].mean(axis=0)
                lab = None
                for pi, p in enumerate(protos):
                    sim = float(np.dot(proto, p) / (np.linalg.norm(proto) * np.linalg.norm(p) + EPS))
                    if sim > 0.95:
                        lab = labels[pi]
                        break
                if lab is None:
                    lab = letters[len(protos) % 26]
                    protos.append(proto)
                    labels.append(lab)
                conf = float(nov[s]) if 0 < s < len(nov) else 0.5
                sections.append({"start": float(beat_frames[s] * hop / sr), "end": float(beat_frames[e] * hop / sr) if e < len(beat_frames) else duration, "label": lab, "confidence": round(min(max(conf, 0.1), 1.0), 4), "confidence_method": "novelty_peak"})
        if not sections:
            sections.append({"start": 0.0, "end": duration, "label": "A", "confidence": None, "confidence_method": "unavailable"})

        # ---- loudness / dynamics --------------------------------------------------------
        env_rms = rms_frames(x, n_fft, hop)
        loudness = {
            "integrated_rms_db": round(db(rms(x)), 2),
            "peak_db": round(db(float(np.max(np.abs(x)))), 2),
            "short_term_rms_db": [round(db(float(v)), 2) for v in env_rms[:: max(1, int(3 * fps))]],
            "method": "rms_dbfs",
        }
        dynamics = []
        for sec in sections:
            a, b = int(sec["start"] * fps), max(int(sec["end"] * fps), int(sec["start"] * fps) + 1)
            level = db(float(np.sqrt(np.mean(env_rms[a:b] ** 2)))) if b > a else loudness["integrated_rms_db"]
            dynamics.append({"start": sec["start"], "end": sec["end"], "rms_db": round(level, 2), "dynamic": loudness_to_dynamic(level)})

        return MIRResult(
            tempo_bpm=round(float(bpm), 3),
            tempo_confidence=round(float(tconf), 4),
            tempo_method="autocorrelation_peak",
            beat_times=[round(b, 4) for b in beat_times],
            downbeat_times=[round(b, 4) for b in downbeats],
            beats_per_bar=bpb,
            time_signature=metre_name(bpb) if ts_method == "accent_periodicity" else f"{bpb}/4",
            time_signature_confidence=ts_conf,
            time_signature_method=ts_method,
            key_events=key_events,
            sections=sections,
            loudness=loudness,
            dynamics=dynamics,
            extra={"key_scores": dict(list(g_scores.items())[:6]), "time_signature_scores": ts_scores, "tempo_curve": tempo_curve, "tempo_stability": tempo_stability, "beat_accents": [round(float(v), 4) for v in weight[beat_frames]] if len(beat_frames) else []},
        )
