"""Baseline global music analysis (tempo, beats, downbeats, key, sections, loudness)."""

from __future__ import annotations

import string

import numpy as np

from mrep.adapters.base import MIRAdapter, MIRResult
from mrep.adapters.registry import register
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
        weight = np.zeros(T)
        for i, bf in enumerate(beat_frames):
            lo_i, hi_i = max(0, int(bf) - 2), min(T, int(bf) + 3)
            weight[bf] = float(low_env[lo_i:hi_i].max() + 0.5 * env[lo_i:hi_i].max())
            if 0 < i < len(beat_frames) - 1:  # harmonic change across the beat (chords change on downbeats)
                prev = chroma_db[:, int(beat_frames[i - 1]) : int(bf)].mean(axis=1)
                nxt = chroma_db[:, int(bf) : int(beat_frames[i + 1])].mean(axis=1)
                weight[bf] += 1.0 - float(np.dot(prev, nxt) / (np.linalg.norm(prev) * np.linalg.norm(nxt) + EPS))
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
        min_change = float(kcfg.get("min_change_confidence", 0.15))
        windows: list[tuple[float, float, int, str, float]] = []
        t = 0.0
        while t < duration:
            a, b = int(t * fps), int(min(duration, t + seg_s) * fps)
            if b - a > 2:
                act = active[a:b]
                cw = chroma[:, a:b][:, act].mean(axis=1) if act.any() else chroma[:, a:b].mean(axis=1)
                tonic, mode, conf, scores = estimate_key(cw)
                windows.append((t, min(duration, t + seg_s), tonic, mode, conf, scores))
            t += hop_s
        key_events: list[dict] = []
        cur = (g_tonic, g_mode)
        cur_start = 0.0
        cur_conf = [g_conf]
        from mrep.utils.music import key_fifths

        for ws, _we, tonic, mode, conf, scores in windows:
            if (tonic, mode) != cur and conf >= min_change:
                # relative major/minor share a key signature: require a clear preference
                if key_fifths(tonic, mode) == key_fifths(cur[0], cur[1]):
                    cur_name = pitch_class_name(cur[0]) + " " + cur[1]
                    new_name = pitch_class_name(tonic) + " " + mode
                    if scores.get(new_name, 0.0) - scores.get(cur_name, 0.0) < 0.1:
                        cur_conf.append(conf)
                        continue
                # accept a key change only if the next window agrees (or it is the last one)
                nxt = [w for w in windows if w[0] > ws]
                if not nxt or (nxt[0][2], nxt[0][3]) == (tonic, mode):
                    key_events.append({"start": cur_start, "end": ws, "tonic": pitch_class_name(cur[0]), "mode": cur[1], "confidence": float(np.mean(cur_conf)), "confidence_method": "ks_correlation_margin"})
                    cur, cur_start, cur_conf = (tonic, mode), ws, [conf]
                    continue
            if (tonic, mode) == cur:
                cur_conf.append(conf)
        key_events.append({"start": cur_start, "end": duration, "tonic": pitch_class_name(cur[0]), "mode": cur[1], "confidence": float(np.mean(cur_conf)), "confidence_method": "ks_correlation_margin"})

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
            time_signature=f"{bpb}/4",
            time_signature_confidence=None,
            time_signature_method="assumed",
            key_events=key_events,
            sections=sections,
            loudness=loudness,
            dynamics=dynamics,
            extra={"key_scores": dict(list(g_scores.items())[:6])},
        )
