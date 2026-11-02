"""Monophonic f0 tracking (YIN-style) and per-note pitch-bend / vibrato estimation."""

from __future__ import annotations

import numpy as np

from mrep.domain.events import NoteEvent
from mrep.utils.dsp import EPS, frame_signal
from mrep.utils.music import hz_to_midi


def f0_track(x: np.ndarray, sr: int, frame: int = 2048, hop: int = 256, fmin: float = 40.0, fmax: float = 2000.0, threshold: float = 0.25, min_rms: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, f0_hz) with NaN where unvoiced. Cumulative-mean-normalised difference (YIN)."""
    x = np.asarray(x, dtype=np.float64)
    frames = frame_signal(x, frame, hop, center=True).astype(np.float64)
    T = frames.shape[0]
    times = np.arange(T) * hop / sr
    f0 = np.full(T, np.nan)
    if T == 0:
        return times, f0
    rms = np.sqrt(np.mean(frames**2, axis=1))
    tau_min = max(2, int(sr / fmax))
    tau_max = min(frame // 2, int(sr / fmin))
    if tau_max <= tau_min:
        return times, f0
    win = np.hanning(frame)
    fw = frames * win
    nfft = 1 << (2 * frame - 1).bit_length()
    F = np.fft.rfft(fw, nfft, axis=1)
    acf = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, : frame // 2 + 1]
    r0 = acf[:, :1]
    diff = 2.0 * (r0 - acf)  # difference function approximation
    diff[:, 0] = 0.0
    taus = np.arange(1, frame // 2 + 1)
    cum = np.cumsum(diff[:, 1:], axis=1) / taus[None, :]
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] / (cum + EPS)
    for t in range(T):
        if rms[t] < min_rms:
            continue
        seg = cmnd[t, tau_min : tau_max + 1]
        below = np.where(seg < threshold)[0]
        if below.size:
            k = below[0]
            # walk to the local minimum after the first crossing
            while k + 1 < seg.size and seg[k + 1] < seg[k]:
                k += 1
        else:
            k = int(np.argmin(seg))
            if seg[k] > 0.6:
                continue
        tau = tau_min + k
        if 1 <= tau < cmnd.shape[1] - 1:
            a, b, c = cmnd[t, tau - 1], cmnd[t, tau], cmnd[t, tau + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                tau = tau + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        f0[t] = sr / tau
    return times, f0


def estimate_note_bends(notes: list[NoteEvent], times: np.ndarray, f0_hz: np.ndarray, max_dev_semitones: float = 1.0, bend_threshold_cents: float = 15.0, min_points: int = 4, max_points: int = 32, vibrato_min_depth_cents: float = 8.0, vibrato_rate_range: tuple[float, float] = (3.0, 10.0)) -> int:
    """Attach ``pitch_bend`` ([[t_rel, semitones], ...]) and ``features['vibrato']`` to notes.

    Returns the number of notes that received a bend curve. Only frames whose f0 is within
    ``max_dev_semitones`` of the note are used, so neighbouring notes do not pollute the curve.
    """
    if times.size == 0:
        return 0
    midi_track = np.where(np.isnan(f0_hz), np.nan, np.array([hz_to_midi(v) if v and v > 0 else np.nan for v in np.nan_to_num(f0_hz, nan=0.0)]))
    count = 0
    for n in notes:
        a = int(np.searchsorted(times, n.onset))
        b = int(np.searchsorted(times, n.offset))
        if b - a < min_points:
            continue
        seg = midi_track[a:b]
        t = times[a:b] - n.onset
        ok = ~np.isnan(seg) & (np.abs(seg - n.midi_note) <= max_dev_semitones)
        if ok.sum() < min_points:
            continue
        dev = seg[ok] - n.midi_note
        tt = t[ok]
        if np.max(np.abs(dev)) * 100.0 >= bend_threshold_cents:
            step = max(1, len(dev) // max_points)
            n.pitch_bend = [[round(float(x), 4), round(float(d), 3)] for x, d in zip(tt[::step], dev[::step])]
            count += 1
        if ok.sum() >= 8 and (tt[-1] - tt[0]) > 0.25:
            dt = float(np.median(np.diff(tt)))
            if dt > 0:
                detr = dev - np.polyval(np.polyfit(tt, dev, 1), tt)
                spec = np.abs(np.fft.rfft(detr * np.hanning(len(detr))))
                freqs = np.fft.rfftfreq(len(detr), dt)
                band = (freqs >= vibrato_rate_range[0]) & (freqs <= vibrato_rate_range[1])
                if band.any() and spec.sum() > 0:
                    k = int(np.argmax(np.where(band, spec, 0.0)))
                    depth_cents = float(np.sqrt(2) * np.std(detr) * 100.0)
                    if spec[k] >= 0.4 * spec[1:].sum() / max(1, band.sum()) * band.sum() * 0.5 and depth_cents >= vibrato_min_depth_cents:
                        n.features["vibrato"] = {"rate_hz": round(float(freqs[k]), 2), "depth_cents": round(depth_cents, 1)}
    return count
