"""Baseline polyphonic AMT: iterative harmonic-summation pitch salience + note segmentation.

Not a neural transcriber - it is the CPU-only fallback that keeps the pipeline runnable.
Better AMT adapters register under the same ``transcription`` kind and are selected in config.
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import TranscriptionAdapter
from mrep.adapters.registry import register
from mrep.domain.events import NoteEvent
from mrep.domain.provenance import Provenance
from mrep.utils.dsp import EPS, frame_times, stft, to_mono
from mrep.utils.music import amplitude_to_velocity, midi_to_hz


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _split_reattacks(curve: np.ndarray, drop: float = 0.45, rise: float = 1.6) -> list[int]:
    """Indices where a new attack starts inside a sustained run (dip then rise)."""
    splits: list[int] = []
    peak = curve[0]
    trough = curve[0]
    dipped = False
    for i in range(1, len(curve)):
        v = curve[i]
        if v > peak:
            peak = v
        if v < trough:
            trough = v
        if not dipped and v < drop * peak:
            dipped = True
            trough = v
        elif dipped and v > rise * trough and v > 0.5 * peak:
            splits.append(i)
            peak, trough, dipped = v, v, False
    return splits


@register("transcription", "spectral_peaks")
class SpectralPeaksAMT(TranscriptionAdapter):
    model = "harmonic_summation"
    model_version = "1.0"

    def transcribe(self, audio: np.ndarray, sr: int, *, track: str, instrument: str, provenance: Provenance) -> list[NoteEvent]:
        x = to_mono(audio)
        if x.size == 0 or float(np.max(np.abs(x))) < 1e-5:
            return []
        cfg = self.config
        lo, hi = (int(v) for v in cfg.get("midi_range", [28, 96]))
        is_bass = "bass" in (instrument or "").lower() or "bass" in (track or "").lower()
        n_fft = int(cfg.get("n_fft", 8192 if is_bass else 4096))
        hop = int(cfg.get("hop", 1024))
        suppress = int(cfg.get("neighbor_suppression_semitones", 1))
        K = int(cfg.get("polyphony", 4))
        thr = float(cfg.get("salience_threshold", 0.08))
        min_dur = float(cfg.get("min_note_seconds", 0.06))
        n_harm = int(cfg.get("harmonics", 6))
        S = np.abs(stft(x, n_fft, hop)) ** 0.7
        F, T = S.shape
        bin_hz = sr / n_fft
        midis = np.arange(lo, hi + 1)
        M = len(midis)
        f0 = midi_to_hz(midis.astype(np.float64))
        weights = np.array([0.8**h for h in range(n_harm)])
        idx = np.zeros((n_harm, M), dtype=int)
        valid = np.zeros((n_harm, M), dtype=bool)
        for h in range(n_harm):
            fh = (h + 1) * f0
            idx[h] = np.clip(np.round(fh / bin_hz).astype(int), 0, F - 1)
            valid[h] = fh < sr / 2
        # semitone spacing in bins is < 1 for low notes: widen the suppression there so that the
        # window main lobe of a note cannot masquerade as its neighbours
        semitone_bins = f0 * (2 ** (1 / 12) - 1) / bin_hz
        n_sup = np.clip(np.ceil(2.0 / np.maximum(semitone_bins, 1e-6)), suppress, 6).astype(int)
        max_sup = int(n_sup.max())
        work = S.copy()
        act = np.zeros((M, T), dtype=np.float32)
        blocked = np.zeros((M, T), dtype=bool)
        frames = np.arange(T)
        ref = None
        for _ in range(K):
            sal = np.zeros((M, T))
            for h in range(n_harm):
                sal += weights[h] * (work[idx[h]] * valid[h][:, None])
            fund = work[idx[0]]
            mean_harm = sal / weights.sum()
            sal = np.where((fund >= 0.25 * mean_harm) & ~blocked, sal, 0.0)
            if ref is None:
                ref = float(np.percentile(sal.max(axis=0), 98)) + EPS
            saln = sal / ref
            best_m = np.argmax(saln, axis=0)
            best_v = saln[best_m, frames]
            keep = best_v >= thr
            if not keep.any():
                break
            act[best_m[keep], frames[keep]] = np.maximum(act[best_m[keep], frames[keep]], best_v[keep])
            for d in range(-max_sup, max_sup + 1):
                sel = keep & (np.abs(d) <= n_sup[best_m])
                rows_m = np.clip(best_m[sel] + d, 0, M - 1)
                blocked[rows_m, frames[sel]] = True
            for h in range(n_harm):
                rows = idx[h][best_m[keep]]
                work[rows, frames[keep]] *= 1.0 - 0.9 * weights[h]
        times = frame_times(T + 1, sr, hop)
        min_frames = max(1, int(round(min_dur * sr / hop)))
        notes: list[NoteEvent] = []
        peaks: list[float] = []
        raw: list[tuple[int, int, int, float, float]] = []
        for mi in range(M):
            row = act[mi]
            mask = row > 0
            filled = mask.copy()
            filled[1:-1] |= mask[:-2] & mask[2:]
            for a, b in _runs(filled):
                seg = row[a : b + 1]
                cuts = [0] + _split_reattacks(seg) + [len(seg)]
                for c0, c1 in zip(cuts[:-1], cuts[1:]):
                    if c1 - c0 < min_frames:
                        continue
                    sub = seg[c0:c1]
                    pk = float(sub.max())
                    peaks.append(pk)
                    raw.append((int(midis[mi]), a + c0, a + c1, pk, float(sub.mean())))
        if not raw:
            return []
        vref = float(np.percentile(peaks, 95)) + EPS
        for midi, fa, fb, pk, mean_sal in raw:
            notes.append(
                NoteEvent(
                    midi_note=midi,
                    onset=round(float(times[fa]), 4),
                    offset=round(float(times[fb]), 4),
                    velocity=amplitude_to_velocity(pk / vref),
                    track=track,
                    instrument=instrument,
                    confidence=round(float(np.clip(mean_sal, 0.0, 1.0)), 4),
                    confidence_method="normalized_salience",
                    provenance=provenance,
                )
            )
        notes.sort(key=lambda n: (n.onset, n.midi_note))
        return notes
