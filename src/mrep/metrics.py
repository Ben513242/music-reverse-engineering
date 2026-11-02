"""Quality metrics: separation (SDR / SI-SDR), transcription (P/R/F1), drums, beat/tempo."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

EPS = 1e-12


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()[: ref.size]
    ref = ref[: est.size]
    noise = ref - est
    return float(10 * np.log10((np.sum(ref**2) + EPS) / (np.sum(noise**2) + EPS)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()[: ref.size]
    ref = ref[: est.size]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))


def leakage_db(estimate: np.ndarray, other_sources: list[np.ndarray]) -> float:
    """Energy of the estimate explained by other sources (lower is better)."""
    est = np.asarray(estimate, dtype=np.float64).ravel()
    leak = 0.0
    for src in other_sources:
        s = np.asarray(src, dtype=np.float64).ravel()[: est.size]
        e = est[: s.size]
        alpha = np.dot(e, s) / (np.dot(s, s) + EPS)
        leak += np.sum((alpha * s) ** 2)
    return float(10 * np.log10((leak + EPS) / (np.sum(est**2) + EPS)))


def _greedy_match(ref: list[tuple], est: list[tuple], match_fn) -> int:
    used = set()
    hits = 0
    for r in ref:
        for i, e in enumerate(est):
            if i in used:
                continue
            if match_fn(r, e):
                used.add(i)
                hits += 1
                break
    return hits


def prf(hits: int, n_ref: int, n_est: int) -> dict[str, float]:
    p = hits / n_est if n_est else 0.0
    r = hits / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "hits": hits, "n_ref": n_ref, "n_est": n_est}


def note_metrics(ref_notes: list[dict[str, Any]], est_notes: list[dict[str, Any]], onset_tol: float = 0.05, offset_ratio: float | None = 0.2, pitch_tol: int = 0) -> dict[str, Any]:
    """Onset(+pitch) F1 and onset+offset F1 in the spirit of mir_eval.transcription."""
    ref = [(n["midi_note"], n["onset"], n["offset"]) for n in ref_notes]
    est = [(n["midi_note"], n["onset"], n["offset"]) for n in est_notes]

    def onset_match(r, e):
        return abs(r[0] - e[0]) <= pitch_tol and abs(r[1] - e[1]) <= onset_tol

    def full_match(r, e):
        if not onset_match(r, e):
            return False
        if offset_ratio is None:
            return True
        tol = max(0.05, offset_ratio * (r[2] - r[1]))
        return abs(r[2] - e[2]) <= tol

    pitch_hits = _greedy_match(ref, est, lambda r, e: abs(r[1] - e[1]) <= onset_tol and abs(r[0] - e[0]) <= pitch_tol)
    return {
        "onset_pitch": prf(_greedy_match(ref, est, onset_match), len(ref), len(est)),
        "onset_offset": prf(_greedy_match(ref, est, full_match), len(ref), len(est)),
        "onset_only": prf(_greedy_match(ref, est, lambda r, e: abs(r[1] - e[1]) <= onset_tol), len(ref), len(est)),
        "pitch_accuracy": round(pitch_hits / len(ref), 4) if ref else 0.0,
    }


def onset_f1(ref_times: list[float], est_times: list[float], tol: float = 0.05) -> dict[str, float]:
    ref = [(t,) for t in sorted(ref_times)]
    est = [(t,) for t in sorted(est_times)]
    hits = _greedy_match(ref, est, lambda r, e: abs(r[0] - e[0]) <= tol)
    return prf(hits, len(ref), len(est))


def drum_metrics(ref_events: list[dict[str, Any]], est_events: list[dict[str, Any]], tol: float = 0.05) -> dict[str, Any]:
    classes = sorted({e["class"] for e in ref_events} | {e["class"] for e in est_events})
    per_class = {}
    confusion: Counter = Counter()
    used = set()
    for r in sorted(ref_events, key=lambda e: e["time"]):
        best = None
        for i, e in enumerate(est_events):
            if i in used:
                continue
            if abs(e["time"] - r["time"]) <= tol and (best is None or e["class"] == r["class"]):
                best = i
                if e["class"] == r["class"]:
                    break
        if best is None:
            confusion[(r["class"], "miss")] += 1
        else:
            used.add(best)
            confusion[(r["class"], est_events[best]["class"])] += 1
    for i, e in enumerate(est_events):
        if i not in used:
            confusion[("false_alarm", e["class"])] += 1
    for c in classes:
        rc = [e for e in ref_events if e["class"] == c]
        ec = [e for e in est_events if e["class"] == c]
        per_class[c] = onset_f1([e["time"] for e in rc], [e["time"] for e in ec], tol)
    overall = onset_f1([e["time"] for e in ref_events], [e["time"] for e in est_events], tol)
    return {"onset_f1": overall, "per_class": per_class, "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())}}


def tempo_error(ref_bpm: float, est_bpm: float, octave_tolerant: bool = True) -> dict[str, Any]:
    candidates = [est_bpm] + ([est_bpm * 2, est_bpm / 2, est_bpm * 3, est_bpm / 3] if octave_tolerant else [])
    best = min(candidates, key=lambda c: abs(c - ref_bpm))
    return {"abs_error_bpm": round(abs(best - ref_bpm), 3), "rel_error": round(abs(best - ref_bpm) / ref_bpm, 4), "octave_equivalent": best != est_bpm, "within_4_percent": abs(best - ref_bpm) / ref_bpm <= 0.04}


def beat_f1(ref_beats: list[float], est_beats: list[float], tol: float = 0.07) -> dict[str, float]:
    return onset_f1(ref_beats, est_beats, tol)
