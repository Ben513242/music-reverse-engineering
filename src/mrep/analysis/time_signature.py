"""Time-signature (beats per bar) estimation from beat-level accent strengths."""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def estimate_time_signature(accents: np.ndarray, candidates: tuple[int, ...] = (3, 4), min_bars: int = 4) -> tuple[int, int, float | None, str, dict[str, float]]:
    """Return (beats_per_bar, downbeat_phase, confidence, method, scores).

    For every candidate bar length L the accents are folded modulo L. A real metre shows one
    phase whose mean accent stands out (contrast) and a strong autocorrelation at lag L.
    Falls back to 4/4 with ``confidence=None`` / ``method="assumed"`` when there is too little
    data to decide.
    """
    a = np.asarray(accents, dtype=np.float64)
    if a.size == 0:
        return 4, 0, None, "assumed", {}
    a = a - a.mean()
    std = a.std() + EPS
    acf = np.correlate(a, a, mode="full")[a.size - 1 :]
    acf = acf / (acf[0] + EPS)
    scores: dict[str, float] = {}
    phases: dict[int, int] = {}
    for L in candidates:
        if a.size < L * min_bars:
            continue
        means = np.array([a[p::L].mean() for p in range(L)])
        contrast = float((means.max() - means.mean()) / std)
        periodicity = float(acf[L]) if L < acf.size else 0.0
        # a bar of 4 also correlates at lag 2 (half-bar); reward the *extra* periodicity at L
        sub = float(acf[L // 2]) if L % 2 == 0 and L // 2 < acf.size else 0.0
        score = 0.6 * contrast + 0.4 * (periodicity - 0.5 * max(sub, 0.0))
        scores[f"{L}/4"] = round(score, 4)
        phases[L] = int(np.argmax(means))
    if not scores:
        return 4, 0, None, "assumed", {}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else best - 1.0
    L = int(best_name.split("/")[0])
    margin = best - second
    confidence = float(np.clip(margin / (abs(best) + 0.5), 0.0, 1.0))
    return L, phases[L], round(confidence, 4), "accent_periodicity", scores
