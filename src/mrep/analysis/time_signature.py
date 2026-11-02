"""Time-signature (beats per bar) estimation from beat-level accent strengths."""

from __future__ import annotations

import numpy as np

EPS = 1e-9


METRE_NAMES = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8", 7: "7/8"}


def metre_name(beats_per_bar: int) -> str:
    """Name a bar length in tracked beats (6 tracked beats per bar are reported as 6/8)."""
    return METRE_NAMES.get(int(beats_per_bar), f"{int(beats_per_bar)}/4")


def estimate_time_signature(accents: np.ndarray, candidates: tuple[int, ...] = (3, 4, 6), min_bars: int = 4) -> tuple[int, int, float | None, str, dict[str, float]]:
    """Return (beats_per_bar, downbeat_phase, confidence, method, scores).

    For every candidate bar length L the accents are folded modulo L. A real metre shows one
    phase whose mean accent stands out (contrast) and a strong autocorrelation at lag L; the
    autocorrelation at L's sub-periods (L/2, L/3) is penalised so that 4 does not win merely
    because of a half-bar pulse, and 6 only wins over 3 when the two half-bars really differ.
    Falls back to 4/4 with ``confidence=None`` / ``method="assumed"`` when there is too little
    data to decide.
    """
    a = np.asarray(accents, dtype=np.float64)
    if a.size == 0:
        return 4, 0, None, "assumed", {}
    a = a - a.mean()
    acf = np.correlate(a, a, mode="full")[a.size - 1 :]
    acf = acf / (acf[0] + EPS)
    scores: dict[str, float] = {}
    phases: dict[int, int] = {}
    N = a.size
    total_ss = float(np.sum(a**2)) + EPS
    for L in candidates:
        if N < L * min_bars:
            continue
        means = np.array([a[p::L].mean() for p in range(L)])
        counts = np.array([a[p::L].size for p in range(L)])
        # fraction of variance explained by the L-phase fold, adjusted for the extra degrees of
        # freedom a longer bar enjoys (otherwise the longest candidate always looks best)
        explained = float(np.sum(counts * means**2) / total_ss)
        adjusted = 1.0 - (1.0 - explained) * (N - 1) / max(1, N - L)
        periodicity = float(acf[L]) if L < acf.size else 0.0
        subs = [float(acf[L // d]) for d in (2, 3) if L % d == 0 and L // d >= 2 and L // d < acf.size]
        sub = max(subs) if subs else 0.0
        score = 0.7 * adjusted + 0.3 * (periodicity - 0.35 * max(sub, 0.0))
        scores[metre_name(L)] = round(score, 4)
        phases[L] = int(np.argmax(means))
    if not scores:
        return 4, 0, None, "assumed", {}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else best - 1.0
    L = next(k for k, v in METRE_NAMES.items() if v == best_name) if best_name in METRE_NAMES.values() else int(best_name.split("/")[0])
    margin = best - second
    confidence = float(np.clip(margin / (abs(best) + 0.5), 0.0, 1.0))
    return L, phases[L], round(confidence, 4), "accent_periodicity", scores
