"""Key segmentation: Viterbi smoothing over windowed key-profile scores.

Each analysis window yields a correlation score for all 24 keys. A hidden-Markov style path
with a switching penalty turns those noisy per-window winners into stable key segments and
suppresses one-window flips (e.g. between a key and its relative major/minor).
"""

from __future__ import annotations

import numpy as np

from mrep.utils.music import key_fifths, pitch_class_from_name


def _parse(key_name: str) -> tuple[int, str]:
    tonic, mode = key_name.rsplit(" ", 1)
    return pitch_class_from_name(tonic), mode


def viterbi_key_path(window_scores: list[dict[str, float]], switch_penalty: float = 0.25, relative_factor: float = 1.5) -> list[str]:
    """Return the most likely key name per window.

    ``switch_penalty`` is subtracted (in score units, where scores are correlations in [-1, 1]
    rescaled to [0, 1]) for every key change; changes between relative keys (same key
    signature) cost ``relative_factor`` times more because their pitch content is identical.
    """
    if not window_scores:
        return []
    keys = sorted(set().union(*window_scores))
    K, N = len(keys), len(window_scores)
    idx = {k: i for i, k in enumerate(keys)}
    emis = np.full((N, K), -1.0)
    for t, scores in enumerate(window_scores):
        for k, v in scores.items():
            emis[t, idx[k]] = (float(v) + 1.0) / 2.0
    fifths = np.array([key_fifths(*_parse(k)) for k in keys])
    penalty = np.full((K, K), switch_penalty)
    for i in range(K):
        for j in range(K):
            if i == j:
                penalty[i, j] = 0.0
            elif fifths[i] == fifths[j]:
                penalty[i, j] = switch_penalty * relative_factor
    dp = emis[0].copy()
    back = np.zeros((N, K), dtype=int)
    for t in range(1, N):
        cand = dp[:, None] - penalty  # (from, to)
        back[t] = np.argmax(cand, axis=0)
        dp = emis[t] + cand[back[t], np.arange(K)]
    path = [int(np.argmax(dp))]
    for t in range(N - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return [keys[i] for i in path]


def segments_from_path(path: list[str], bounds: list[tuple[float, float]], confidences: list[float] | None = None) -> list[dict]:
    """Merge consecutive windows with the same key into segments (start, end, key, confidence)."""
    segments: list[dict] = []
    for i, key in enumerate(path):
        start, end = bounds[i]
        conf = confidences[i] if confidences is not None else None
        if segments and segments[-1]["key"] == key:
            segments[-1]["end"] = end
            if conf is not None:
                segments[-1]["_confs"].append(conf)
            continue
        segments.append({"start": start, "end": end, "key": key, "_confs": [conf] if conf is not None else []})
    for seg in segments:
        confs = seg.pop("_confs")
        seg["confidence"] = float(np.mean(confs)) if confs else None
        tonic, mode = seg["key"].rsplit(" ", 1)
        seg["tonic"], seg["mode"] = tonic, mode
    return segments
