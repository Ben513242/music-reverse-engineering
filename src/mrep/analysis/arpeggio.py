"""Arpeggio detection on a monophonic-ish note sequence of one track."""

from __future__ import annotations

from typing import Any

import numpy as np

from mrep.domain.events import NoteEvent
from mrep.utils.music import CHORD_TEMPLATES, key_name, scale_degrees

_RHYTHM = {1.0: "1/4", 0.5: "1/8", 0.25: "1/16", 1 / 3: "triplet_8", 1 / 6: "triplet_16", 2.0: "1/2"}


def _rhythm_name(ioi_beats: float) -> str:
    best = min(_RHYTHM, key=lambda k: abs(k - ioi_beats))
    return _RHYTHM[best] if abs(best - ioi_beats) / best < 0.25 else f"{ioi_beats:.2f}_beats"


def _fit_chord(pcs: set[int]) -> tuple[int, str, float] | None:
    best = None
    for q in ("major", "minor", "dom7", "maj7", "min7", "sus2", "sus4", "dim", "aug"):
        for root in range(12):
            tpl = {(root + i) % 12 for i in CHORD_TEMPLATES[q]}
            if pcs <= tpl:
                score = len(pcs) / len(tpl) + (0.05 if root in pcs else 0.0)
                if best is None or score > best[2]:
                    best = (root, q, score)
    return best


def detect_arpeggios(notes: list[NoteEvent], beat_seconds: float, min_notes: int = 4, ioi_cv_max: float = 0.3, max_note_beats: float = 1.0, min_confidence: float = 0.25) -> list[dict[str, Any]]:
    """Find regular, chord-tone note sequences. Sustained notes (pads) and low-confidence
    notes are ignored so that an arpeggio played over a held chord is still found."""
    seq = sorted(
        (n for n in notes if n.duration <= max_note_beats * beat_seconds and (n.confidence is None or n.confidence >= min_confidence)),
        key=lambda n: (n.onset, n.midi_note),
    )
    runs: list[list[NoteEvent]] = []
    cur: list[NoteEvent] = []
    for n in seq:
        if cur and n.onset - cur[-1].onset < 0.03:  # simultaneous = chord, not arpeggio
            if len(cur) >= min_notes:
                runs.append(cur)
            cur = []
            continue
        if len(cur) >= 2:
            iois = np.diff([c.onset for c in cur] + [n.onset])
            med = float(np.median(iois[:-1]))
            if med > 0 and abs(iois[-1] - med) / med > ioi_cv_max * 2:
                if len(cur) >= min_notes:
                    runs.append(cur)
                cur = [cur[-1]]
        cur.append(n)
    if len(cur) >= min_notes:
        runs.append(cur)

    out: list[dict[str, Any]] = []
    for run in runs:
        pcs = {n.midi_note % 12 for n in run}
        if not 2 <= len(pcs) <= 4:
            continue
        iois = np.diff([n.onset for n in run])
        cv = float(np.std(iois) / (np.mean(iois) + 1e-9))
        if cv > ioi_cv_max:
            continue
        fit = _fit_chord(pcs)
        if fit is None:
            continue
        root, q, score = fit
        degrees = scale_degrees([n.midi_note % 12 for n in run], root, q)
        pattern = degrees
        for period in range(2, min(8, len(degrees)) + 1):
            if all(degrees[i] == degrees[i % period] for i in range(len(degrees))):
                pattern = degrees[:period]
                break
        ioi_beats = float(np.mean(iois)) / beat_seconds if beat_seconds > 0 else float(np.mean(iois))
        conf = float(np.clip((1.0 - cv / ioi_cv_max) * 0.6 + score * 0.4, 0.0, 1.0))
        out.append(
            {
                "start": round(run[0].onset, 4),
                "end": round(run[-1].offset, 4),
                "track": run[0].track,
                "chord": key_name(root, "major" if q in ("major", "dom7", "maj7", "sus2", "sus4", "aug") else "minor").split(" ")[0] + ("" if q == "major" else f" {q}"),
                "chord_root": root,
                "chord_quality": q,
                "pattern": "-".join(str(d) for d in pattern),
                "rhythm": _rhythm_name(ioi_beats),
                "ioi_beats": round(ioi_beats, 4),
                "note_count": len(run),
                "confidence": round(conf, 4),
                "confidence_method": "ioi_regularity_chord_fit",
            }
        )
    return out
