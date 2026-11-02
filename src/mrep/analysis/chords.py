"""Chord estimation from a blend of audio chroma and transcribed notes (plus bass for slash chords)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mrep.domain.events import ChordEvent, NoteEvent
from mrep.domain.provenance import Provenance
from mrep.utils.music import CHORD_TEMPLATES, chord_symbol, pitch_class_name

EPS = 1e-9


def chroma_from_notes(notes: list[NoteEvent], start: float, end: float) -> np.ndarray:
    v = np.zeros(12)
    for n in notes:
        ov = min(n.offset, end) - max(n.onset, start)
        if ov > 0:
            v[n.midi_note % 12] += ov * (n.velocity / 127.0) * (n.confidence if n.confidence is not None else 0.5)
    return v


def build_templates(qualities: list[str]) -> list[tuple[int, str, np.ndarray]]:
    out = []
    for q in qualities:
        if q not in CHORD_TEMPLATES:
            continue
        for root in range(12):
            t = np.zeros(12)
            for i, interval in enumerate(CHORD_TEMPLATES[q]):
                t[(root + interval) % 12] = 1.0 if i == 0 else 0.85
            out.append((root, q, t / np.linalg.norm(t)))
    return out


def lowest_note_pc(notes: list[NoteEvent], start: float, end: float) -> int | None:
    """Pitch class of the most prominent bass note in the window (overlap x velocity x confidence)."""
    best, best_w = None, 0.0
    for n in notes:
        ov = min(n.offset, end) - max(n.onset, start)
        if ov <= 0:
            continue
        w = ov * (n.velocity / 127.0) * (n.confidence if n.confidence is not None else 0.5)
        if w > best_w:
            best, best_w = n, w
    return None if best is None else best.midi_note % 12


def detect_chords(
    windows: list[tuple[float, float]],
    notes: list[NoteEvent],
    bass_notes: list[NoteEvent],
    audio_chroma: Callable[[float, float], np.ndarray | None] | None,
    qualities: list[str],
    min_confidence: float,
    provenance: Provenance | None,
    audio_weight: float = 0.6,
) -> list[ChordEvent]:
    templates = build_templates(qualities)
    raw: list[tuple[float, float, int, str, int | None, float]] = []
    for start, end in windows:
        v = np.zeros(12)
        vn = chroma_from_notes(notes, start, end)
        if vn.sum() > 0:
            v += (1 - audio_weight) * vn / (np.linalg.norm(vn) + EPS)
        if audio_chroma is not None:
            va = audio_chroma(start, end)
            if va is not None and np.sum(va) > 0:
                v += audio_weight * va / (np.linalg.norm(va) + EPS)
        if v.sum() <= 0:
            continue
        v = v / (np.linalg.norm(v) + EPS)
        scores = []
        for root, q, t in templates:
            sim = float(np.dot(v, t)) - 0.02 * max(0, len(CHORD_TEMPLATES[q]) - 3)
            scores.append((sim, root, q))
        scores.sort(reverse=True)
        best, root, q = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        conf = float(np.clip(best * (0.5 + 0.5 * np.clip((best - second) / 0.1, 0.0, 1.0)), 0.0, 1.0))
        bass_pc = lowest_note_pc(bass_notes, start, end)
        raw.append((start, end, root, q, bass_pc, conf))
    # merge consecutive identical chords
    events: list[ChordEvent] = []
    for start, end, root, q, bass_pc, conf in raw:
        if events and events[-1].root == pitch_class_name(root) and events[-1].quality == q and abs(events[-1].end - start) < 1e-6:
            prev = events[-1]
            prev.end = end
            prev.confidence = round((prev.confidence or 0) * 0.5 + conf * 0.5, 4)
            continue
        pcs = [(root + i) % 12 for i in CHORD_TEMPLATES[q]]
        bass_name = pitch_class_name(bass_pc) if bass_pc is not None else pitch_class_name(root)
        inversion = pcs.index(bass_pc) if bass_pc in pcs else 0
        events.append(ChordEvent(start=round(start, 4), end=round(end, 4), root=pitch_class_name(root), quality=q, bass=bass_name, symbol=chord_symbol(pitch_class_name(root), q, bass_name), inversion=inversion, pitch_classes=pcs, confidence=round(conf, 4), confidence_method="template_cosine_margin", provenance=provenance))
    return [e for e in events if (e.confidence or 0) >= min_confidence]
