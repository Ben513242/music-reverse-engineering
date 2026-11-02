"""Event-fusion rules: consensus between adapters and between stem / full-mix AMT paths."""

from __future__ import annotations

from mrep.domain.events import DrumEvent, NoteEvent


def combine_confidence(values: list[float | None]) -> tuple[float | None, str]:
    known = [v for v in values if v is not None]
    if not known:
        return None, "unavailable"
    if len(known) == 1:
        return known[0], "single_source"
    p = 1.0
    for v in known:
        p *= 1.0 - v
    return round(1.0 - p, 4), "noisy_or_agreement"


def _match(a: NoteEvent, b: NoteEvent, onset_tol: float, pitch_tol: int) -> bool:
    return abs(a.midi_note - b.midi_note) <= pitch_tol and abs(a.onset - b.onset) <= onset_tol


def merge_note_runs(runs: list[list[NoteEvent]], onset_tol: float = 0.05, pitch_tol: int = 0, unmatched_weight: float = 0.8) -> list[NoteEvent]:
    """Consensus merge of several adapters' outputs for the *same* stem.

    Matched notes (same pitch, onset within tolerance) are merged: onset/offset are
    confidence-weighted averages, confidence is combined (noisy-OR). Notes seen by only one
    adapter are kept with their confidence scaled by ``unmatched_weight`` (when >1 adapters).
    """
    if not runs:
        return []
    if len(runs) == 1:
        return list(runs[0])
    pool: list[list[NoteEvent]] = [[n] for n in runs[0]]
    for other in runs[1:]:
        used = set()
        for n in other:
            best = None
            for gi, group in enumerate(pool):
                if gi in used:
                    continue
                if _match(group[0], n, onset_tol, pitch_tol):
                    best = gi
                    break
            if best is None:
                pool.append([n])
            else:
                pool[best].append(n)
                used.add(best)
    out: list[NoteEvent] = []
    for group in pool:
        if len(group) == 1:
            n = group[0]
            c = n.confidence
            out.append(NoteEvent(midi_note=n.midi_note, onset=n.onset, offset=n.offset, velocity=n.velocity, pitch_bend=n.pitch_bend, features=dict(n.features), track=n.track, instrument=n.instrument, sources=[n.event_id], confidence=None if c is None else round(c * unmatched_weight, 4), confidence_method=n.confidence_method if c is None else "single_adapter_scaled", provenance=n.provenance))
            continue
        w = [max(0.05, (g.confidence if g.confidence is not None else 0.5)) for g in group]
        tot = sum(w)
        onset = sum(g.onset * wi for g, wi in zip(group, w)) / tot
        offset = sum(g.offset * wi for g, wi in zip(group, w)) / tot
        vel = int(round(sum(g.velocity * wi for g, wi in zip(group, w)) / tot))
        conf, method = combine_confidence([g.confidence for g in group])
        lead = max(group, key=lambda g: g.confidence or 0.0)
        bend = next((g.pitch_bend for g in group if g.pitch_bend), None)
        feats: dict = {}
        for g in group:
            feats.update(g.features)
        out.append(NoteEvent(midi_note=lead.midi_note, onset=round(onset, 4), offset=round(offset, 4), velocity=vel, pitch_bend=bend, features=feats, track=lead.track, instrument=lead.instrument, sources=[g.event_id for g in group], confidence=conf, confidence_method=method, provenance=lead.provenance))
    out.sort(key=lambda n: (n.onset, n.midi_note))
    return out


def fuse_with_fullmix(stem_notes: list[NoteEvent], fullmix_notes: list[NoteEvent], onset_tol: float = 0.05, pitch_tol: int = 0, fullmix_only_weight: float = 0.6, unassigned_track: str = "unassigned") -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Path A (stems) + Path B (full mix). Returns (fused_stem_notes, extra_fullmix_only_notes)."""
    if not fullmix_notes:
        return list(stem_notes), []
    stem_sorted = sorted(stem_notes, key=lambda n: n.onset)
    matched: dict[str, NoteEvent] = {}
    extras: list[NoteEvent] = []
    for fm in fullmix_notes:
        cand = None
        for sn in stem_sorted:
            if sn.onset > fm.onset + onset_tol:
                break
            if sn.event_id in matched:
                continue
            if _match(sn, fm, onset_tol, pitch_tol):
                cand = sn
                break
        if cand is None:
            c = fm.confidence
            extras.append(NoteEvent(midi_note=fm.midi_note, onset=fm.onset, offset=fm.offset, velocity=fm.velocity, track=unassigned_track, instrument="unknown", sources=[fm.event_id], confidence=None if c is None else round(c * fullmix_only_weight, 4), confidence_method="fullmix_only_scaled" if c is not None else fm.confidence_method, provenance=fm.provenance))
        else:
            matched[cand.event_id] = fm
    out: list[NoteEvent] = []
    for sn in stem_notes:
        fm = matched.get(sn.event_id)
        if fm is None:
            out.append(sn)
            continue
        wa = max(0.05, sn.confidence if sn.confidence is not None else 0.5)
        wb = max(0.05, fm.confidence if fm.confidence is not None else 0.5)
        onset = (sn.onset * wa + fm.onset * wb) / (wa + wb)
        offset = (sn.offset * wa + fm.offset * wb) / (wa + wb)
        conf, method = combine_confidence([sn.confidence, fm.confidence])
        out.append(NoteEvent(midi_note=sn.midi_note, onset=round(onset, 4), offset=round(offset, 4), velocity=sn.velocity, pitch_bend=sn.pitch_bend or fm.pitch_bend, features={**fm.features, **sn.features}, track=sn.track, instrument=sn.instrument, sources=list(sn.sources or [sn.event_id]) + [fm.event_id], confidence=conf, confidence_method="stem_fullmix_agreement" if conf is not None else method, provenance=sn.provenance))
    out.sort(key=lambda n: (n.onset, n.midi_note))
    return out, extras


def merge_drum_runs(broad: list[DrumEvent], components: list[DrumEvent], tol: float = 0.03, add_unmatched: bool = True, relabel_min_score: float = 0.35) -> list[DrumEvent]:
    """Merge broad-stem drum events with per-component events (which carry class hints).

    ``add_unmatched=False`` treats component events as evidence only (sensible for crude
    band-filter components): they can confirm/relabel broad events but never add new ones.
    """
    if not components:
        return list(broad)
    out: list[DrumEvent] = []
    used: set[str] = set()
    broad_sorted = sorted(broad, key=lambda e: e.time)
    for comp in sorted(components, key=lambda e: e.time):
        match = None
        for b in broad_sorted:
            if b.event_id in used:
                continue
            if abs(b.time - comp.time) <= tol and (b.drum_class == comp.drum_class or b.class_scores.get(comp.drum_class, 0.0) >= relabel_min_score):
                match = b
                break
        if match is None:
            if add_unmatched:
                out.append(comp)
            continue
        used.add(match.event_id)
        conf, method = combine_confidence([match.confidence, comp.confidence])
        out.append(DrumEvent(time=round((match.time + comp.time) / 2, 4), drum_class=comp.drum_class, velocity=max(match.velocity, comp.velocity), duration=match.duration or comp.duration, track=match.track, class_scores=match.class_scores, confidence=conf, confidence_method=method, provenance=match.provenance))
    out.extend(b for b in broad_sorted if b.event_id not in used)
    out.sort(key=lambda e: (e.time, e.drum_class))
    return dedupe_drums(out, tol)


def dedupe_drums(events: list[DrumEvent], tol: float = 0.02) -> list[DrumEvent]:
    out: list[DrumEvent] = []
    for e in sorted(events, key=lambda e: (e.time, e.drum_class)):
        if out and out[-1].drum_class == e.drum_class and abs(out[-1].time - e.time) <= tol:
            if (e.confidence or 0) > (out[-1].confidence or 0):
                out[-1] = e
            continue
        out.append(e)
    return out


def drum_leak_filter(notes: list[NoteEvent], drums: list[DrumEvent], max_note_seconds: float = 0.15, tol: float = 0.03, classes: tuple[str, ...] = ("kick", "snare", "low_tom", "mid_tom", "high_tom")) -> tuple[list[NoteEvent], int]:
    """Drop very short pitched notes that start exactly on a drum hit: in stem-based AMT these
    are almost always drum leakage (e.g. a kick's pitch sweep inside the bass stem)."""
    if not drums:
        return list(notes), 0
    times = sorted(d.time for d in drums if d.drum_class in classes)
    import bisect

    kept: list[NoteEvent] = []
    dropped = 0
    for n in notes:
        if n.duration <= max_note_seconds:
            i = bisect.bisect_left(times, n.onset - tol)
            if i < len(times) and abs(times[i] - n.onset) <= tol:
                dropped += 1
                continue
        kept.append(n)
    return kept, dropped


def merge_drum_adapter_runs(runs: list[list[DrumEvent]], tol: float = 0.03, unmatched_weight: float = 0.8) -> list[DrumEvent]:
    """Consensus between several drum-transcription adapters on the same stem (class + time)."""
    if not runs:
        return []
    if len(runs) == 1:
        return list(runs[0])
    pool: list[list[DrumEvent]] = [[e] for e in runs[0]]
    for other in runs[1:]:
        used: set[int] = set()
        for e in other:
            match = None
            for gi, group in enumerate(pool):
                if gi in used:
                    continue
                if group[0].drum_class == e.drum_class and abs(group[0].time - e.time) <= tol:
                    match = gi
                    break
            if match is None:
                pool.append([e])
            else:
                pool[match].append(e)
                used.add(match)
    out: list[DrumEvent] = []
    for group in pool:
        if len(group) == 1:
            e = group[0]
            c = e.confidence
            out.append(DrumEvent(time=e.time, drum_class=e.drum_class, velocity=e.velocity, duration=e.duration, track=e.track, class_scores=e.class_scores, confidence=None if c is None else round(c * unmatched_weight, 4), confidence_method=e.confidence_method if c is None else "single_adapter_scaled", provenance=e.provenance))
            continue
        w = [max(0.05, g.confidence if g.confidence is not None else 0.5) for g in group]
        t = sum(g.time * wi for g, wi in zip(group, w)) / sum(w)
        conf, method = combine_confidence([g.confidence for g in group])
        lead = max(group, key=lambda g: g.confidence or 0.0)
        out.append(DrumEvent(time=round(t, 4), drum_class=lead.drum_class, velocity=max(g.velocity for g in group), duration=lead.duration, track=lead.track, class_scores=lead.class_scores, confidence=conf, confidence_method=method, provenance=lead.provenance))
    out.sort(key=lambda e: (e.time, e.drum_class))
    return out
