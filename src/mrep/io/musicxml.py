"""Minimal, dependency-free MusicXML (partwise 3.1) writer.

Input is already quantised: positions and durations are integer *divisions*
(``DIVISIONS`` per quarter note). The writer handles voices, ties across barlines,
rests, chord symbols, dynamics, percussion parts and key changes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from xml.dom import minidom

from mrep.utils.music import MUSICXML_KIND, key_uses_flats, spell_pitch

DIVISIONS = 24  # per quarter note: 16th = 6, triplet 16th = 4, 32nd = 3

# duration (divisions) -> (type, dots, triplet)
_NOTE_TYPES: dict[int, tuple[str, int, bool]] = {
    96: ("whole", 0, False),
    72: ("half", 1, False),
    48: ("half", 0, False),
    36: ("quarter", 1, False),
    24: ("quarter", 0, False),
    18: ("eighth", 1, False),
    12: ("eighth", 0, False),
    8: ("eighth", 0, True),
    6: ("16th", 0, False),
    4: ("16th", 0, True),
    3: ("32nd", 0, False),
    2: ("32nd", 0, True),
}
_PIECES = sorted(_NOTE_TYPES, reverse=True)


def decompose_duration(d: int, max_piece: int | None = None) -> list[int]:
    """Split ``d`` divisions into notated pieces (fewest pieces, largest first)."""
    d = int(d)
    if d <= 0:
        return []
    pieces = [p for p in _PIECES if max_piece is None or p <= max_piece]
    best: list[int | None] = [None] * (d + 1)
    best[0] = 0
    choice = [0] * (d + 1)
    for v in range(1, d + 1):
        for p in pieces:
            if p <= v and best[v - p] is not None:
                cand = best[v - p] + 1  # type: ignore[operator]
                if best[v] is None or cand < best[v]:  # type: ignore[operator]
                    best[v] = cand
                    choice[v] = p
    if best[d] is None:  # 1 division is not notatable; round up to the smallest piece
        return decompose_duration(d + 1, max_piece)
    out: list[int] = []
    v = d
    while v > 0:
        out.append(choice[v])
        v -= choice[v]
    return sorted(out, reverse=True)


@dataclass
class XmlNote:
    start: int  # divisions from song start
    duration: int
    midi: int | None = None  # None for unpitched
    velocity: int = 64
    drum_class: str | None = None
    articulation: str = "normal"  # normal | staccato | legato
    slur: str | None = None  # start | stop


@dataclass
class XmlPart:
    part_id: str
    name: str
    notes: list[XmlNote] = field(default_factory=list)
    is_drum: bool = False
    program: int = 0
    drum_midi: dict[str, int] = field(default_factory=dict)
    drum_notation: dict[str, dict] = field(default_factory=dict)


@dataclass
class XmlHarmony:
    start: int
    root: str
    quality: str
    bass: str | None = None


@dataclass
class ScoreSpec:
    title: str = "MREP score"
    beats: int = 4
    beat_type: int = 4
    bpm: float = 120.0
    key_changes: list[tuple[int, int, str]] = field(default_factory=list)  # (measure, fifths, mode)
    parts: list[XmlPart] = field(default_factory=list)
    harmonies: list[XmlHarmony] = field(default_factory=list)
    dynamics: list[tuple[int, str]] = field(default_factory=list)  # (start_div, mark)
    tempo_changes: list[tuple[int, float]] = field(default_factory=list)  # (measure, bpm) after measure 1
    software: str = "MREP"

    @property
    def measure_length(self) -> int:
        return int(DIVISIONS * self.beats * 4 / self.beat_type)


def _step_alter(name: str) -> tuple[str, int]:
    step = name[0].upper()
    alter = name[1:].count("#") - name[1:].count("b")
    return step, alter


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


def _assign_voices(groups: list[tuple[int, int, list[XmlNote]]], max_voices: int = 4) -> list[list[tuple[int, int, list[XmlNote]]]]:
    voices: list[list[tuple[int, int, list[XmlNote]]]] = []
    ends: list[int] = []
    for g in groups:
        start, dur, _ = g
        placed = False
        for vi, end in enumerate(ends):
            if start >= end:
                voices[vi].append(g)
                ends[vi] = start + dur
                placed = True
                break
        if placed:
            continue
        if len(voices) < max_voices:
            voices.append([g])
            ends.append(start + dur)
        else:  # truncate the voice that frees up first
            vi = min(range(len(ends)), key=lambda i: ends[i])
            ps, pd, pn = voices[vi][-1]
            new_dur = start - ps
            if new_dur >= 2:
                voices[vi][-1] = (ps, new_dur, pn)
                voices[vi].append(g)
                ends[vi] = start + dur
    return voices


class MusicXMLWriter:
    def __init__(self, spec: ScoreSpec):
        self.spec = spec
        self.mlen = spec.measure_length
        last = 0
        for p in spec.parts:
            for n in p.notes:
                last = max(last, n.start + n.duration)
        for h in spec.harmonies:
            last = max(last, h.start + 1)
        self.n_measures = max(1, int(math.ceil(last / self.mlen)))
        self.key_by_measure: dict[int, tuple[int, str]] = {}
        changes = sorted(spec.key_changes) or [(1, 0, "major")]
        if changes[0][0] != 1:
            changes.insert(0, (1, 0, "major"))
        for m, fifths, mode in changes:
            self.key_by_measure[int(m)] = (int(fifths), mode)

    # ---- helpers -----------------------------------------------------------------
    def _key_at(self, measure: int) -> tuple[int, str]:
        cur = (0, "major")
        for m in sorted(self.key_by_measure):
            if m <= measure:
                cur = self.key_by_measure[m]
        return cur

    def _note_elements(self, parent: ET.Element, group: list[XmlNote], duration: int, voice: int, part: XmlPart, flats: bool, tie_start: bool, tie_stop: bool) -> None:
        pieces = decompose_duration(duration)
        for pi, piece in enumerate(pieces):
            p_tie_start = tie_start or pi < len(pieces) - 1
            p_tie_stop = tie_stop or pi > 0
            for ni, n in enumerate(group):
                note_el = _sub(parent, "note")
                if ni > 0:
                    _sub(note_el, "chord")
                if part.is_drum:
                    cls = n.drum_class or "other_percussion"
                    notation = part.drum_notation.get(cls, {"step": "B", "octave": 4, "notehead": "normal"})
                    unp = _sub(note_el, "unpitched")
                    _sub(unp, "display-step", notation.get("step", "B"))
                    _sub(unp, "display-octave", str(notation.get("octave", 4)))
                else:
                    step, alter, octave = spell_pitch(int(n.midi or 60), flats)
                    pitch = _sub(note_el, "pitch")
                    _sub(pitch, "step", step)
                    if alter:
                        _sub(pitch, "alter", str(alter))
                    _sub(pitch, "octave", str(octave))
                _sub(note_el, "duration", str(piece))
                if p_tie_stop and not part.is_drum:
                    _sub(note_el, "tie", type="stop")
                if p_tie_start and not part.is_drum:
                    _sub(note_el, "tie", type="start")
                if part.is_drum:
                    cls = n.drum_class or "other_percussion"
                    _sub(note_el, "instrument", id=f"{part.part_id}-I{part.drum_midi.get(cls, 75)}")
                _sub(note_el, "voice", str(voice))
                ntype, dots, triplet = _NOTE_TYPES[piece]
                _sub(note_el, "type", ntype)
                for _ in range(dots):
                    _sub(note_el, "dot")
                if not part.is_drum:
                    _, alter, _ = spell_pitch(int(n.midi or 60), flats)
                    if alter:
                        _sub(note_el, "accidental", "sharp" if alter > 0 else "flat")
                if triplet:
                    tm = _sub(note_el, "time-modification")
                    _sub(tm, "actual-notes", "3")
                    _sub(tm, "normal-notes", "2")
                if part.is_drum:
                    nh = part.drum_notation.get(n.drum_class or "", {}).get("notehead", "normal")
                    if nh != "normal":
                        _sub(note_el, "notehead", nh)
                lead = group[0]
                want_stacc = (not part.is_drum) and ni == 0 and pi == 0 and lead.articulation == "staccato"
                want_slur_start = (not part.is_drum) and ni == 0 and pi == 0 and lead.slur == "start" and not tie_stop
                want_slur_stop = (not part.is_drum) and ni == 0 and pi == len(pieces) - 1 and lead.slur == "stop" and not tie_start
                if ((p_tie_start or p_tie_stop) and not part.is_drum) or want_stacc or want_slur_start or want_slur_stop:
                    notations = _sub(note_el, "notations")
                    if p_tie_stop and not part.is_drum:
                        _sub(notations, "tied", type="stop")
                    if p_tie_start and not part.is_drum:
                        _sub(notations, "tied", type="start")
                    if want_slur_stop:
                        _sub(notations, "slur", type="stop", number="1")
                    if want_slur_start:
                        _sub(notations, "slur", type="start", number="1")
                    if want_stacc:
                        arts = _sub(notations, "articulations")
                        _sub(arts, "staccato")

    def _rest_elements(self, parent: ET.Element, duration: int, voice: int, whole_measure: bool = False) -> None:
        if whole_measure:
            note_el = _sub(parent, "note")
            _sub(note_el, "rest", measure="yes")
            _sub(note_el, "duration", str(duration))
            _sub(note_el, "voice", str(voice))
            return
        for piece in decompose_duration(duration):
            note_el = _sub(parent, "note")
            _sub(note_el, "rest")
            _sub(note_el, "duration", str(piece))
            _sub(note_el, "voice", str(voice))
            ntype, dots, triplet = _NOTE_TYPES[piece]
            _sub(note_el, "type", ntype)
            for _ in range(dots):
                _sub(note_el, "dot")
            if triplet:
                tm = _sub(note_el, "time-modification")
                _sub(tm, "actual-notes", "3")
                _sub(tm, "normal-notes", "2")

    # ---- main --------------------------------------------------------------------
    def build(self) -> ET.Element:
        spec = self.spec
        root = ET.Element("score-partwise", version="3.1")
        work = _sub(root, "work")
        _sub(work, "work-title", spec.title)
        ident = _sub(root, "identification")
        enc = _sub(ident, "encoding")
        _sub(enc, "software", spec.software)
        part_list = _sub(root, "part-list")
        for part in spec.parts:
            sp = _sub(part_list, "score-part", id=part.part_id)
            _sub(sp, "part-name", part.name)
            if part.is_drum:
                for cls, midi in sorted(part.drum_midi.items(), key=lambda kv: kv[1]):
                    si = _sub(sp, "score-instrument", id=f"{part.part_id}-I{midi}")
                    _sub(si, "instrument-name", cls.replace("_", " ").title())
                for _cls, midi in sorted(part.drum_midi.items(), key=lambda kv: kv[1]):
                    mi = _sub(sp, "midi-instrument", id=f"{part.part_id}-I{midi}")
                    _sub(mi, "midi-channel", "10")
                    _sub(mi, "midi-unpitched", str(midi + 1))
            else:
                si = _sub(sp, "score-instrument", id=f"{part.part_id}-I1")
                _sub(si, "instrument-name", part.name)
                mi = _sub(sp, "midi-instrument", id=f"{part.part_id}-I1")
                _sub(mi, "midi-channel", "1")
                _sub(mi, "midi-program", str(int(part.program) + 1))

        harmonies_by_measure: dict[int, list[XmlHarmony]] = {}
        for h in spec.harmonies:
            harmonies_by_measure.setdefault(h.start // self.mlen + 1, []).append(h)
        dynamics_by_measure: dict[int, list[tuple[int, str]]] = {}
        for start, mark in spec.dynamics:
            dynamics_by_measure.setdefault(start // self.mlen + 1, []).append((start, mark))

        for pi, part in enumerate(spec.parts):
            part_el = _sub(root, "part", id=part.part_id)
            groups_map: dict[tuple[int, int], list[XmlNote]] = {}
            for n in part.notes:
                if n.duration <= 0:
                    continue
                groups_map.setdefault((n.start, n.duration), []).append(n)
            groups = sorted((s, d, ns) for (s, d), ns in groups_map.items())
            voices = _assign_voices(groups) or [[]]
            mean_midi = (sum(n.midi or 60 for n in part.notes) / len(part.notes)) if part.notes else 60
            for m in range(1, self.n_measures + 1):
                m_start, m_end = (m - 1) * self.mlen, m * self.mlen
                measure = _sub(part_el, "measure", number=str(m))
                fifths, mode = self._key_at(m)
                flats = key_uses_flats(fifths)
                if m == 1 or m in self.key_by_measure:
                    attrs = _sub(measure, "attributes")
                    if m == 1:
                        _sub(attrs, "divisions", str(DIVISIONS))
                    key_el = _sub(attrs, "key")
                    _sub(key_el, "fifths", str(fifths))
                    _sub(key_el, "mode", mode)
                    if m == 1:
                        time_el = _sub(attrs, "time")
                        _sub(time_el, "beats", str(spec.beats))
                        _sub(time_el, "beat-type", str(spec.beat_type))
                        clef = _sub(attrs, "clef")
                        if part.is_drum:
                            _sub(clef, "sign", "percussion")
                            _sub(clef, "line", "2")
                        elif mean_midi < 57:
                            _sub(clef, "sign", "F")
                            _sub(clef, "line", "4")
                        else:
                            _sub(clef, "sign", "G")
                            _sub(clef, "line", "2")
                tempo_here = spec.bpm if m == 1 else next((b for mm, b in spec.tempo_changes if mm == m), None)
                if pi == 0 and tempo_here is not None:
                    d = _sub(measure, "direction", placement="above")
                    dt = _sub(d, "direction-type")
                    met = _sub(dt, "metronome")
                    _sub(met, "beat-unit", "quarter")
                    _sub(met, "per-minute", f"{tempo_here:.0f}")
                    _sub(d, "sound", tempo=f"{tempo_here:.2f}")
                for start, mark in dynamics_by_measure.get(m, []):
                    d = _sub(measure, "direction", placement="below")
                    dt = _sub(d, "direction-type")
                    dyn = _sub(dt, "dynamics")
                    _sub(dyn, mark)
                    if start - m_start > 0:
                        _sub(d, "offset", str(start - m_start))
                if pi == 0:
                    for h in harmonies_by_measure.get(m, []):
                        harm = _sub(measure, "harmony")
                        r = _sub(harm, "root")
                        step, alter = _step_alter(h.root)
                        _sub(r, "root-step", step)
                        if alter:
                            _sub(r, "root-alter", str(alter))
                        _sub(harm, "kind", MUSICXML_KIND.get(h.quality, "major"))
                        if h.quality == "add9":
                            deg = _sub(harm, "degree")
                            _sub(deg, "degree-value", "9")
                            _sub(deg, "degree-alter", "0")
                            _sub(deg, "degree-type", "add")
                        if h.bass and h.bass != h.root:
                            b = _sub(harm, "bass")
                            step, alter = _step_alter(h.bass)
                            _sub(b, "bass-step", step)
                            if alter:
                                _sub(b, "bass-alter", str(alter))
                        if h.start - m_start > 0:
                            _sub(harm, "offset", str(h.start - m_start))
                for vi, voice in enumerate(voices, start=1):
                    if vi > 1:
                        backup = _sub(measure, "backup")
                        _sub(backup, "duration", str(self.mlen))
                    cursor = m_start
                    wrote = False
                    for start, dur, notes in voice:
                        end = start + dur
                        if end <= m_start or start >= m_end:
                            continue
                        p_start, p_end = max(start, m_start), min(end, m_end)
                        if p_start > cursor:
                            self._rest_elements(measure, p_start - cursor, vi)
                        self._note_elements(measure, notes, p_end - p_start, vi, part, flats, tie_start=end > m_end, tie_stop=start < m_start)
                        cursor = p_end
                        wrote = True
                    if not wrote:
                        self._rest_elements(measure, self.mlen, vi, whole_measure=True)
                    elif cursor < m_end:
                        self._rest_elements(measure, m_end - cursor, vi)
        return root


def write_musicxml(path: Path, spec: ScoreSpec) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = MusicXMLWriter(spec).build()
    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    body = pretty.split("\n", 1)[1] if pretty.startswith("<?xml") else pretty
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def read_musicxml_summary(path: Path) -> dict:
    """Small structural summary used by tests (parts, measures, notes, harmonies)."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    parts = root.findall("part")
    return {
        "parts": [p.get("id") for p in parts],
        "part_names": [pn.text for pn in root.findall("part-list/score-part/part-name")],
        "measures": max((len(p.findall("measure")) for p in parts), default=0),
        "notes": sum(1 for p in parts for n in p.iter("note") if n.find("rest") is None),
        "rests": sum(1 for p in parts for n in p.iter("note") if n.find("rest") is not None),
        "harmonies": sum(1 for p in parts for _ in p.iter("harmony")),
        "ties": sum(1 for p in parts for _ in p.iter("tie")),
        "staccatos": sum(1 for p in parts for _ in p.iter("staccato")),
        "slurs": sum(1 for p in parts for _ in p.iter("slur")),
        "tempo_marks": sum(1 for p in parts for _ in p.iter("metronome")),
    }
