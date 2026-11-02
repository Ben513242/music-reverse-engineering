"""Music-theory helpers (pure Python, no numpy)."""

from __future__ import annotations

import math
import re

NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_to_hz(midi: float, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((midi - 69.0) / 12.0)


def hz_to_midi(hz: float, a4: float = 440.0) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(hz / a4)


def midi_to_name(midi: int, flats: bool = False) -> str:
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    return f"{names[int(midi) % 12]}{int(midi) // 12 - 1}"


def name_to_midi(name: str) -> int:
    m = re.fullmatch(r"([A-Ga-g])([#b]*)(-?\d+)", name.strip())
    if not m:
        raise ValueError(f"bad note name {name!r}")
    step, acc, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pc = _PC[step] + acc.count("#") - acc.count("b")
    return (octave + 1) * 12 + pc


def pitch_class_name(pc: int, flats: bool = False) -> str:
    return (NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP)[pc % 12]


def pitch_class_from_name(name: str) -> int:
    return name_to_midi(f"{name}4") % 12


# ---------------------------------------------------------------- keys
# circle of fifths: index = number of sharps (positive) or flats (negative)
_MAJOR_FIFTHS = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
                 -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb"}
_MINOR_FIFTHS = {0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
                 -1: "D", -2: "G", -3: "C", -4: "F", -5: "Bb", -6: "Eb", -7: "Ab"}


def key_fifths(tonic_pc: int, mode: str) -> int:
    """Return the key signature (count of sharps, negative for flats) for a tonic pitch class."""
    table = _MINOR_FIFTHS if mode == "minor" else _MAJOR_FIFTHS
    candidates = [f for f, name in table.items() if pitch_class_from_name(name) == tonic_pc % 12]
    if not candidates:
        return 0
    return min(candidates, key=abs)


def key_name(tonic_pc: int, mode: str) -> str:
    fifths = key_fifths(tonic_pc, mode)
    table = _MINOR_FIFTHS if mode == "minor" else _MAJOR_FIFTHS
    return f"{table[fifths]} {mode}"


def key_uses_flats(fifths: int) -> bool:
    return fifths < 0


def spell_pitch(midi: int, flats: bool = False) -> tuple[str, int, int]:
    """Return (step, alter, octave) for MusicXML."""
    pc = int(midi) % 12
    octave = int(midi) // 12 - 1
    if flats:
        name = NOTE_NAMES_FLAT[pc]
        alter = -1 if name.endswith("b") else 0
    else:
        name = NOTE_NAMES_SHARP[pc]
        alter = 1 if name.endswith("#") else 0
    return name[0], alter, octave


# ---------------------------------------------------------------- chords
CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "add9": (0, 2, 4, 7),
    "power": (0, 7),
}
CHORD_SUFFIX = {
    "major": "",
    "minor": "m",
    "dom7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "sus2": "sus2",
    "sus4": "sus4",
    "dim": "dim",
    "aug": "aug",
    "add9": "add9",
    "power": "5",
}
MUSICXML_KIND = {
    "major": "major",
    "minor": "minor",
    "dom7": "dominant",
    "maj7": "major-seventh",
    "min7": "minor-seventh",
    "sus2": "suspended-second",
    "sus4": "suspended-fourth",
    "dim": "diminished",
    "aug": "augmented",
    "add9": "major",
    "power": "power",
}


def chord_symbol(root: str, quality: str, bass: str | None = None) -> str:
    sym = f"{root}{CHORD_SUFFIX.get(quality, quality)}"
    if bass and bass != root:
        sym += f"/{bass}"
    return sym


def chord_pitch_classes(root_pc: int, quality: str) -> list[int]:
    return [(root_pc + i) % 12 for i in CHORD_TEMPLATES[quality]]


def scale_degrees(pcs: list[int], root_pc: int, quality: str) -> list[int]:
    """Map pitch classes to chord-degree numbers (1, 3, 5, 7, 9) for arpeggio patterns."""
    template = CHORD_TEMPLATES[quality]
    degree_of = {}
    for idx, interval in enumerate(template):
        degree_of[(root_pc + interval) % 12] = (1, 3, 5, 7)[idx] if quality != "add9" else (1, 9, 3, 5)[idx]
        if quality in ("sus2", "sus4") and idx == 1:
            degree_of[(root_pc + interval) % 12] = 2 if quality == "sus2" else 4
    return [degree_of.get(pc % 12, 0) for pc in pcs]


# ---------------------------------------------------------------- key profiles (Krumhansl-Schmuckler)
KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


# ---------------------------------------------------------------- dynamics
DYNAMIC_MARKS = ("pp", "p", "mp", "mf", "f", "ff")


def velocity_to_dynamic(velocity: int) -> str:
    if velocity < 32:
        return "pp"
    if velocity < 48:
        return "p"
    if velocity < 64:
        return "mp"
    if velocity < 80:
        return "mf"
    if velocity < 100:
        return "f"
    return "ff"


def loudness_to_dynamic(rms_db: float) -> str:
    """Map an RMS level (dBFS) to a dynamic marking, relative to a -12 dBFS 'mf' reference."""
    if rms_db < -34:
        return "pp"
    if rms_db < -28:
        return "p"
    if rms_db < -22:
        return "mp"
    if rms_db < -16:
        return "mf"
    if rms_db < -10:
        return "f"
    return "ff"


def amplitude_to_velocity(ratio: float, floor: int = 20) -> int:
    """Map a 0..1 amplitude ratio to MIDI velocity with a gentle square-root curve."""
    ratio = max(0.0, min(1.0, float(ratio)))
    return int(round(floor + (127 - floor) * math.sqrt(ratio)))
