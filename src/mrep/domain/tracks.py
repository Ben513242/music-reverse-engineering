"""Stems (audio artifacts of separation) and tracks (musical lines)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mrep.domain.events import DrumEvent, FXEvent, NoteEvent
from mrep.domain.labels import SoundLabel
from mrep.domain.provenance import Provenance


@dataclass
class Stem:
    name: str  # unique id, e.g. "drums", "drums/kick", "other"
    path: str  # relative to project root
    source_hint: str  # family hint from the separator ("drums", "bass", "other", ...)
    parent: str | None = None  # name of the stem this was refined from
    level: str = "broad"  # broad | specialist
    sample_rate: int = 44100
    channels: int = 2
    duration: float = 0.0
    energy_ratio: float | None = None  # energy relative to the mix
    confidence: float | None = None
    confidence_method: str = "unavailable"
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["provenance"] = self.provenance.to_dict() if self.provenance else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Stem:
        data = dict(d)
        data["provenance"] = Provenance.from_dict(data.get("provenance"))
        return cls(**data)

    @property
    def is_leaf(self) -> bool:
        return self.level == "specialist" or self.parent is None

    @property
    def file_stem(self) -> str:
        return self.name.replace("/", "_")


@dataclass
class Track:
    """A musical line ready for scoring / rendering."""

    name: str
    stem: str | None = None
    label: SoundLabel | None = None
    notes: list[NoteEvent] = field(default_factory=list)
    drums: list[DrumEvent] = field(default_factory=list)
    fx: list[FXEvent] = field(default_factory=list)
    program: int = 0
    is_drum: bool = False
    is_fx: bool = False

    @property
    def display_name(self) -> str:
        return track_display_name(self.name, self.label)


def track_display_name(stem_name: str, label: SoundLabel | None) -> str:
    """Human-friendly file/track name, e.g. ``Synth_Pad`` or ``Bass``."""
    base = stem_name.split("/")[-1]
    if label is not None and label.source_family not in {"unknown", "harmonic"}:
        if label.source_subtype:
            base = label.source_subtype
        else:
            base = label.source_family
        if label.source_family == "synthesizer" and label.musical_role:
            base = f"synth_{label.musical_role[0]}"
    parts = [p for p in base.replace("-", "_").split("_") if p]
    return "_".join(p.capitalize() for p in parts) or "Track"
