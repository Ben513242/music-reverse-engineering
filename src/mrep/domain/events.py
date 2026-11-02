"""Event model. Every AI-derived event carries ``confidence`` + ``provenance``.

RAW events (seconds, as estimated by models) and SCORE events (quantised, in beats) are
different classes and are stored in different files. Score events reference the raw event
they came from and never overwrite it.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from mrep.domain.labels import validate_confidence
from mrep.domain.provenance import Provenance
from mrep.utils.music import midi_to_name


def new_event_id(prefix: str = "ev") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(kw_only=True)
class Event:
    TYPE: ClassVar[str] = "event"

    event_id: str = ""
    confidence: float | None = None
    confidence_method: str = "unavailable"
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        self.confidence = validate_confidence(self.confidence)
        if self.confidence is None and not self.confidence_method:
            self.confidence_method = "unavailable"
        if not self.event_id:
            self.event_id = new_event_id(self.TYPE)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.TYPE
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        data = dict(d)
        data.pop("type", None)
        prov = data.get("provenance")
        if isinstance(prov, dict):
            data["provenance"] = Provenance.from_dict(prov)
        # tolerate unknown keys from newer schema versions
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)


@dataclass(kw_only=True)
class NoteEvent(Event):
    """A raw pitched note in seconds."""

    TYPE: ClassVar[str] = "note"

    midi_note: int
    onset: float
    offset: float
    velocity: int = 64
    pitch: str = ""
    pitch_bend: list[list[float]] | None = None  # [[time, semitones], ...] or None
    track: str = ""
    instrument: str = ""
    sources: list[str] = field(default_factory=list)  # ids of events merged into this one
    features: dict[str, Any] = field(default_factory=dict)  # e.g. {"vibrato": {"rate_hz", "depth_cents"}}

    def __post_init__(self) -> None:
        super().__post_init__()
        self.midi_note = int(self.midi_note)
        self.velocity = int(max(1, min(127, self.velocity)))
        if self.offset < self.onset:
            self.offset = self.onset
        if not self.pitch:
            self.pitch = midi_to_name(self.midi_note)

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass(kw_only=True)
class DrumEvent(Event):
    TYPE: ClassVar[str] = "drum"

    time: float
    drum_class: str
    velocity: int = 100
    duration: float | None = None
    track: str = "drums"
    class_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.velocity = int(max(1, min(127, self.velocity)))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["class"] = d.pop("drum_class")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DrumEvent:
        data = dict(d)
        if "class" in data and "drum_class" not in data:
            data["drum_class"] = data.pop("class")
        return super().from_dict(data)  # type: ignore[return-value]


@dataclass(kw_only=True)
class FXEvent(Event):
    TYPE: ClassVar[str] = "fx"

    fx_class: str
    start: float
    end: float
    clip_path: str | None = None
    trigger_note: int | None = None
    features: dict[str, Any] = field(default_factory=dict)
    track: str = "fx"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["class"] = d.pop("fx_class")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FXEvent:
        data = dict(d)
        if "class" in data and "fx_class" not in data:
            data["fx_class"] = data.pop("class")
        return super().from_dict(data)  # type: ignore[return-value]


@dataclass(kw_only=True)
class ChordEvent(Event):
    TYPE: ClassVar[str] = "chord"

    start: float
    end: float
    root: str
    quality: str
    bass: str | None = None
    symbol: str = ""
    inversion: int = 0
    pitch_classes: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class KeyEvent(Event):
    TYPE: ClassVar[str] = "key"

    start: float
    end: float | None = None
    tonic: str = "C"
    mode: str = "major"

    @property
    def key(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["key"] = self.key
        return d


@dataclass(kw_only=True)
class TempoEvent(Event):
    TYPE: ClassVar[str] = "tempo"

    time: float
    bpm: float


@dataclass(kw_only=True)
class BeatEvent(Event):
    TYPE: ClassVar[str] = "beat"

    time: float
    position: int = 1  # 1-based beat within the bar
    is_downbeat: bool = False


@dataclass(kw_only=True)
class SectionEvent(Event):
    TYPE: ClassVar[str] = "section"

    start: float
    end: float
    label: str
    dynamic: str | None = None  # pp .. ff
    loudness_db: float | None = None


@dataclass(kw_only=True)
class ScoreNoteEvent(Event):
    """A quantised note, in beats. ``raw_event_id`` points back to the raw NoteEvent."""

    TYPE: ClassVar[str] = "score_note"

    raw_event_id: str
    midi_note: int
    beat_onset: float  # beats from the start of the song (0-based)
    beat_duration: float
    measure: int  # 1-based
    beat_in_measure: float  # 0-based position inside the measure, in beats
    grid: str = "1/16"
    velocity: int = 64
    dynamic: str = "mf"
    pitch: str = ""
    track: str = ""
    instrument: str = ""
    timing_offset_beats: float = 0.0  # raw - quantised (micro timing, for humanisation)
    articulation: str = "normal"  # normal | staccato | legato
    duration_ratio: float | None = None  # raw duration / inter-onset interval (humanisation)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.pitch:
            self.pitch = midi_to_name(int(self.midi_note))


EVENT_TYPES: dict[str, type[Event]] = {
    cls.TYPE: cls
    for cls in (
        NoteEvent,
        DrumEvent,
        FXEvent,
        ChordEvent,
        KeyEvent,
        TempoEvent,
        BeatEvent,
        SectionEvent,
        ScoreNoteEvent,
    )
}


def event_from_dict(d: dict[str, Any]) -> Event:
    cls = EVENT_TYPES.get(str(d.get("type", "")))
    if cls is None:
        raise ValueError(f"unknown event type {d.get('type')!r}")
    return cls.from_dict(d)
