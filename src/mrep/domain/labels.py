"""Label taxonomies and the multi-label ``SoundLabel``.

A sound is described along four *independent* axes (never a single "instrument" field):

* source  - what is making the sound (piano, synthesizer, kick ...)
* musical role - what it does in the song (lead, bassline, pad, arpeggio ...)
* frequency role - where it lives in the spectrum (sub_bass ... air)
* sound design - timbral character (warm, bright, noise_based ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mrep.domain.provenance import Provenance

# --------------------------------------------------------------------------- drums / fx
DRUM_CLASSES: tuple[str, ...] = (
    "kick",
    "snare",
    "closed_hat",
    "open_hat",
    "pedal_hat",
    "clap",
    "rimshot",
    "high_tom",
    "mid_tom",
    "low_tom",
    "crash",
    "ride",
    "shaker",
    "tambourine",
    "cowbell",
    "conga",
    "other_percussion",
)
DRUM_CORE: tuple[str, ...] = ("kick", "snare", "closed_hat", "open_hat")

FX_CLASSES: tuple[str, ...] = (
    "riser",
    "downlifter",
    "white_noise",
    "impact",
    "glitch",
    "granular_texture",
    "atmosphere",
)

# --------------------------------------------------------------------------- sources
SOURCE_FAMILIES: frozenset[str] = frozenset(
    {
        "drums",
        "percussion",
        "bass",
        "synthesizer",
        "piano",
        "electric_piano",
        "organ",
        "guitar",
        "strings",
        "brass",
        "woodwinds",
        "harp",
        "vocal",
        "fx",
        "harmonic",  # pitched content whose family could not be resolved
        "unknown",
    }
)

SOURCE_SUBTYPES: dict[str, frozenset[str]] = {
    "drums": frozenset(DRUM_CLASSES),
    "percussion": frozenset({"shaker", "tambourine", "cowbell", "conga", "other_percussion"}),
    "bass": frozenset({"electric_bass", "acoustic_bass", "synth_bass", "808"}),
    "synthesizer": frozenset(
        {
            "synth_lead",
            "synth_pad",
            "synth_bass",
            "synth_arp",
            "synth_pluck",
            "synth_keys",
            "synth_bell",
            "synth_brass",
            "synth_strings",
            "synth_chord",
            "synth_drone",
            "synth_fx",
        }
    ),
    "guitar": frozenset({"acoustic_guitar", "electric_guitar"}),
    "strings": frozenset({"violin", "viola", "cello", "strings_section"}),
    "brass": frozenset({"trumpet", "trombone", "french_horn", "brass_section"}),
    "woodwinds": frozenset({"flute", "clarinet", "saxophone"}),
    "vocal": frozenset({"lead_vocal", "backing_vocal", "vocal_chop"}),
    "fx": frozenset(FX_CLASSES),
}

MUSICAL_ROLES: frozenset[str] = frozenset(
    {
        "lead",
        "melody",
        "bassline",
        "pad",
        "arpeggio",
        "chord",
        "pluck",
        "drone",
        "rhythm",
        "percussion",
        "fx",
        "texture",
        "atmosphere",
        "vocal",
    }
)

FREQUENCY_ROLES: tuple[str, ...] = ("sub_bass", "low_bass", "low_mid", "mid", "high", "air")
FREQUENCY_BANDS_HZ: dict[str, tuple[float, float]] = {
    "sub_bass": (20.0, 60.0),
    "low_bass": (60.0, 250.0),
    "low_mid": (250.0, 800.0),
    "mid": (800.0, 3000.0),
    "high": (3000.0, 10000.0),
    "air": (10000.0, 20000.0),
}

SOUND_DESIGN: frozenset[str] = frozenset(
    {
        "warm",
        "bright",
        "dark",
        "analog_style",
        "digital",
        "pluck",
        "bell_like",
        "noise_based",
        "granular",
        "short_attack",
        "slow_attack",
        "short_decay",
        "long_release",
        "distorted",
        "wide_stereo",
        "mono",
        "supersaw_like",
        "fm_like",
        "subtractive",
    }
)

# Families whose stems are candidates for pitched transcription (AMT).
PITCHED_FAMILIES: frozenset[str] = frozenset(
    {
        "bass",
        "synthesizer",
        "piano",
        "electric_piano",
        "organ",
        "guitar",
        "strings",
        "brass",
        "woodwinds",
        "harp",
        "vocal",
        "harmonic",
    }
)
# Families that must never be forced into conventional notation.
NON_SCORE_FAMILIES: frozenset[str] = frozenset({"fx"})


def validate_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be in [0, 1] or None, got {value}")
    return value


@dataclass
class Candidate:
    """A possible label with its probability (``None`` when the model gives none)."""

    label: str
    probability: float | None = None
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "probability": self.probability, "similarity": self.similarity}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        return cls(str(d["label"]), d.get("probability"), d.get("similarity"))


@dataclass
class SoundLabel:
    """Multi-label description of a stem / track. Axes are independent, not exclusive."""

    source_family: str = "unknown"
    source_subtype: str | None = None
    source_candidates: list[Candidate] = field(default_factory=list)
    musical_role: list[str] = field(default_factory=list)
    frequency_role: list[str] = field(default_factory=list)
    sound_design: list[str] = field(default_factory=list)
    confidence: float | None = None
    confidence_method: str = "unavailable"
    provenance: Provenance | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = validate_confidence(self.confidence)
        if self.confidence is None:
            self.confidence_method = self.confidence_method or "unavailable"

    # ---- taxonomy checks ---------------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of taxonomy problems (empty when the label is well-formed)."""
        problems: list[str] = []
        if self.source_family not in SOURCE_FAMILIES:
            problems.append(f"unknown source_family {self.source_family!r}")
        if self.source_subtype is not None:
            allowed = SOURCE_SUBTYPES.get(self.source_family, frozenset())
            if self.source_subtype not in allowed:
                problems.append(
                    f"subtype {self.source_subtype!r} not valid for family {self.source_family!r}"
                )
        for role in self.musical_role:
            if role not in MUSICAL_ROLES:
                problems.append(f"unknown musical_role {role!r}")
        for fr in self.frequency_role:
            if fr not in FREQUENCY_ROLES:
                problems.append(f"unknown frequency_role {fr!r}")
        for sd in self.sound_design:
            if sd not in SOUND_DESIGN:
                problems.append(f"unknown sound_design {sd!r}")
        return problems

    @property
    def is_pitched(self) -> bool:
        return self.source_family in PITCHED_FAMILIES

    @property
    def is_fx(self) -> bool:
        return self.source_family in NON_SCORE_FAMILIES or "fx" in self.musical_role

    @property
    def display_name(self) -> str:
        if self.source_subtype:
            return self.source_subtype
        return self.source_family

    # ---- serialisation -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_family": self.source_family,
            "source_subtype": self.source_subtype,
            "source_candidates": [c.to_dict() for c in self.source_candidates],
            "musical_role": list(self.musical_role),
            "frequency_role": list(self.frequency_role),
            "sound_design": list(self.sound_design),
            "confidence": self.confidence,
            "confidence_method": self.confidence_method,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SoundLabel:
        return cls(
            source_family=str(d.get("source_family", "unknown")),
            source_subtype=d.get("source_subtype"),
            source_candidates=[Candidate.from_dict(c) for c in d.get("source_candidates", [])],
            musical_role=list(d.get("musical_role", [])),
            frequency_role=list(d.get("frequency_role", [])),
            sound_design=list(d.get("sound_design", [])),
            confidence=d.get("confidence"),
            confidence_method=str(d.get("confidence_method", "unavailable")),
            provenance=Provenance.from_dict(d.get("provenance")),
            features=dict(d.get("features", {})),
        )
