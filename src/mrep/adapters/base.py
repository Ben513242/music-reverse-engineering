"""Adapter interfaces. The pipeline only talks to these; models are plugged in behind them.

Every adapter:
* is constructed with ``(config: dict, device: str)``,
* never downloads anything implicitly (``ensure_models(download=False)`` must raise instead),
* reports its ``AdapterInfo`` (name, model, version) for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mrep.domain.events import DrumEvent, FXEvent, NoteEvent
from mrep.domain.labels import SoundLabel
from mrep.domain.provenance import Provenance


class AdapterError(RuntimeError):
    """Raised by an adapter when inference fails."""


class AdapterUnavailable(AdapterError):
    """Raised when an adapter cannot run here (missing dependency, model, binary, VRAM ...)."""


@dataclass
class AdapterInfo:
    kind: str
    name: str
    model: str | None = None
    model_version: str | None = None
    device: str = "cpu"
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_models: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "model": self.model,
            "model_version": self.model_version,
            "device": self.device,
        }


class BaseAdapter:
    kind: str = "base"
    name: str = "base"
    model: str | None = None
    model_version: str | None = None
    requires_models: bool = False

    def __init__(self, config: dict[str, Any] | None = None, device: str = "cpu"):
        self.config = dict(config or {})
        self.device = device

    # -- lifecycle ---------------------------------------------------------------------
    def available(self) -> bool:
        """Return True when the adapter can run in this environment (deps, binaries...)."""
        return True

    def ensure_models(self, download: bool = False) -> None:
        """Make model files available. Must raise ``AdapterUnavailable`` instead of
        downloading when ``download`` is False."""
        return None

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            kind=self.kind,
            name=self.name,
            model=self.model,
            model_version=self.model_version,
            device=self.device,
            parameters=dict(self.config),
            requires_models=self.requires_models,
        )

    def provenance(self, stage: str, run_id: str, input_name: str, params_hash: str = "") -> Provenance:
        return Provenance(
            stage=stage,
            adapter=self.name,
            model=self.model,
            model_version=self.model_version,
            run_id=run_id,
            input=input_name,
            parameters_hash=params_hash,
        )


# ------------------------------------------------------------------------------ results
@dataclass
class SeparationResult:
    stems: dict[str, np.ndarray]  # name -> (n, ch) float32
    sample_rate: int
    confidence: dict[str, float | None] = field(default_factory=dict)
    confidence_method: str = "unavailable"


@dataclass
class DiscoveryResult:
    probabilities: dict[str, float | None]
    confidence_method: str = "unavailable"


@dataclass
class ClassificationResult:
    label: SoundLabel
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class MIRResult:
    tempo_bpm: float | None = None
    tempo_confidence: float | None = None
    tempo_method: str = "unavailable"
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    beats_per_bar: int = 4
    time_signature: str = "4/4"
    time_signature_confidence: float | None = None
    time_signature_method: str = "assumed"
    key_events: list[dict[str, Any]] = field(default_factory=list)  # {start, end, tonic, mode, confidence}
    sections: list[dict[str, Any]] = field(default_factory=list)  # {start, end, label, confidence}
    loudness: dict[str, Any] = field(default_factory=dict)
    dynamics: list[dict[str, Any]] = field(default_factory=list)  # {start, end, rms_db, dynamic}
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    mix: np.ndarray
    sample_rate: int
    tracks: dict[str, np.ndarray] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # track -> midi_rendered | original_audio_clip


# ------------------------------------------------------------------------------ interfaces
class SeparationAdapter(BaseAdapter):
    kind = "separation"
    #: stems this adapter can produce (documentation/gating only)
    outputs: tuple[str, ...] = ()

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        raise NotImplementedError


class DiscoveryAdapter(BaseAdapter):
    kind = "discovery"

    def discover(self, audio: np.ndarray, sr: int) -> DiscoveryResult:
        raise NotImplementedError


class ClassificationAdapter(BaseAdapter):
    kind = "classification"

    def classify(self, audio: np.ndarray, sr: int, *, stem_name: str, source_hint: str, provenance: Provenance) -> ClassificationResult:
        raise NotImplementedError


class TranscriptionAdapter(BaseAdapter):
    kind = "transcription"

    def transcribe(self, audio: np.ndarray, sr: int, *, track: str, instrument: str, provenance: Provenance) -> list[NoteEvent]:
        raise NotImplementedError


class DrumTranscriptionAdapter(BaseAdapter):
    kind = "drum_transcription"

    def transcribe_drums(self, audio: np.ndarray, sr: int, *, track: str, class_hint: str | None, provenance: Provenance) -> list[DrumEvent]:
        raise NotImplementedError


class MIRAdapter(BaseAdapter):
    kind = "mir"

    def analyze(self, audio: np.ndarray, sr: int) -> MIRResult:
        raise NotImplementedError


class FXAdapter(BaseAdapter):
    kind = "fx"

    def detect(self, audio: np.ndarray, sr: int, *, provenance: Provenance) -> list[FXEvent]:
        raise NotImplementedError


class RenderAdapter(BaseAdapter):
    kind = "render"

    def render(
        self,
        *,
        tracks: list[Any],
        sample_rate: int,
        duration: float,
        fx_clips: list[tuple[FXEvent, np.ndarray]],
        tempo_bpm: float,
        midi_path: str | None = None,
    ) -> RenderResult:
        raise NotImplementedError
