from __future__ import annotations

import numpy as np

from mrep.adapters.base import AdapterError, DrumTranscriptionAdapter, TranscriptionAdapter
from mrep.adapters.registry import register
from mrep.domain.events import DrumEvent, NoteEvent
from mrep.domain.provenance import Provenance


@register("transcription", "fake")
class FakeTranscription(TranscriptionAdapter):
    """Emits a C-major arpeggio (quarter notes) across the whole duration."""

    model = "fake"
    model_version = "0"

    def transcribe(self, audio: np.ndarray, sr: int, *, track: str, instrument: str, provenance: Provenance) -> list[NoteEvent]:
        duration = len(audio) / sr
        pattern = self.config.get("pattern", [60, 64, 67, 72])
        step = float(self.config.get("step", 0.5))
        notes = []
        t = 0.0
        i = 0
        while t + step <= duration:
            notes.append(NoteEvent(midi_note=pattern[i % len(pattern)], onset=round(t, 4), offset=round(t + step * 0.9, 4), velocity=80, track=track, instrument=instrument, confidence=0.8, confidence_method="fake_constant", provenance=provenance))
            t += step
            i += 1
        return notes


@register("transcription", "failing")
class FailingTranscription(TranscriptionAdapter):
    model = "fake"
    model_version = "0"

    def transcribe(self, audio: np.ndarray, sr: int, *, track: str, instrument: str, provenance: Provenance) -> list[NoteEvent]:
        raise AdapterError("failing adapter: simulated transcription crash")


@register("drum_transcription", "fake")
class FakeDrumTranscription(DrumTranscriptionAdapter):
    """kick on beats 1/3, snare on 2/4, closed hat on 8ths at 120 bpm."""

    model = "fake"
    model_version = "0"

    def transcribe_drums(self, audio: np.ndarray, sr: int, *, track: str, class_hint: str | None, provenance: Provenance) -> list[DrumEvent]:
        duration = len(audio) / sr
        beat = 60.0 / float(self.config.get("bpm", 120.0))
        events = []
        t = 0.0
        i = 0
        while t < duration:
            cls = class_hint or ("kick" if i % 2 == 0 else "snare")
            events.append(DrumEvent(time=round(t, 4), drum_class=cls, velocity=100, track=track, confidence=0.9, confidence_method="fake_constant", provenance=provenance))
            if class_hint is None:
                events.append(DrumEvent(time=round(t, 4), drum_class="closed_hat", velocity=70, track=track, confidence=0.9, confidence_method="fake_constant", provenance=provenance))
                events.append(DrumEvent(time=round(t + beat / 2, 4), drum_class="closed_hat", velocity=60, track=track, confidence=0.9, confidence_method="fake_constant", provenance=provenance))
            t += beat
            i += 1
        return events
