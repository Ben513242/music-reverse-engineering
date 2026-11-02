"""Spotify Basic Pitch AMT adapter - optional, explicit opt-in (``pip install basic-pitch``).

The ICASSP 2022 model ships inside the package, so no download step is needed; the heavy
dependency (TensorFlow / ONNX runtime) is imported lazily inside the method.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from mrep.adapters.base import AdapterError, AdapterUnavailable, TranscriptionAdapter
from mrep.adapters.registry import register
from mrep.domain.events import NoteEvent
from mrep.domain.provenance import Provenance
from mrep.utils.music import amplitude_to_velocity


@register("transcription", "basic_pitch")
class BasicPitchAMT(TranscriptionAdapter):
    model = "basic_pitch_icassp_2022"
    model_version = None
    requires_models = False

    def available(self) -> bool:
        return importlib.util.find_spec("basic_pitch") is not None

    def ensure_models(self, download: bool = False) -> None:
        if not self.available():
            raise AdapterUnavailable("basic_pitch adapter needs `pip install basic-pitch`")

    def transcribe(self, audio: np.ndarray, sr: int, *, track: str, instrument: str, provenance: Provenance) -> list[NoteEvent]:
        self.ensure_models()
        import soundfile as sf
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        try:
            import basic_pitch

            self.model_version = getattr(basic_pitch, "__version__", None)
        except Exception:  # noqa: BLE001
            pass
        x = np.asarray(audio, dtype=np.float32)
        mono = x if x.ndim == 1 else x.mean(axis=1)
        cfg = self.config
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stem.wav"
            sf.write(str(path), mono, sr)
            try:
                _model_output, _midi, note_events = predict(
                    str(path),
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                    onset_threshold=float(cfg.get("onset_threshold", 0.5)),
                    frame_threshold=float(cfg.get("frame_threshold", 0.3)),
                    minimum_note_length=float(cfg.get("min_note_seconds", 0.06)) * 1000.0,
                    minimum_frequency=float(cfg.get("min_frequency_hz", 32.7)),
                    maximum_frequency=float(cfg.get("max_frequency_hz", 4186.0)),
                    multiple_pitch_bends=bool(cfg.get("multiple_pitch_bends", False)),
                    melodia_trick=bool(cfg.get("melodia_trick", True)),
                )
            except Exception as exc:  # noqa: BLE001
                raise AdapterError(f"basic_pitch inference failed: {exc}") from exc
        notes: list[NoteEvent] = []
        for ev in note_events:
            start, end, pitch, amplitude = ev[0], ev[1], int(ev[2]), float(ev[3])
            bends = ev[4] if len(ev) > 4 else None
            pb = None
            if bends is not None and len(bends):
                step = max(1, len(bends) // 32)
                frame_dur = (end - start) / max(1, len(bends))
                pb = [[round(i * frame_dur, 4), round(float(b) / 3.0, 3)] for i, b in enumerate(bends) if i % step == 0]  # basic-pitch bends are in 1/3-semitone bins
            notes.append(
                NoteEvent(
                    midi_note=pitch,
                    onset=round(float(start), 4),
                    offset=round(float(end), 4),
                    velocity=amplitude_to_velocity(amplitude),
                    pitch_bend=pb,
                    track=track,
                    instrument=instrument,
                    confidence=round(float(np.clip(amplitude, 0.0, 1.0)), 4),
                    confidence_method="model_amplitude",
                    provenance=provenance,
                )
            )
        notes.sort(key=lambda n: (n.onset, n.midi_note))
        return notes
