"""Optional SoundFont rendering through the ``fluidsynth`` binary (explicit opt-in).

Requires: ``fluidsynth`` on PATH and ``stages.reconstruction.soundfont`` pointing to a .sf2
file that the user downloaded themselves. Nothing is fetched automatically.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mrep.adapters.base import AdapterError, AdapterUnavailable, RenderAdapter, RenderResult
from mrep.adapters.registry import register
from mrep.domain.events import FXEvent


@register("render", "fluidsynth")
class FluidsynthRender(RenderAdapter):
    model = "fluidsynth"
    model_version = None
    requires_models = True

    def available(self) -> bool:
        return shutil.which("fluidsynth") is not None

    def ensure_models(self, download: bool = False) -> None:
        sf2 = self.config.get("soundfont")
        if not sf2 or not Path(sf2).is_file():
            raise AdapterUnavailable(
                "fluidsynth adapter needs a SoundFont: set stages.reconstruction.soundfont to a .sf2 path "
                "(download one manually, e.g. FluidR3_GM.sf2). MREP never downloads it for you."
            )

    def render(self, *, tracks: list[Any], sample_rate: int, duration: float, fx_clips: list[tuple[FXEvent, np.ndarray]], tempo_bpm: float, midi_path: str | None = None) -> RenderResult:
        self.ensure_models(download=False)
        if not midi_path or not Path(midi_path).is_file():
            raise AdapterError("fluidsynth adapter needs the FULL_SONG.mid produced by the scoring stage")
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "render.wav"
            cmd = ["fluidsynth", "-ni", "-g", "0.8", "-r", str(sample_rate), "-F", str(out), str(self.config["soundfont"]), str(midi_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.is_file():
                raise AdapterError(f"fluidsynth failed: {proc.stderr.strip()[:500]}")
            data, sr = sf.read(str(out), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        n = max(len(data), int((duration + 1.5) * sr))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[: len(data)] = data[:, :2]
        sources = {tr.name: "midi_rendered_soundfont" for tr in tracks}
        for ev, clip in fx_clips:
            i = int(ev.start * sr)
            if i >= n:
                continue
            seg = clip[: n - i]
            if seg.ndim == 1:
                seg = np.repeat(seg[:, None], 2, axis=1)
            mix[i : i + len(seg)] += seg[:, :2]
            sources[f"fx/{ev.fx_class}_{ev.event_id[-4:]}"] = "original_audio_clip"
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.891:
            mix *= 0.891 / peak
        return RenderResult(mix=mix, sample_rate=sr, tracks={}, sources=sources)
