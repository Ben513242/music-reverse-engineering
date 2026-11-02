from __future__ import annotations

from typing import Any

import numpy as np

from mrep.adapters.base import RenderAdapter, RenderResult
from mrep.adapters.registry import register
from mrep.domain.events import FXEvent


@register("render", "fake")
class FakeRender(RenderAdapter):
    """Renders a quiet click per note so that output files exist and are non-silent."""

    model = "fake"
    model_version = "0"

    def render(self, *, tracks: list[Any], sample_rate: int, duration: float, fx_clips: list[tuple[FXEvent, np.ndarray]], tempo_bpm: float, midi_path: str | None = None) -> RenderResult:
        n = int(duration * sample_rate) + sample_rate
        mix = np.zeros((n, 2), dtype=np.float32)
        outs: dict[str, np.ndarray] = {}
        sources: dict[str, str] = {}
        click = np.zeros(int(0.01 * sample_rate), dtype=np.float32)
        click[: len(click) // 2] = 0.2
        for tr in tracks:
            buf = np.zeros((n, 2), dtype=np.float32)
            times = [e.onset for e in tr.notes] + [e.time for e in tr.drums]
            for t in times:
                i = int(t * sample_rate)
                if i + len(click) < n:
                    buf[i : i + len(click), :] += click[:, None]
            outs[tr.name] = buf
            sources[tr.name] = "midi_rendered"
            mix += buf
        for ev, clip in fx_clips:
            i = int(ev.start * sample_rate)
            seg = clip[: max(0, n - i)]
            if seg.ndim == 1:
                seg = np.repeat(seg[:, None], 2, axis=1)
            mix[i : i + len(seg)] += seg[:, :2]
            sources[f"fx/{ev.fx_class}"] = "original_audio_clip"
        return RenderResult(mix=np.clip(mix, -1, 1), sample_rate=sample_rate, tracks=outs, sources=sources)
