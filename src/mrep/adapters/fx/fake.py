from __future__ import annotations

import numpy as np

from mrep.adapters.base import FXAdapter
from mrep.adapters.registry import register
from mrep.domain.events import FXEvent
from mrep.domain.provenance import Provenance


@register("fx", "fake")
class FakeFX(FXAdapter):
    model = "fake"
    model_version = "0"

    def detect(self, audio: np.ndarray, sr: int, *, provenance: Provenance) -> list[FXEvent]:
        duration = len(audio) / sr
        if duration < 3.0:
            return []
        return [FXEvent(fx_class="riser", start=max(0.0, duration - 3.0), end=duration - 1.0, confidence=0.8, confidence_method="fake_constant", provenance=provenance)]
