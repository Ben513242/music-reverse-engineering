from __future__ import annotations

import numpy as np

from mrep.adapters.base import ClassificationAdapter, ClassificationResult
from mrep.adapters.registry import register
from mrep.domain.labels import SoundLabel
from mrep.domain.provenance import Provenance

_FAKE = {
    "drums": ("drums", None, ["rhythm", "percussion"], ["low_bass", "high"]),
    "bass": ("bass", "synth_bass", ["bassline"], ["sub_bass", "low_bass"]),
    "other": ("piano", None, ["chord"], ["mid"]),
}


@register("classification", "fake")
class FakeClassification(ClassificationAdapter):
    model = "fake"
    model_version = "0"

    def classify(self, audio: np.ndarray, sr: int, *, stem_name: str, source_hint: str, provenance: Provenance) -> ClassificationResult:
        key = source_hint if source_hint in _FAKE else stem_name.split("/")[-1]
        fam, sub, roles, freq = _FAKE.get(key, ("harmonic", None, ["chord"], ["mid"]))
        if "/" in stem_name:  # drum component
            fam, sub, roles, freq = "drums", stem_name.split("/")[-1], ["percussion"], ["low_bass"]
        label = SoundLabel(source_family=fam, source_subtype=sub, musical_role=roles, frequency_role=freq, sound_design=["digital"], confidence=0.9, confidence_method="fake_constant", provenance=provenance)
        return ClassificationResult(label=label, features={"fake": True})
