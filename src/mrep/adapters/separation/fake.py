"""Fake separation adapters for contract tests (deterministic, model-free)."""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import (
    AdapterError,
    DiscoveryAdapter,
    DiscoveryResult,
    SeparationAdapter,
    SeparationResult,
)
from mrep.adapters.registry import register


@register("separation", "fake")
class FakeSeparation(SeparationAdapter):
    """Splits the mix into energy-scaled copies: drums / bass / other."""

    model = "fake"
    model_version = "0"
    outputs = ("drums", "bass", "other")

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        x = np.asarray(audio, dtype=np.float32)
        weights = self.config.get("weights", {"drums": 0.3, "bass": 0.3, "other": 0.4})
        stems = {name: (x * w).astype(np.float32) for name, w in weights.items()}
        return SeparationResult(stems=stems, sample_rate=sr, confidence={n: 0.5 for n in stems}, confidence_method="fake_constant")


@register("separation", "fake_components")
class FakeComponentSeparation(SeparationAdapter):
    """Fake specialist: splits any stem into kick / snare / closed_hat copies."""

    model = "fake"
    model_version = "0"
    outputs = ("kick", "snare", "closed_hat")

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        x = np.asarray(audio, dtype=np.float32)
        return SeparationResult(stems={n: (x / 3).astype(np.float32) for n in self.outputs}, sample_rate=sr)


@register("separation", "failing")
class FailingSeparation(SeparationAdapter):
    """Always fails - used to test partial-success behaviour."""

    model = "fake"
    model_version = "0"

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        raise AdapterError("failing adapter: simulated model crash")


@register("discovery", "fake")
class FakeDiscovery(DiscoveryAdapter):
    model = "fake"
    model_version = "0"

    def discover(self, audio: np.ndarray, sr: int) -> DiscoveryResult:
        probs = self.config.get("probabilities", {"drums": 0.95, "bass": 0.9, "piano": 0.8, "synthesizer": 0.7, "strings": 0.2, "vocal": 0.05})
        return DiscoveryResult(probabilities=dict(probs), confidence_method="fake_constant")
