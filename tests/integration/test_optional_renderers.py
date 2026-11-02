"""Integration tests that need external binaries / real models. Skipped unless available."""

from __future__ import annotations

import shutil

import pytest

from mrep.adapters import create_adapter
from mrep.adapters.base import AdapterUnavailable

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_fluidsynth_adapter_requires_explicit_soundfont():
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not installed")
    adapter = create_adapter("render", "fluidsynth", {"soundfont": None}, "cpu")
    with pytest.raises(AdapterUnavailable):
        adapter.ensure_models(download=False)
