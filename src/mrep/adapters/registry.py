"""Adapter registry: ``(kind, name) -> class``. Built-in adapters register on import."""

from __future__ import annotations

import importlib
from typing import Any

from mrep.adapters.base import AdapterUnavailable, BaseAdapter

_REGISTRY: dict[tuple[str, str], type[BaseAdapter]] = {}
_BUILTIN_MODULES = (
    "mrep.adapters.separation.fake",
    "mrep.adapters.separation.hpss",
    "mrep.adapters.separation.drum_bandsplit",
    "mrep.adapters.separation.discovery",
    "mrep.adapters.separation.demucs",
    "mrep.adapters.classification.fake",
    "mrep.adapters.classification.heuristic",
    "mrep.adapters.transcription.fake",
    "mrep.adapters.transcription.onset_drums",
    "mrep.adapters.transcription.spectral_peaks",
    "mrep.adapters.transcription.basic_pitch",
    "mrep.adapters.sound_analysis.fake",
    "mrep.adapters.sound_analysis.dsp",
    "mrep.adapters.mir.fake",
    "mrep.adapters.mir.dsp",
    "mrep.adapters.fx.fake",
    "mrep.adapters.fx.heuristic",
    "mrep.adapters.render.fake",
    "mrep.adapters.render.additive",
    "mrep.adapters.render.fluidsynth",
)
_loaded = False


def register(kind: str, name: str):
    def deco(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        cls.kind = kind
        cls.name = name
        _REGISTRY[(kind, name)] = cls
        return cls

    return deco


def load_builtin() -> None:
    global _loaded
    if _loaded:
        return
    for mod in _BUILTIN_MODULES:
        importlib.import_module(mod)
    _loaded = True


def get_adapter_class(kind: str, name: str) -> type[BaseAdapter]:
    load_builtin()
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        if ":" in name:  # fully qualified "package.module:ClassName" plug-in
            mod_name, cls_name = name.split(":", 1)
            cls = getattr(importlib.import_module(mod_name), cls_name)
            _REGISTRY[(kind, name)] = cls
            return cls
        available = sorted(n for k, n in _REGISTRY if k == kind)
        raise AdapterUnavailable(f"no adapter {name!r} registered for kind {kind!r}; available: {available}") from None


def create_adapter(kind: str, name: str, config: dict[str, Any] | None = None, device: str = "cpu") -> BaseAdapter:
    cls = get_adapter_class(kind, name)
    adapter = cls(config=config or {}, device=device)
    if not adapter.available():
        raise AdapterUnavailable(f"adapter {kind}/{name} is not available in this environment")
    return adapter


def list_adapters(kind: str | None = None) -> list[tuple[str, str]]:
    load_builtin()
    return sorted(k for k in _REGISTRY if kind is None or k[0] == kind)
