"""Device resolution. Never imports torch at module import time."""

from __future__ import annotations

import importlib.util


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_device(preference: str | None) -> str:
    """Resolve ``auto|cpu|cuda|mps`` to a concrete device. CPU is always a valid fallback."""
    pref = (preference or "auto").lower()
    if pref == "cpu" or not torch_available():
        return "cpu"
    try:
        import torch  # noqa: WPS433 (lazy import by design)
    except Exception:  # pragma: no cover - torch broken/incompatible
        return "cpu"
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if pref in ("auto", "mps") and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def gpu_memory_info(device: str) -> dict[str, float] | None:
    """Return VRAM usage in MB for CUDA devices, otherwise ``None``."""
    if not device.startswith("cuda") or not torch_available():
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 2**20,
            "reserved_mb": torch.cuda.memory_reserved() / 2**20,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        }
    except Exception:  # pragma: no cover
        return None
