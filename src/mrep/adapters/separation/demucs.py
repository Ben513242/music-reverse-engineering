"""Demucs (hybrid transformer) source separation adapter - optional, explicit opt-in.

Requires ``pip install torch demucs``. Model weights live in the torch hub cache and are only
fetched by ``mrep models download separation demucs`` (``ensure_models(download=True)``);
running the pipeline never downloads anything implicitly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from mrep.adapters.base import AdapterError, AdapterUnavailable, SeparationAdapter, SeparationResult
from mrep.adapters.registry import register


@register("separation", "demucs")
class DemucsSeparation(SeparationAdapter):
    model = "htdemucs"
    model_version = None
    requires_models = True
    outputs = ("drums", "bass", "other", "vocals")

    def __init__(self, config=None, device: str = "cpu"):
        super().__init__(config, device)
        self.model = str(self.config.get("demucs_model", "htdemucs"))
        self._model = None

    def available(self) -> bool:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("demucs") is not None

    def _checkpoint_dir(self) -> Path:
        import torch

        return Path(torch.hub.get_dir()) / "checkpoints"

    def ensure_models(self, download: bool = False) -> None:
        if not self.available():
            raise AdapterUnavailable("demucs adapter needs `pip install torch demucs`")
        ckpt = self._checkpoint_dir()
        has_weights = ckpt.is_dir() and any(ckpt.glob("*.th"))
        if not has_weights and not download:
            raise AdapterUnavailable(f"no demucs weights in {ckpt}; run `mrep models download separation demucs`")
        if download or self._model is None:
            self._load()

    def _load(self):
        if self._model is None:
            from demucs.pretrained import get_model

            self._model = get_model(self.model)
            self._model.eval()
            self.model_version = getattr(self._model, "signature", None) or getattr(self._model, "name", None)
            self.outputs = tuple(self._model.sources)
        return self._model

    def separate(self, audio: np.ndarray, sr: int, *, source_hint: str | None = None) -> SeparationResult:
        self.ensure_models(download=bool(self.config.get("allow_download", False)))
        import torch
        from demucs.apply import apply_model

        model = self._load()
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        wav = torch.from_numpy(x.T.copy())  # (channels, samples)
        model_sr = int(model.samplerate)
        if sr != model_sr:
            from mrep.utils.dsp import resample

            wav = torch.from_numpy(resample(x, sr, model_sr).T.copy())
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
        device = self.device if self.device in ("cpu", "cuda", "mps") else "cpu"
        try:
            with torch.no_grad():
                out = apply_model(model, wav_n[None], device=device, shifts=int(self.config.get("shifts", 0)), split=True, overlap=float(self.config.get("overlap", 0.25)), progress=False)[0]
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"demucs inference failed: {exc}") from exc
        out = out * (ref.std() + 1e-8) + ref.mean()
        stems: dict[str, np.ndarray] = {}
        for name, src in zip(model.sources, out):
            y = src.cpu().numpy().T.astype(np.float32)
            if sr != model_sr:
                from mrep.utils.dsp import resample

                y = resample(y, model_sr, sr)
            stems[name] = y[: len(audio)]
        return SeparationResult(stems=stems, sample_rate=sr, confidence={k: None for k in stems}, confidence_method="unavailable")
