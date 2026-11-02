"""Stages 4-6 - source discovery, broad separation and gated specialist separation.

Specialists only run when discovery says the source is likely present (or when forced).
A failing specialist is recorded as an error and the stage ends PARTIAL - never fatal.
"""

from __future__ import annotations

import numpy as np

from mrep.adapters.base import AdapterUnavailable, SeparationAdapter, SeparationResult
from mrep.domain.tracks import Stem
from mrep.io.audio import write_audio
from mrep.io.json import read_json, write_json
from mrep.pipeline.normalization import load_normalized
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.utils.dsp import rms

STEM_INDEX = "stems.json"


def separate_chunked(adapter: SeparationAdapter, audio: np.ndarray, sr: int, chunk_s: float, overlap_s: float, source_hint: str | None = None) -> SeparationResult:
    """Run an adapter on overlapping chunks and cross-fade (for low-memory profiles)."""
    n = len(audio)
    chunk = int(chunk_s * sr)
    if chunk_s <= 0 or n <= chunk:
        return adapter.separate(audio, sr, source_hint=source_hint)
    overlap = int(overlap_s * sr)
    hop = max(chunk - overlap, 1)
    acc: dict[str, np.ndarray] = {}
    weight = np.zeros(n, dtype=np.float32)
    conf: dict[str, float | None] = {}
    method = "unavailable"
    start = 0
    while start < n:
        end = min(n, start + chunk)
        res = adapter.separate(audio[start:end], sr, source_hint=source_hint)
        method = res.confidence_method
        seg_len = end - start
        w = np.ones(seg_len, dtype=np.float32)
        fade = min(overlap, seg_len // 2)
        if fade > 0:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            if start > 0:
                w[:fade] = ramp
            if end < n:
                w[-fade:] = ramp[::-1]
        for name, stem in res.stems.items():
            if name not in acc:
                acc[name] = np.zeros_like(audio)
            acc[name][start:end] += stem[:seg_len] * w[:, None]
            conf[name] = res.confidence.get(name)
        weight[start:end] += w
        if end >= n:
            break
        start += hop
    weight = np.maximum(weight, 1e-6)[:, None]
    return SeparationResult(stems={k: (v / weight).astype(np.float32) for k, v in acc.items()}, sample_rate=sr, confidence=conf, confidence_method=method)


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("separation")
    cfg = ctx.stage_cfg("separation")
    result = StageResult()
    data, sr = load_normalized(ctx)
    mix_rms = rms(data) + 1e-9
    stems_dir = ctx.project.stems_dir
    stems: list[Stem] = []

    # ---- discovery ------------------------------------------------------------------
    probs: dict[str, float | None] = {}
    disc_method = "unavailable"
    disc_name = cfg.get("discovery_adapter")
    if disc_name:
        try:
            disc = ctx.adapter("discovery", disc_name, cfg)
            result.adapters.append(disc.info())
            dres = disc.discover(data, sr)
            probs, disc_method = dres.probabilities, dres.confidence_method
            log.info("discovery: %s", {k: (round(v, 2) if v is not None else None) for k, v in probs.items()})
        except Exception as exc:  # noqa: BLE001
            result.add_error("discovery", exc)
            log.warning("discovery failed: %s", exc)
    write_json(ctx.project.analysis_dir / "instruments.json", {"discovery": {"probabilities": probs, "confidence_method": disc_method, "adapter": disc_name}})
    result.artifacts["instruments"] = ctx.rel(ctx.project.analysis_dir / "instruments.json")

    # ---- broad separation -------------------------------------------------------------
    broad_name = cfg.get("broad_adapter", "hpss")
    try:
        broad = ctx.adapter("separation", broad_name, cfg)
    except AdapterUnavailable as exc:
        fb = cfg.get("fallback_adapter")
        if not fb:
            raise
        result.warnings.append(f"separation/{broad_name} unavailable ({exc}); using fallback {fb}")
        broad = ctx.adapter("separation", fb, cfg)
    result.adapters.append(broad.info())
    bres = separate_chunked(broad, data, sr, float(cfg.get("chunk_seconds", 0) or 0), float(cfg.get("chunk_overlap_seconds", 2.0)))
    prov = ctx.provenance("separation", broad, "normalized_audio")
    for name, audio in bres.stems.items():
        path = write_audio(stems_dir / f"{name}.wav", audio, sr) if cfg.get("write_stems", True) else stems_dir / f"{name}.wav"
        stems.append(Stem(name=name, path=ctx.rel(path), source_hint=name, parent=None, level="broad", sample_rate=sr, channels=int(np.atleast_2d(audio).shape[-1]) if audio.ndim > 1 else 1, duration=round(len(audio) / sr, 4), energy_ratio=round(float(rms(audio) / mix_rms), 4), confidence=bres.confidence.get(name), confidence_method=bres.confidence_method, provenance=prov))
        ctx.cache[f"stem:{name}"] = audio
    log.info("broad stems: %s", list(bres.stems))

    # ---- specialists (gated) ----------------------------------------------------------
    threshold = float(cfg.get("specialist_threshold", 0.6))
    forced = set(cfg.get("force_specialists", []) or [])
    for source, adapter_name in (cfg.get("specialists") or {}).items():
        prob = probs.get(source)
        present_as_stem = source in bres.stems
        if source not in forced and not present_as_stem and (prob is None or prob < threshold):
            result.warnings.append(f"specialist {source}/{adapter_name} skipped (discovery {prob})")
            continue
        if source not in forced and present_as_stem and prob is not None and prob < threshold and float(rms(bres.stems[source]) / mix_rms) < 0.05:
            result.warnings.append(f"specialist {source}/{adapter_name} skipped (stem energy too low)")
            continue
        try:
            spec = ctx.adapter("separation", adapter_name, cfg)
            result.adapters.append(spec.info())
            src_audio = bres.stems.get(source, data)
            sres = separate_chunked(spec, src_audio, sr, float(cfg.get("chunk_seconds", 0) or 0), float(cfg.get("chunk_overlap_seconds", 2.0)), source_hint=source)
            sprov = ctx.provenance("separation", spec, f"stem:{source}" if source in bres.stems else "normalized_audio")
            for comp, audio in sres.stems.items():
                name = f"{source}/{comp}"
                path = write_audio(stems_dir / source / f"{comp}.wav", audio, sr)
                stems.append(Stem(name=name, path=ctx.rel(path), source_hint=comp, parent=source if source in bres.stems else None, level="specialist", sample_rate=sr, channels=2 if audio.ndim > 1 else 1, duration=round(len(audio) / sr, 4), energy_ratio=round(float(rms(audio) / mix_rms), 4), confidence=sres.confidence.get(comp), confidence_method=sres.confidence_method, provenance=sprov))
                ctx.cache[f"stem:{name}"] = audio
            log.info("specialist %s -> %s", source, list(sres.stems))
        except Exception as exc:  # noqa: BLE001
            result.add_error(f"specialist:{source}:{adapter_name}", exc)
            log.warning("specialist %s failed: %s", source, exc)

    index = {"sample_rate": sr, "stems": [s.to_dict() for s in stems], "discovery": probs}
    write_json(stems_dir / STEM_INDEX, index)
    result.artifacts["stem_index"] = ctx.rel(stems_dir / STEM_INDEX)
    for s in stems:
        result.artifacts[f"stem:{s.name}"] = s.path
    result.summary = {"broad": list(bres.stems), "stems": [s.name for s in stems], "specialists_failed": [e["where"] for e in result.errors]}
    result.confidence = {"discovery": probs, "stems": {s.name: s.confidence for s in stems}}
    return result


def load_stems(ctx: PipelineContext) -> list[Stem]:
    index = read_json(ctx.project.stems_dir / STEM_INDEX)
    return [Stem.from_dict(d) for d in index["stems"]]


def load_stem_audio(ctx: PipelineContext, stem: Stem) -> tuple[np.ndarray, int]:
    key = f"stem:{stem.name}"
    if key in ctx.cache:
        return ctx.cache[key], stem.sample_rate
    from mrep.io.audio import read_audio

    data, sr = read_audio(ctx.project.abs(stem.path))
    ctx.cache[key] = data
    return data, sr
