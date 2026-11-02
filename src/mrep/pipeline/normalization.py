"""Stage 2 - normalization: canonical internal audio (float32 stereo at the target rate)
plus an ``analysis_mono.wav`` for analysis models. The applied gain is recorded."""

from __future__ import annotations

from pathlib import Path

from mrep.io.audio import peak_normalize, read_audio, resample, to_mono, to_stereo, write_audio
from mrep.io.json import write_json
from mrep.pipeline.runner import PipelineContext, StageResult

CANONICAL = "audio.wav"
MONO = "analysis_mono.wav"


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("normalization")
    acfg = ctx.config.get("audio", {})
    target_sr = int(acfg.get("target_sample_rate", 44100))
    src = Path(ctx.manifest.source["path"])
    data, sr = read_audio(src)
    data = to_stereo(data) if int(acfg.get("channels", 2)) == 2 else to_mono(data)[:, None]
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    gain_db = 0.0
    target_dbfs = acfg.get("peak_normalize_dbfs")
    if target_dbfs is not None:
        data, gain_db = peak_normalize(data, float(target_dbfs))
    out_dir = ctx.project.normalized_dir
    canonical = write_audio(out_dir / CANONICAL, data, target_sr)
    artifacts = {"normalized_audio": ctx.rel(canonical)}
    if acfg.get("write_analysis_mono", True):
        mono = write_audio(out_dir / MONO, to_mono(data), target_sr)
        artifacts["analysis_mono"] = ctx.rel(mono)
    meta = {
        "sample_rate": target_sr,
        "channels": int(data.shape[1]),
        "dtype": "float32",
        "duration": round(len(data) / target_sr, 4),
        "source_sample_rate": sr,
        "gain_db": round(gain_db, 3),
        "peak_target_dbfs": target_dbfs,
    }
    write_json(out_dir / "normalization.json", meta)
    artifacts["normalization_meta"] = ctx.rel(out_dir / "normalization.json")
    ctx.cache["normalized"] = (data, target_sr)
    log.info("normalized to %d Hz stereo float32 (gain %.2f dB)", target_sr, gain_db)
    return StageResult(artifacts=artifacts, summary=meta)


def load_normalized(ctx: PipelineContext):
    """Return (stereo_audio, sr) from cache or disk."""
    if "normalized" in ctx.cache:
        return ctx.cache["normalized"]
    path = ctx.project.abs(ctx.manifest.artifact_path("normalized_audio") or f"01_normalized/{CANONICAL}")
    data, sr = read_audio(path)
    ctx.cache["normalized"] = (data, sr)
    return data, sr
