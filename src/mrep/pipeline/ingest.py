"""Stage 1 - ingest: validate the input, read metadata, hash it, create the project id.

The original audio file is never modified or copied.
"""

from __future__ import annotations

from pathlib import Path

from mrep.io.audio import FIRST_CLASS_FORMATS, SUPPORTED_FORMATS, AudioFormatError, audio_info
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.utils.hashing import sha256_file
from mrep.utils.timing import utc_now_iso


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("ingest")
    src = ctx.source_path or (Path(ctx.manifest.source["path"]) if ctx.manifest.source.get("path") else None)
    if src is None or not Path(src).is_file():
        raise FileNotFoundError(f"input audio not found: {src}")
    src = Path(src).resolve()
    ext = src.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise AudioFormatError(f"unsupported format {ext}; supported: {sorted(SUPPORTED_FORMATS)}")
    info = audio_info(src)
    if info["duration"] <= 0:
        raise AudioFormatError(f"{src.name} has no audio frames")
    digest = sha256_file(src)
    project_id = f"mrep_{digest[:16]}"
    ctx.manifest.project_id = project_id
    ctx.manifest.source = {
        "path": str(src),
        "filename": src.name,
        "format": ext.lstrip("."),
        "first_class_format": ext in FIRST_CLASS_FORMATS,
        "sha256": digest,
        "size_bytes": src.stat().st_size,
        "duration": round(info["duration"], 4),
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "subtype": info.get("subtype"),
        "decoder": info.get("decoder"),
        "ingested_at": utc_now_iso(),
    }
    log.info("ingested %s (%.1fs, %d Hz, %d ch)", src.name, info["duration"], info["sample_rate"], info["channels"])
    result = StageResult(summary={"project_id": project_id, "duration": info["duration"], "sample_rate": info["sample_rate"], "channels": info["channels"], "format": ext})
    if ext not in FIRST_CLASS_FORMATS:
        result.warnings.append(f"{ext} is supported via ffmpeg fallback, not a first-class format")
    return result
