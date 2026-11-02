"""Stage 11 - reconstruction: render fused events back to audio (MIDI-like synthesis or a
SoundFont renderer) and place original FX clips. Every track's source is recorded."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mrep.adapters.base import AdapterUnavailable, RenderAdapter
from mrep.domain.tracks import Track, track_display_name
from mrep.io.audio import encode_mp3, ffmpeg_available, read_audio, write_audio
from mrep.io.json import write_json
from mrep.pipeline.classification import load_labels
from mrep.pipeline.fusion import load_fused
from mrep.pipeline.global_analysis import load_global
from mrep.pipeline.runner import PipelineContext, StageResult


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("reconstruction")
    cfg = ctx.stage_cfg("reconstruction")
    result = StageResult()
    notes, drums, fx = load_fused(ctx)
    labels = load_labels(ctx)
    glob = load_global(ctx)
    bpm = float(glob["tempo"].get("tempo") or 120.0)
    sr = int(ctx.config.get("audio", {}).get("target_sample_rate", 44100))
    duration = float(ctx.manifest.source.get("duration") or (max([n.offset for n in notes] + [d.time for d in drums] + [f.end for f in fx] + [1.0])))

    by_track: dict[str, list] = defaultdict(list)
    for n in notes:
        by_track[n.track].append(n)
    tracks: list[Track] = []
    for stem, ns in by_track.items():
        label = labels.get(stem)
        tracks.append(Track(name=track_display_name(stem, label) if stem != "unassigned" else "Unassigned", stem=stem, label=label, notes=ns))
    if drums:
        tracks.append(Track(name="Drums", stem="drums", label=labels.get("drums"), drums=drums, is_drum=True))
    fx_clips = []
    for ev in fx:
        if ev.clip_path and ctx.project.abs(ev.clip_path).exists():
            clip, csr = read_audio(ctx.project.abs(ev.clip_path))
            if csr != sr:
                from mrep.io.audio import resample

                clip = resample(clip, csr, sr)
            fx_clips.append((ev, clip))

    name = cfg.get("adapter", "additive")
    adapter_cfg = dict(cfg)
    try:
        adapter: RenderAdapter = ctx.adapter("render", name, adapter_cfg)  # type: ignore[assignment]
        adapter.ensure_models(download=False)
    except AdapterUnavailable as exc:
        fb = cfg.get("fallback_adapter", "additive")
        if not fb or fb == name:
            raise
        result.warnings.append(f"render/{name} unavailable ({exc}); falling back to {fb}")
        log.warning("render/%s unavailable, falling back to %s", name, fb)
        adapter = ctx.adapter("render", fb, adapter_cfg)  # type: ignore[assignment]
    result.adapters.append(adapter.info())
    midi_path = ctx.project.abs(ctx.manifest.artifact_path("midi:FULL_SONG") or "04_midi/FULL_SONG.mid")
    rres = adapter.render(tracks=tracks, sample_rate=sr, duration=duration, fx_clips=fx_clips, tempo_bpm=bpm, midi_path=str(midi_path) if midi_path.exists() else None)

    out_dir = ctx.project.reconstruction_dir
    wav = write_audio(out_dir / "reconstructed.wav", rres.mix, rres.sample_rate)
    result.artifacts["reconstructed_wav"] = ctx.rel(wav)
    if cfg.get("track_renders", True) and rres.tracks:
        for tname, audio in rres.tracks.items():
            p = write_audio(out_dir / "track_renders" / f"{tname}.wav", audio, rres.sample_rate)
            result.artifacts[f"render:{tname}"] = ctx.rel(p)
    mp3_setting = cfg.get("mp3", "auto")
    mp3_status = "skipped"
    if mp3_setting is True or (str(mp3_setting).lower() == "auto" and ffmpeg_available()):
        try:
            if encode_mp3(wav, out_dir / "reconstructed.mp3"):
                result.artifacts["reconstructed_mp3"] = ctx.rel(out_dir / "reconstructed.mp3")
                mp3_status = "ok"
            else:
                mp3_status = "ffmpeg not found"
        except Exception as exc:  # noqa: BLE001
            mp3_status = f"failed: {exc}"
            result.warnings.append(f"mp3 encoding failed: {exc}")
    elif mp3_setting is True:
        mp3_status = "ffmpeg not found"
    meta = {
        "adapter": adapter.info().to_dict(),
        "sources": rres.sources,
        "tracks": [{"name": t.name, "stem": t.stem, "notes": len(t.notes), "drums": len(t.drums), "source": rres.sources.get(t.name, "midi_rendered")} for t in tracks],
        "fx_clips": [{"class": ev.fx_class, "clip": ev.clip_path, "start": ev.start, "end": ev.end, "source": "original_audio_clip"} for ev, _ in fx_clips],
        "mp3": mp3_status,
        "soundfont": cfg.get("soundfont"),
        "sample_rate": rres.sample_rate,
        "peak": round(float(np.max(np.abs(rres.mix))) if rres.mix.size else 0.0, 4),
    }
    write_json(out_dir / "reconstruction.json", meta)
    result.artifacts["reconstruction_meta"] = ctx.rel(out_dir / "reconstruction.json")
    result.summary = {"tracks": len(tracks), "fx_clips": len(fx_clips), "mp3": mp3_status, "adapter": adapter.name}
    log.info("rendered %d tracks + %d fx clips with %s", len(tracks), len(fx_clips), adapter.name)
    return result
