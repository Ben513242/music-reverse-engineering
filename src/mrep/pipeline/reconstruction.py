"""Stage 11 - reconstruction: render fused events back to audio (MIDI-like synthesis or a
SoundFont renderer) and place original FX clips. Every track's source is recorded."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from mrep.adapters.base import AdapterUnavailable, RenderAdapter
from mrep.domain.tracks import Track, track_display_name
from mrep.io.audio import encode_mp3, ffmpeg_available, read_audio, write_audio
from mrep.io.daw import write_reaper_rpp, write_session_json
from mrep.io.json import read_jsonl, write_json
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

    # articulation decided by the scoring stage (humanisation) -> drives release / overlap in rendering
    score_path = ctx.project.score_dir / "score_events.jsonl"
    if score_path.exists():
        art = {s["raw_event_id"]: s.get("articulation", "normal") for s in read_jsonl(score_path)}
        for n in notes:
            if n.event_id in art:
                n.features["articulation"] = art[n.event_id]
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

    # ---- DAW export -------------------------------------------------------------------
    dcfg = cfg.get("daw_export", {}) or {}
    if dcfg.get("enabled", True):
        try:
            daw_dir = out_dir / "daw"
            daw_dir.mkdir(parents=True, exist_ok=True)
            from mrep.pipeline.separation import load_stems

            stems = [st for st in load_stems(ctx) if st.is_leaf or st.level == "broad"]
            audio_tracks = []
            for st in stems:
                rel = os.path.relpath(ctx.project.abs(st.path), daw_dir)
                audio_tracks.append({"name": track_display_name(st.name, labels.get(st.name)) if "/" not in st.name else st.name.replace("/", " / "), "file": rel.replace(os.sep, "/"), "volume": 1.0, "mute": False, "stem": st.name})
            audio_tracks.append({"name": "MREP reconstruction", "file": os.path.relpath(wav, daw_dir).replace(os.sep, "/"), "volume": 1.0, "mute": True, "stem": "reconstruction"})
            markers = [(float(sec["start"]), f"{sec['label']} ({sec.get('dynamic') or ''})".replace(" ()", "")) for sec in glob["structure"].get("sections", [])]
            tempo_changes = [(float(e["time"]), float(e["bpm"])) for e in glob["tempo"].get("tempo_events", [])]
            bpb = int(glob["tempo"].get("beats_per_bar", 4))
            rpp = write_reaper_rpp(daw_dir / f"{ctx.project.song_name}.rpp", bpm=bpm, time_signature=(bpb, 4), duration=duration, audio_tracks=audio_tracks, markers=markers, tempo_changes=tempo_changes if len(tempo_changes) > 1 else None, title=ctx.project.song_name)
            midi_files = sorted(p.name for p in ctx.project.midi_dir.glob("*.mid"))
            session = {
                "format": "mrep-session/1",
                "song": ctx.project.song_name,
                "sample_rate": rres.sample_rate,
                "duration": duration,
                "tempo_bpm": bpm,
                "tempo_events": glob["tempo"].get("tempo_events", []),
                "time_signature": glob["tempo"].get("time_signature"),
                "key_events": [{k: v for k, v in e.items() if k in ("start", "end", "key", "tonic", "mode", "confidence")} for e in __import__("mrep.io.json", fromlist=["read_json"]).read_json(ctx.project.analysis_dir / "key.json").get("key_events", [])] if (ctx.project.analysis_dir / "key.json").exists() else [],
                "markers": [{"time": t, "name": n} for t, n in markers],
                "audio_tracks": audio_tracks,
                "midi_files": [os.path.relpath(ctx.project.midi_dir / f, daw_dir).replace(os.sep, "/") for f in midi_files],
                "fx_clips": [{"class": ev.fx_class, "file": os.path.relpath(ctx.project.abs(ev.clip_path), daw_dir).replace(os.sep, "/"), "start": ev.start, "end": ev.end, "trigger_note": ev.trigger_note} for ev in fx if ev.clip_path],
                "labels": {k: {"source_family": v.source_family, "musical_role": v.musical_role} for k, v in labels.items()},
            }
            write_session_json(daw_dir / "session.json", session)
            result.artifacts["daw_reaper"] = ctx.rel(rpp)
            result.artifacts["daw_session"] = ctx.rel(daw_dir / "session.json")
            log.info("DAW export: %d audio tracks, %d markers", len(audio_tracks), len(markers))
        except Exception as exc:  # noqa: BLE001
            result.add_error("daw_export", exc)
    result.summary = {"tracks": len(tracks), "fx_clips": len(fx_clips), "mp3": mp3_status, "adapter": adapter.name}
    log.info("rendered %d tracks + %d fx clips with %s", len(tracks), len(fx_clips), adapter.name)
    return result
