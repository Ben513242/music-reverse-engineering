"""Stage 10 - score conversion: MIDI (performance timing + quantised), MusicXML, FX triggers.

RAW events stay untouched in ``03_events``; quantised score events are written separately to
``05_score/score_events.jsonl`` and reference their raw event ids.
"""

from __future__ import annotations

from collections import defaultdict

from mrep.analysis.quantize import QuantizeSettings, quantize_drums, quantize_notes
from mrep.config import load_resource_yaml
from mrep.domain.events import DrumEvent, NoteEvent
from mrep.domain.labels import SoundLabel
from mrep.domain.tracks import track_display_name
from mrep.io.json import read_json, write_events, write_json
from mrep.io.midi import MidiNote, MidiTrackSpec, TempoMap, build_midi, write_midi
from mrep.io.musicxml import DIVISIONS, ScoreSpec, XmlHarmony, XmlNote, XmlPart, write_musicxml
from mrep.pipeline.classification import load_labels
from mrep.pipeline.fusion import load_fused
from mrep.pipeline.global_analysis import load_global
from mrep.pipeline.runner import PipelineContext, StageResult
from mrep.utils.music import key_fifths, pitch_class_from_name

SCORE_README = """MREP score folder
=================
*.musicxml   - one part per track + Full_Score.musicxml (all parts, chord symbols, dynamics)
score_events.jsonl - quantised note events (beats). Raw events live in 03_events/ and are
                     never modified; each score event references raw_event_id.

Render to PDF / SVG / PNG with MuseScore:
    mscore Full_Score.musicxml -o Full_Score.pdf
or import the files into any notation editor (Dorico, Sibelius, Finale, LilyPond via musicxml2ly).
"""


def program_for(label: SoundLabel | None, programs: dict) -> int:
    if label is None:
        return int(programs.get("default", 0))
    for key in (label.source_subtype, label.source_family):
        if key and key in programs:
            return int(programs[key])
    return int(programs.get("default", 0))


def unique_names(stems: list[str], labels: dict[str, SoundLabel]) -> dict[str, str]:
    names: dict[str, str] = {}
    seen: dict[str, int] = defaultdict(int)
    for stem in stems:
        base = track_display_name(stem, labels.get(stem))
        if stem == "unassigned":
            base = "Unassigned"
        seen[base] += 1
        names[stem] = base if seen[base] == 1 else f"{base}_{seen[base]}"
    return names


def run(ctx: PipelineContext) -> StageResult:
    log = ctx.logger("scoring")
    cfg = ctx.stage_cfg("scoring")
    result = StageResult()
    glob = load_global(ctx)
    tempo_info = glob["tempo"]
    bpm = float(tempo_info.get("tempo") or 120.0)
    beats = [b["time"] for b in glob["beat"]["beats"]]
    bpb = int(tempo_info.get("beats_per_bar", 4))
    tempo_map = TempoMap(bpm, beats)
    notes, drums, fx = load_fused(ctx)
    labels = load_labels(ctx)
    mcfg = cfg.get("midi", {}) or {}
    qcfg = cfg.get("quantize", {}) or {}
    tpb = int(mcfg.get("ticks_per_beat", 480))
    drum_map = load_resource_yaml(mcfg.get("drum_map", "drum_map.yaml"))
    programs = load_resource_yaml(mcfg.get("programs", "gm_programs.yaml"))
    drum_midi: dict[str, int] = {k: int(v) for k, v in drum_map.get("midi", {}).items()}
    other_perc = drum_midi.get("other_percussion", 75)
    key_events = glob["key"].get("key_events", [])
    key_sig = None
    if key_events:
        k0 = key_events[0]
        key_sig = f"{k0['tonic']}{'m' if k0['mode'] == 'minor' else ''}"

    by_track: dict[str, list[NoteEvent]] = defaultdict(list)
    for n in notes:
        by_track[n.track].append(n)
    names = unique_names(list(by_track), labels)
    midi_dir = ctx.project.midi_dir
    score_dir = ctx.project.score_dir
    settings = QuantizeSettings(grid=str(qcfg.get("grid", "1/16")), triplets=bool(qcfg.get("triplets", True)), swing=float(qcfg.get("swing", 0.0)), beats_per_bar=bpb)

    # ---- FX trigger map ---------------------------------------------------------------
    base_note = int(mcfg.get("fx_trigger_base_note", 24))
    trigger_map = {}
    fx_notes: list[MidiNote] = []
    for i, ev in enumerate(sorted(fx, key=lambda e: e.start)):
        note = base_note + i
        ev.trigger_note = note
        trigger_map[str(note)] = {"class": ev.fx_class, "clip": ev.clip_path, "start": ev.start, "end": ev.end, "event_id": ev.event_id}
        fx_notes.append(MidiNote(tempo_map.beats_at(ev.start), tempo_map.beats_at(ev.end), note, 100))
    write_json(ctx.project.fx_dir / "trigger_map.json", {"base_note": base_note, "triggers": trigger_map})
    result.artifacts["fx_trigger_map"] = ctx.rel(ctx.project.fx_dir / "trigger_map.json")

    # ---- MIDI (performance timing) ----------------------------------------------------
    def drum_spec(events: list[DrumEvent]) -> MidiTrackSpec:
        return MidiTrackSpec(name="Drums", is_drum=True, program=None, notes=[MidiNote(tempo_map.beats_at(e.time), tempo_map.beats_at(e.time + (e.duration or 0.1)), drum_midi.get(e.drum_class, other_perc), e.velocity) for e in events])

    specs: list[MidiTrackSpec] = []
    channel = 0
    for stem, ns in by_track.items():
        prog = program_for(labels.get(stem), programs)
        spec = MidiTrackSpec(name=names[stem], channel=channel, program=prog, notes=[MidiNote(tempo_map.beats_at(n.onset), tempo_map.beats_at(n.offset), n.midi_note, n.velocity) for n in ns])
        specs.append(spec)
        channel = channel + 1 if channel + 1 != 9 else 10
        channel %= 16
        write_midi(midi_dir / f"{names[stem]}.mid", build_midi([spec], tempo_map, tpb, (bpb, 4), key_sig, names[stem]))
        result.artifacts[f"midi:{names[stem]}"] = ctx.rel(midi_dir / f"{names[stem]}.mid")
    if drums:
        ds = drum_spec(drums)
        specs.append(ds)
        write_midi(midi_dir / "Drums.mid", build_midi([ds], tempo_map, tpb, (bpb, 4), key_sig, "Drums"))
        result.artifacts["midi:Drums"] = ctx.rel(midi_dir / "Drums.mid")
    if fx_notes:
        specs.append(MidiTrackSpec(name="FX_Triggers", channel=15, program=None, notes=fx_notes))
    full = midi_dir / "FULL_SONG.mid"
    write_midi(full, build_midi(specs, tempo_map, tpb, (bpb, 4), key_sig, ctx.project.song_name))
    result.artifacts["midi:FULL_SONG"] = ctx.rel(full)

    # ---- quantised score events ---------------------------------------------------------
    score_notes = quantize_notes(notes, tempo_map, settings)
    score_drums = quantize_drums(drums, tempo_map, settings)
    write_events(score_dir / "score_events.jsonl", score_notes)
    write_json(score_dir / "score_drums.json", {"events": score_drums})
    result.artifacts["score_events"] = ctx.rel(score_dir / "score_events.jsonl")
    if mcfg.get("write_quantized", True):
        qdir = midi_dir / "quantized"
        qspecs = []
        for stem in by_track:
            sn = [s for s in score_notes if s.track == stem]
            qspecs.append(MidiTrackSpec(name=names[stem], channel=0, program=program_for(labels.get(stem), programs), notes=[MidiNote(s.beat_onset, s.beat_onset + s.beat_duration, s.midi_note, s.velocity) for s in sn]))
        if score_drums:
            qspecs.append(MidiTrackSpec(name="Drums", is_drum=True, program=None, notes=[MidiNote(d["beat_onset"], d["beat_onset"] + d["beat_duration"], drum_midi.get(d["class"], other_perc), d["velocity"]) for d in score_drums]))
        write_midi(qdir / "FULL_SONG_quantized.mid", build_midi(qspecs, TempoMap(bpm), tpb, (bpb, 4), key_sig, f"{ctx.project.song_name} (quantized)"))
        result.artifacts["midi:FULL_SONG_quantized"] = ctx.rel(qdir / "FULL_SONG_quantized.mid")

    # ---- MusicXML -------------------------------------------------------------------
    if (cfg.get("musicxml", {}) or {}).get("enabled", True):
        key_changes = []
        for k in key_events:
            measure = int(tempo_map.beats_at(float(k["start"])) // bpb) + 1
            key_changes.append((max(1, measure), key_fifths(pitch_class_from_name(k["tonic"]), k["mode"]), k["mode"]))
        harmonies = []
        chords_path = ctx.project.analysis_dir / "chords.json"
        if chords_path.exists():
            for c in read_json(chords_path).get("chords", []):
                beat = tempo_map.beats_at(float(c["start"]))
                harmonies.append(XmlHarmony(start=int(round(beat * 4) / 4 * DIVISIONS), root=c["root"], quality=c["quality"], bass=c.get("bass")))
        dynamics = []
        for s in glob["structure"].get("sections", []):
            if s.get("dynamic"):
                beat = tempo_map.beats_at(float(s["start"]))
                dynamics.append((int(round(beat) * DIVISIONS), s["dynamic"]))
        parts: list[XmlPart] = []
        pid = 1
        for stem in by_track:
            sn = [s for s in score_notes if s.track == stem]
            part = XmlPart(part_id=f"P{pid}", name=names[stem], program=program_for(labels.get(stem), programs), notes=[XmlNote(start=int(round(s.beat_onset * DIVISIONS)), duration=max(2, int(round(s.beat_duration * DIVISIONS))), midi=s.midi_note, velocity=s.velocity) for s in sn])
            parts.append(part)
            pid += 1
            spec = ScoreSpec(title=f"{ctx.project.song_name} - {names[stem]}", beats=bpb, beat_type=4, bpm=bpm, key_changes=key_changes, parts=[part], harmonies=harmonies, dynamics=dynamics)
            write_musicxml(score_dir / f"{names[stem]}.musicxml", spec)
            result.artifacts[f"musicxml:{names[stem]}"] = ctx.rel(score_dir / f"{names[stem]}.musicxml")
        if score_drums:
            dpart = XmlPart(part_id=f"P{pid}", name="Drums", is_drum=True, drum_midi=drum_midi, drum_notation=drum_map.get("notation", {}), notes=[XmlNote(start=int(round(d["beat_onset"] * DIVISIONS)), duration=max(2, int(round(d["beat_duration"] * DIVISIONS))), midi=None, velocity=d["velocity"], drum_class=d["class"] if d["class"] in drum_midi else "other_percussion") for d in score_drums])
            parts.append(dpart)
            write_musicxml(score_dir / "Drums.musicxml", ScoreSpec(title=f"{ctx.project.song_name} - Drums", beats=bpb, bpm=bpm, key_changes=key_changes, parts=[dpart], dynamics=dynamics))
            result.artifacts["musicxml:Drums"] = ctx.rel(score_dir / "Drums.musicxml")
        write_musicxml(score_dir / "Full_Score.musicxml", ScoreSpec(title=ctx.project.song_name, beats=bpb, bpm=bpm, key_changes=key_changes, parts=parts, harmonies=harmonies, dynamics=dynamics))
        result.artifacts["musicxml:Full_Score"] = ctx.rel(score_dir / "Full_Score.musicxml")
    (score_dir / "README.txt").write_text(SCORE_README, encoding="utf-8")
    result.summary = {"tracks": list(names.values()), "score_notes": len(score_notes), "score_drums": len(score_drums), "fx_triggers": len(trigger_map), "grid": settings.grid, "triplets": settings.triplets, "swing": settings.swing}
    log.info("wrote %d MIDI tracks, %d score notes", len(specs), len(score_notes))
    return result
