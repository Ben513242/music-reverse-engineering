from pathlib import Path

from mrep.config import find_config_dir, load_resource_yaml
from mrep.io.midi import MidiNote, MidiTrackSpec, TempoMap, build_midi, read_midi_notes, write_midi


def test_drum_mapping_is_config_driven_general_midi():
    dm = load_resource_yaml("drum_map.yaml")["midi"]
    assert dm["kick"] == 36 and dm["snare"] == 38 and dm["closed_hat"] == 42 and dm["open_hat"] == 46 and dm["clap"] == 39
    assert (find_config_dir() / "drum_map.yaml").is_file()
    notation = load_resource_yaml("drum_map.yaml")["notation"]
    assert notation["closed_hat"]["notehead"] == "x"


def test_midi_roundtrip_with_tempo_map(tmp_path: Path):
    tm = TempoMap(100.0)
    spec = MidiTrackSpec(name="Piano", notes=[MidiNote(0.0, 1.0, 60, 90), MidiNote(1.0, 1.5, 64, 70)], program=0)
    drums = MidiTrackSpec(name="Drums", is_drum=True, program=None, notes=[MidiNote(0.0, 0.25, 36, 100)])
    mf = build_midi([spec, drums], tm, 480, (4, 4), "C", "test")
    path = tmp_path / "t.mid"
    write_midi(path, mf)
    notes = read_midi_notes(path)
    assert len(notes) == 3
    piano = sorted((x for x in notes if x["channel"] == 0), key=lambda x: x["onset"])
    assert abs(piano[0]["onset"] - 0.0) < 1e-6 and abs(piano[0]["offset"] - 0.6) < 1e-3  # 1 beat at 100 bpm
    assert abs(piano[1]["onset"] - 0.6) < 1e-3
    assert any(x["channel"] == 9 and x["note"] == 36 for x in notes)
    assert piano[0]["velocity"] == 90 and piano[1]["velocity"] == 70


def test_midi_velocities_vary_not_fixed(tmp_path: Path):
    tm = TempoMap(120.0, [0.0, 0.5, 1.0, 1.5, 2.0])
    spec = MidiTrackSpec(name="X", notes=[MidiNote(i * 0.5, i * 0.5 + 0.25, 60 + i, 40 + i * 20) for i in range(4)])
    path = tmp_path / "v.mid"
    write_midi(path, build_midi([spec], tm))
    vel = {x["velocity"] for x in read_midi_notes(path)}
    assert len(vel) == 4
