import xml.etree.ElementTree as ET

from mrep.io.musicxml import (
    DIVISIONS,
    ScoreSpec,
    XmlHarmony,
    XmlNote,
    XmlPart,
    decompose_duration,
    read_musicxml_summary,
    write_musicxml,
)


def test_decompose_durations_use_standard_note_values():
    assert decompose_duration(24) == [24]
    assert decompose_duration(30) == [24, 6]
    assert decompose_duration(8) == [8]  # triplet eighth
    assert decompose_duration(1) == [2]  # not notatable -> rounded up
    assert sum(decompose_duration(97)) >= 97


def test_musicxml_parts_ties_chords_and_harmony(tmp_path):
    q = DIVISIONS
    piano = XmlPart(part_id="P1", name="Piano", notes=[XmlNote(0, q, 60), XmlNote(0, q, 64), XmlNote(0, q, 67), XmlNote(3 * q, 2 * q, 62)])  # chord + note tied across bar 1|2
    drums = XmlPart(part_id="P2", name="Drums", is_drum=True, drum_midi={"kick": 36, "snare": 38}, drum_notation={"kick": {"step": "F", "octave": 4, "notehead": "normal"}, "snare": {"step": "C", "octave": 5, "notehead": "normal"}}, notes=[XmlNote(0, q // 4, None, drum_class="kick"), XmlNote(q, q // 4, None, drum_class="snare")])
    spec = ScoreSpec(title="t", beats=4, beat_type=4, bpm=120, key_changes=[(1, 2, "major"), (2, -1, "major")], parts=[piano, drums], harmonies=[XmlHarmony(0, "D", "major", "F#"), XmlHarmony(4 * q, "G", "min7")], dynamics=[(0, "mf")])
    path = write_musicxml(tmp_path / "s.musicxml", spec)
    root = ET.parse(path).getroot()
    assert root.tag == "score-partwise"
    summary = read_musicxml_summary(path)
    assert summary["parts"] == ["P1", "P2"] and summary["measures"] == 2
    assert summary["harmonies"] == 2 and summary["ties"] == 2
    p1 = root.find("part[@id='P1']")
    first_measure_notes = p1.find("measure[@number='1']").findall("note")
    assert first_measure_notes[1].find("chord") is not None
    assert p1.find("measure[@number='2']/attributes/key/fifths").text == "-1"
    assert root.find("part[@id='P2']/measure/note/unpitched/display-step").text == "F"
    assert root.find("part-list/score-part[@id='P2']/midi-instrument/midi-unpitched").text == "37"
    bass = root.find("part[@id='P1']/measure/harmony/bass/bass-step")
    assert bass is not None and bass.text == "F"
