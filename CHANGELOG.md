# Changelog

## Unreleased

### Added (round 3)
- Viterbi key segmentation (switch penalty, relative keys cost more) replaces the rule-based key-change logic.
- 6/8 metre candidate with sub-period penalties; `metre_name()`.
- Tempo curve (`tempo_events`, `tempo_stability`), per-measure MusicXML tempo marks, Reaper tempo envelope.
- Humanised rendering: articulation-dependent envelopes, vibrato and pitch-bend curves in the additive renderer.
- `mrep evaluate` (tempo / beats / downbeats / key / time signature / notes / drums / FX / chords / separation
  SDR, SI-SDR, leakage) and `mrep synth --stems` reference stems + beat ground truth.
- DAW export: `08_reconstruction/daw/session.json` and a Reaper `.rpp` with stem tracks, markers and tempo.
- Tests for `.ogg` / `.m4a` / `.flac` ingestion through ffmpeg and for 48 kHz input.

### Added (round 2)
- Time-signature estimation (3/4 vs 4/4) from beat accents, with confidence and per-candidate scores.
- `sound_analysis` adapter kind (`dsp`): ADSR, spectral shape, harmonicity / noise, inharmonicity,
  vibrato / tremolo / filter movement, stereo width -> sound-design descriptors and *approximate*
  patch candidates (probability + similarity) in `06_analysis/sound_design.json` and labels.
- Pitch-bend / vibrato estimation on monophonic stems (YIN-style f0 tracking) -> `NoteEvent.pitch_bend`,
  `features.vibrato`, MIDI pitch-wheel messages.
- Humanisation metadata: per-note articulation (legato / staccato), per-track timing / velocity statistics
  (`05_score/humanization.json`), MusicXML staccato marks and slurs.
- Low-confidence handling: `06_analysis/ambiguities.json` with suggestions; `SoundLabel.ambiguous`.
- Drum transcription ensembles (`drums_adapters: [...]`) fused by consensus.
- Optional ML adapter scaffolds: `separation/demucs`, `transcription/basic_pitch` (lazy imports,
  explicit model download, integration tests skip without the dependencies).

## 0.1.0

- Initial MREP architecture (M0) and vertical slice (M1) with model-free baseline adapters.
