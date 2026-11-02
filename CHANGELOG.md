# Changelog

## Unreleased

### Added
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
