# AI Music Reverse Engineering Pipeline
## MREP — Music Reverse Engineering Pipeline

**文件性質：系統設計規格 / Development Specification**

**主要使用對象：**

- Codex
- Claude Code
- Cursor
- GitHub Copilot Agent
- 其他 autonomous coding agents
- Python 開發者

---

# 1. 專案目標

本專案目標是建立一套 Python 音樂逆向分析系統。

使用者只需要輸入：

```text
MP3
WAV
FLAC
```

系統應盡可能把最終混音的音樂逆向拆解為：

1. 各類樂器與音源
2. 鼓組各個 component
3. 合成器類型與音樂角色
4. Bass 與 Sub Bass
5. 音符
6. 節奏
7. BPM
8. 拍號
9. 調性
10. 和弦
11. 音樂段落
12. Dynamic / Velocity
13. FX
14. MIDI
15. MusicXML
16. 可閱讀樂譜
17. 可重新演奏的多軌音樂

最終目標不是單純：

```text
Audio → Stems
```

而是：

```text
Audio
↓
Source Separation
↓
Instrument Recognition
↓
Musical Role Recognition
↓
Automatic Music Transcription
↓
Music Analysis
↓
MIDI / Score
↓
Reconstruction
```

---

# 2. 核心原則

本系統必須區分：

## 2.1 Source

「這是什麼聲音？」

例如：

```text
Piano
Guitar
Synth
Bass
Strings
Vocal
Kick
Snare
Hi-Hat
```

---

## 2.2 Musical Role

「這個聲音在歌曲裡扮演什麼功能？」

例如：

```text
Lead
Bassline
Pad
Arpeggio
Chord
Pluck
Drone
Rhythm
Percussion
FX
```

---

## 2.3 Timbre / Sound Design

「這個聲音具有什麼音色特性？」

例如：

```text
Warm
Bright
Analog-like
Digital
Pluck
Bell-like
Noise-based
Granular
Short attack
Long release
Distorted
```

---

## 2.4 Frequency Role

例如：

```text
Sub Bass
Low Bass
Low Mid
Mid
High
Air
```

---

因此不得使用：

```text
instrument = synth
```

作為唯一分類。

同一個聲音必須允許：

```json
{
  "source_family": "synthesizer",
  "musical_role": [
    "bassline",
    "pluck"
  ],
  "frequency_role": [
    "sub_bass"
  ],
  "sound_design": [
    "short_decay",
    "analog_style"
  ]
}
```

也就是：

**Multi-label classification**

而不是 mutually exclusive classification。

---

# 3. 預期輸入

第一階段必須支援：

```text
.wav
.flac
.mp3
```

未來可增加：

```text
.m4a
.aac
.aiff
.ogg
```

---

# 4. 預期輸出

建議每首歌曲建立：

```text
output/
└── SONG_NAME/
    │
    ├── manifest.json
    │
    ├── 01_normalized/
    │
    ├── 02_stems/
    │
    ├── 03_events/
    │
    ├── 04_midi/
    │
    ├── 05_score/
    │
    ├── 06_analysis/
    │
    ├── 07_fx/
    │
    └── 08_reconstruction/
```

---

# 5. Stem 分離目標

## 5.1 Drums

不得只有：

```text
drums.wav
```

盡可能進一步拆成：

```text
Kick
Snare
Closed Hi-Hat
Open Hi-Hat
Clap
Rimshot

High Tom
Mid Tom
Low Tom

Crash
Ride

Shaker
Tambourine
Cowbell
Conga

Other Percussion
```

---

# 6. Bass 分類

Bass 必須將「音源」與「功能」分開。

例如：

```text
Electric Bass
Acoustic Bass
Synth Bass
808
```

同時還可以具有：

```text
Bassline
Sub Bass
```

例如：

```json
{
  "source_family": "synth_bass",
  "source_subtype": "808",
  "musical_role": "bassline",
  "frequency_role": "sub_bass"
}
```

---

# 7. Synth 分類

Synth 不得只有：

```text
Synth
```

至少應進一步分析：

```text
Synth Lead
Synth Pad
Synth Bass
Synth Arp
Synth Pluck
Synth Keys
Synth Bell
Synth Brass
Synth Strings
Synth Chord
Synth Drone
Synth FX
```

---

# 8. Electronic Music 音色分類

系統需要特別支援以下類型：

## Drum

1. Kick
2. Snare
3. Closed Hi-Hat
4. Open Hi-Hat
5. Clap
6. Rimshot

## Bass

7. Bassline
8. Sub Bass
9. 808

## Synth

10. Synth Lead
11. Pad
12. Arpeggio
13. Pluck
14. Chord
15. Drone

## FX

16. Riser
17. Downlifter
18. White Noise
19. Impact
20. Glitch
21. Granular Texture

## Other

22. Vocal
23. Vocal Chop
24. Percussion
25. Atmosphere

---

# 9. Acoustic / Traditional Instrument 分類

另外至少應支援：

```text
Piano
Electric Piano
Organ

Acoustic Guitar
Electric Guitar

Violin
Viola
Cello
Strings

Trumpet
Trombone
French Horn
Brass

Flute
Clarinet
Saxophone
Woodwinds

Harp

Lead Vocal
Backing Vocal
```

---

# 10. 系統 Pipeline

完整 pipeline：

```text
INPUT AUDIO
│
▼
Audio Validation
│
▼
Audio Normalization
│
├───────────────┐
│               │
▼               ▼
Global MIR      Source Discovery
│               │
│               ▼
│        Broad Stem Separation
│               │
│               ▼
│       Specialist Separation
│               │
│               ▼
│       Instrument Classification
│               │
│               ▼
│        Role Classification
│               │
│               ▼
│            AMT
│               │
└───────┬───────┘
        ▼
 Event Fusion
        │
        ▼
 Musical Analysis
        │
        ├─ Notes
        ├─ Drums
        ├─ Chords
        ├─ Key
        ├─ Tempo
        ├─ Structure
        └─ Dynamics
        │
        ▼
 Score Conversion
        │
        ├─ MIDI
        ├─ MusicXML
        └─ PDF Score
        │
        ▼
 Reconstruction
        │
        ▼
 Reconstructed WAV
```

---

# 11. Stage 1 — Audio Ingest

工作內容：

```text
檢查格式
取得 duration
取得 sample rate
取得 channels
計算 SHA256
建立 Project ID
建立 Manifest
```

不得修改原始音訊。

---

# 12. Stage 2 — Normalization

建立 canonical internal audio。

建議：

```text
44.1 kHz 或 48 kHz
float32 processing
stereo
```

另外建立：

```text
analysis_mono.wav
```

供部分分析模型使用。

---

# 13. Stage 3 — Global Music Analysis

分析：

```text
Tempo
Beat
Downbeat
Time Signature
Key
Key Change
Section
Dynamics
Loudness
```

例如：

```json
{
  "tempo": 99.38,
  "time_signature": "4/4",
  "key_events": [
    {
      "start": 0,
      "key": "D major"
    },
    {
      "start": 129.7,
      "key": "E major"
    }
  ]
}
```

---

# 14. Stage 4 — Broad Source Discovery

先回答：

> 這首歌裡可能有哪些聲源？

例如：

```text
Drums       0.99
Bass        0.97
Piano       0.88
Synth       0.94
Strings     0.74
Guitar      0.62
Vocals      0.03
```

這只是 Discovery。

不能直接視為最終分類。

---

# 15. Stage 5 — Broad Separation

第一輪可使用較通用模型：

```text
Drums
Bass
Vocals
Piano
Guitar
Other
```

或者：

```text
10 stems
20 stems
50+ stems
```

視硬體與 profile 決定。

---

# 16. Stage 6 — Specialist Separation

重要原則：

**不要所有歌曲都強制跑所有 specialist model。**

例如 Broad Detection 發現：

```text
Piano 0.93
Synth 0.89
Strings 0.72
```

才分別執行：

```text
Piano specialist
Synth specialist
Strings specialist
```

如果：

```text
Trumpet 0.02
```

則沒有必要執行 Trumpet specialist。

---

# 17. Stage 7 — Drum Event Detection

Drum 不適合完全當成一般 pitch transcription。

每一擊表示成：

```json
{
  "time": 12.482,
  "class": "snare",
  "velocity": 96,
  "confidence": 0.94
}
```

最低要求：

```text
Kick
Snare
Closed Hi-Hat
Open Hi-Hat
```

之後加入：

```text
Clap
Rimshot
Tom
Crash
Ride
Shaker
Tambourine
```

---

# 18. MIDI Drum Mapping

使用 General MIDI Drum Mapping。

例如：

```text
Kick → MIDI 36
Snare → MIDI 38
Closed HH → MIDI 42
Open HH → MIDI 46
Clap → MIDI 39
```

Mapping 不應 hard-code。

必須 config-driven。

---

# 19. Stage 8 — Automatic Music Transcription

對適合轉成音符的 stem：

```text
Piano
Bass
Guitar
Synth Lead
Synth Pad
Synth Arp
Strings
Brass
Woodwind
Vocal Melody
```

分析：

```text
Pitch
Onset
Offset
Duration
Velocity
Pitch Bend
Confidence
```

例如：

```json
{
  "pitch": "F#4",
  "midi_note": 66,
  "onset": 12.500,
  "offset": 13.125,
  "velocity": 81,
  "confidence": 0.92
}
```

---

# 20. Two-path AMT

不要只相信單一 AMT。

使用：

## Path A

```text
Stem
↓
Per-stem AMT
↓
Notes
```

## Path B

```text
Original Mix
↓
Multi-instrument AMT
↓
Multi-track Notes
```

最後：

```text
A + B
↓
Event Fusion
```

---

# 21. Event Fusion

融合依據：

```text
Model confidence
Stem energy
Pitch agreement
Onset agreement
Duration agreement
Instrument compatibility
Musical-role compatibility
```

例如：

```text
Piano Stem AMT:
F#4 @ 23.51
Confidence 0.88

Full Mix AMT:
F#4 @ 23.50
Confidence 0.79

Result:
F#4 @ 23.505
Confidence ↑
```

---

# 22. Raw Event 與 Score Event 必須分開

這點非常重要。

原始模型可能得到：

```text
0.493 sec
0.998 sec
1.496 sec
```

樂譜可能量化成：

```text
1/8
1/4
1/8
```

不能覆蓋原始資料。

因此：

```text
RAW EVENTS
```

和：

```text
SCORE EVENTS
```

必須分別保存。

---

# 23. Chord Analysis

Chord 不只是從單一 stem 判斷。

應綜合：

```text
Chroma
Bass
Piano
Guitar
Synth
Strings
AMT notes
```

輸出：

```json
{
  "start": 12.0,
  "end": 14.4,
  "root": "D",
  "quality": "major",
  "bass": "D",
  "confidence": 0.91
}
```

應支援：

```text
Major
Minor
7
maj7
m7
sus2
sus4
dim
aug
add9
inversion
slash chord
```

---

# 24. Arpeggio Detection

若事件：

```text
D4
A4
F#4
A4
D4
A4
F#4
A4
```

可分析：

```text
Chord = D Major

Pattern:
1 - 5 - 3 - 5

Rhythm:
1/8 notes

Role:
Arpeggio
```

---

# 25. Synth Sound Analysis

Synth 可以分析：

```text
Attack
Decay
Sustain
Release

Spectral Centroid
Spectral Rolloff
Spectral Flatness

Harmonicity
Noise Ratio

Stereo Width

Pitch Modulation
Amplitude Modulation

Filter Movement
```

例如：

```json
{
  "source": "synthesizer",
  "role": "pad",
  "character": [
    "warm",
    "analog_style"
  ],
  "attack_ms": 420,
  "release_ms": 1800,
  "stereo_width": 0.72
}
```

---

# 26. Synth Preset Identification 限制

禁止直接輸出：

```text
Serum Preset X
Juno-106 Patch 14
Omnisphere Patch ABC
```

除非具有額外 evidence。

只能合理描述：

```text
Analog-style Pad
Supersaw-like Lead
FM Bell
Digital Pluck
Subtractive Synth Bass
```

未來可以增加：

```text
Synth Patch Approximation
```

但必須輸出：

```text
candidate
probability
similarity
```

而不是宣稱唯一答案。

---

# 27. FX 的特殊處理

以下不應硬轉成傳統樂譜：

```text
Riser
Downlifter
White Noise
Impact
Glitch
Granular
Atmosphere
```

應保存：

```text
start
end
type
confidence
audio clip
features
```

例如：

```json
{
  "class": "riser",
  "start": 28.1,
  "end": 31.95,
  "confidence": 0.86
}
```

另外可建立：

```text
FX Trigger MIDI
```

例如：

```text
C1 = Impact 001
C#1 = Riser 001
D1 = Downlifter 001
```

---

# 28. Output Stems

例如：

```text
02_stems/
├── drums/
│   ├── kick.wav
│   ├── snare.wav
│   ├── clap.wav
│   ├── closed_hat.wav
│   ├── open_hat.wav
│   └── percussion.wav
│
├── bass/
│   ├── electric_bass.wav
│   ├── synth_bass.wav
│   └── sub_bass.wav
│
├── synth/
│   ├── lead.wav
│   ├── pad.wav
│   ├── arp.wav
│   ├── pluck.wav
│   └── chord.wav
│
├── piano.wav
├── guitar.wav
├── strings.wav
├── brass.wav
└── vocal.wav
```

注意：

不保證所有歌曲都會產生所有 stem。

只建立偵測存在或使用者指定的項目。

---

# 29. MIDI Output

輸出：

```text
04_midi/
├── Piano.mid
├── Bass.mid
├── Guitar.mid
├── Strings.mid
├── Synth_Lead.mid
├── Synth_Pad.mid
├── Synth_Arp.mid
├── Drums.mid
└── FULL_SONG.mid
```

---

# 30. MusicXML

輸出：

```text
05_score/
├── Piano.musicxml
├── Bass.musicxml
├── Guitar.musicxml
├── Strings.musicxml
├── Synth.musicxml
├── Drums.musicxml
└── Full_Score.musicxml
```

可以由 MuseScore 等軟體轉換成：

```text
PDF
SVG
PNG
```

---

# 31. Score Quantization

必須支援：

```text
1/4
1/8
1/16
Triplets
Swing
```

但不可破壞 Raw Event。

例如：

```text
Raw:
0.503 sec

Score:
Beat 1.5
```

兩份都要留。

---

# 32. Dynamics

不要把所有 MIDI 音符 velocity 設成固定值。

分析：

```text
Amplitude
Transient intensity
Local loudness
Section dynamics
Relative instrument dynamics
```

轉換成：

```text
MIDI velocity
```

甚至可推論：

```text
pp
p
mp
mf
f
ff
```

---

# 33. Humanization Metadata

重新演奏時可保存：

```text
micro timing
velocity variation
note length
legato
staccato
sustain
pitch bend
vibrato
```

避免：

```text
所有音符完全 grid-aligned
所有 velocity = 100
```

造成 MIDI 機械感。

---

# 34. Reconstruction

最低標準：

```text
MIDI
+
SoundFont
+
FX Samples
↓
Rendered WAV
```

輸出：

```text
08_reconstruction/
├── full_mix.wav
├── full_mix.mp3
└── track_renders/
```

---

# 35. Reconstruction Metadata

必須紀錄來源。

例如：

```json
{
  "piano": "midi_rendered",
  "bass": "midi_rendered",
  "strings": "midi_rendered",
  "riser": "original_audio_clip",
  "granular_fx": "original_audio_clip"
}
```

---

# 36. Model Adapter Architecture

核心系統不能直接依賴某個模型。

介面：

```python
class SeparationAdapter:
    def separate(...):
        ...

class TranscriptionAdapter:
    def transcribe(...):
        ...

class ClassificationAdapter:
    def classify(...):
        ...

class MIRAdapter:
    def analyze(...):
        ...

class ReconstructionAdapter:
    def render(...):
        ...
```

---

# 37. 為什麼必須 Adapter

因為模型可能：

```text
更新
停止維護
Python incompatibility
CUDA incompatibility
VRAM 不足
License 不適合
品質變差
```

所以核心架構不得是：

```python
from specific_model import model
```

遍布整個 codebase。

應集中在：

```text
adapters/
```

---

# 38. Python 專案架構

建議：

```text
music-reverse-engineering/
│
├── README.md
├── ARCHITECTURE.md
├── AGENTS.md
├── CLAUDE.md
├── ROADMAP.md
├── TASKS.md
│
├── pyproject.toml
│
├── config/
│   ├── default.yaml
│   ├── light.yaml
│   ├── standard.yaml
│   └── max_quality.yaml
│
├── src/
│   └── mrep/
│       │
│       ├── cli.py
│       ├── config.py
│       ├── manifest.py
│       ├── logging.py
│       │
│       ├── domain/
│       │   ├── project.py
│       │   ├── labels.py
│       │   ├── events.py
│       │   └── tracks.py
│       │
│       ├── pipeline/
│       │   ├── ingest.py
│       │   ├── normalization.py
│       │   ├── global_analysis.py
│       │   ├── separation.py
│       │   ├── classification.py
│       │   ├── transcription.py
│       │   ├── fusion.py
│       │   ├── scoring.py
│       │   └── reconstruction.py
│       │
│       ├── adapters/
│       │   ├── separation/
│       │   ├── transcription/
│       │   ├── classification/
│       │   ├── mir/
│       │   ├── score/
│       │   └── render/
│       │
│       ├── io/
│       │   ├── audio.py
│       │   ├── midi.py
│       │   ├── musicxml.py
│       │   └── json.py
│       │
│       └── utils/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
└── output/
```

---

# 39. Hardware Profiles

系統不能假設所有電腦都有大型 GPU。

## Light

```text
低 VRAM
CPU 可執行
小型 separator
chunked processing
只執行必要 specialist model
```

---

## Standard

```text
一般 CUDA GPU
Broad separation
Specialist refinement
Per-stem AMT
必要時 Full-mix AMT
```

---

## Max Quality

```text
High VRAM
Large multi-stem discovery
Multiple specialist models
Multiple AMT
Ensemble
Confidence fusion
```

---

# 40. CLI

目標：

```bash
mrep run song.flac
```

完整：

```bash
mrep run song.flac \
    --profile standard \
    --device auto
```

其他：

```bash
mrep analyze song.flac
mrep separate song.flac
mrep transcribe output/song
mrep score output/song
mrep reconstruct output/song
```

---

# 41. Resume

AI inference 很耗時間。

因此：

```bash
mrep run song.wav --resume
```

必須能跳過：

```text
已完成 separation
已完成 transcription
```

只執行尚未完成的 stage。

---

# 42. Force Stage

例如：

```bash
mrep run song.wav \
    --force-stage transcription
```

只重新計算 transcription 與其 downstream stages。

---

# 43. Manifest

每個 Project 必須具有：

```text
manifest.json
```

內容包含：

```text
Project ID
Source SHA256
Audio metadata
Configuration
Model
Model version
Stage status
Artifacts
Confidence
Errors
Timestamp
```

---

# 44. Provenance

所有 AI 推論必須知道：

> 這個答案是誰推論的？

例如：

```json
{
  "confidence": 0.92,
  "provenance": {
    "stage": "transcription",
    "adapter": "amt_x",
    "model": "checkpoint_x",
    "model_version": "abc123",
    "run_id": "run_001",
    "input": "stem_piano_01",
    "parameters_hash": "..."
  }
}
```

---

# 45. Confidence

所有 AI 推論應盡可能：

```text
0.0 – 1.0
```

例如：

```text
Piano       0.96
Strings     0.78
Synth Pad   0.84
808         0.54
Granular    0.31
```

如果模型沒有 confidence：

```json
{
  "confidence": null,
  "confidence_method": "unavailable"
}
```

不可偽造精確數值。

---

# 46. Low Confidence Handling

例如：

```text
808 confidence = 0.48
```

系統可以：

```text
重新分析
換 specialist model
比較其他模型
標記 ambiguous
```

而不是直接：

```text
這就是 808
```

---

# 47. Model Ensemble

Max Quality 模式：

```text
Model A
Model B
Model C
↓
Consensus
```

可用：

```text
Pitch agreement
Onset agreement
Class agreement
Stem energy
Confidence
```

計算 final confidence。

---

# 48. Error Handling

單一模型失敗：

```text
Synth specialist failed
```

不可讓整個歌曲分析失敗。

輸出：

```text
Project Status:
PARTIAL
```

並繼續：

```text
Piano
Bass
Drums
Strings
```

---

# 49. Logging

至少：

```text
project_id
stage
model
device
runtime
VRAM
cache hit
output
error
```

---

# 50. Testing

測試分成：

## Unit

不載模型。

測：

```text
Schema
IDs
Quantization
Fusion
MIDI mapping
Timeline
Manifest
```

---

## Contract

使用 Fake Adapter。

測：

```text
Pipeline
Resume
Cache
Partial failure
```

---

## Integration

真正載入模型。

標示：

```python
@pytest.mark.integration
@pytest.mark.slow
```

---

## End-to-End

使用：

```text
短音訊
自製音訊
Synthetic audio
Public-domain audio
```

不要把 copyrighted commercial song 放入 repository。

---

# 51. Synthetic Tests

可以 Python 自己產生：

```text
440 Hz sine
Sub Bass
Major chord
Minor chord
Arpeggio
Kick
Snare
Noise Hat
Riser
White Noise
```

用來確認整個 pipeline 是否正常。

---

# 52. Quality Metrics

Separation：

```text
SDR
SI-SDR
Leakage
```

Transcription：

```text
Precision
Recall
F1
Onset F1
Offset F1
Pitch Accuracy
```

Drums：

```text
Onset F1
Class Confusion Matrix
```

Beat：

```text
Tempo error
Beat F1
Downbeat F1
```

---

# 53. 重要開發規範

Codex / Claude Code 必須遵守：

1. 不要建立 monolithic script。
2. 不要把專案做成單一 Notebook。
3. Domain layer 不得 import PyTorch/TensorFlow/JAX。
4. ML model code 放在 adapters。
5. Import package 不得自動下載模型。
6. Model download 必須明確。
7. Raw events 不得被 quantized events 覆蓋。
8. FX 不得硬轉為一般音符。
9. Synth preset 不可假裝精確辨識。
10. 所有 AI 結果保存 provenance。
11. 所有 AI 結果保存 confidence。
12. Pipeline 必須 resume。
13. CPU fallback 必須存在。
14. 模型失敗必須 partial success。
15. 對大型模型不得成為基本功能的必要條件。

---

# 54. 建議開發順序

## Milestone 0

建立：

```text
Repository
Python package
CLI
Config
Manifest
Stage Runner
Cache
Resume
Mock Adapter
Tests
```

完全先不要急著接 AI model。

---

# 55. Milestone 1

建立最小可運作版本：

```text
Audio
↓
Normalize
↓
Basic separation
↓
Tempo
↓
Kick/Snare/HH
↓
One pitched AMT
↓
MIDI
↓
SoundFont
↓
WAV
```

這是第一個 Vertical Slice。

---

# 56. Milestone 2

加入：

```text
Piano
Guitar
Synth
Strings
Bass
Vocal
```

Specialist separation。

---

# 57. Milestone 3

加入電子音樂詳細分類：

```text
808
Sub Bass
Lead
Pad
Arp
Pluck
Chord
Drone

Riser
Downlifter
Impact
Noise
Glitch
Granular
```

---

# 58. Milestone 4

加入：

```text
MusicXML
Score
Drum notation
Chord notation
Key change
Dynamics
```

---

# 59. Milestone 5

加入第二組 AMT：

```text
Per-stem AMT
+
Full Mix AMT
↓
Fusion
```

---

# 60. Milestone 6

改善 Reconstruction：

```text
Velocity
Dynamics
Humanization
Articulation
Instrument mapping
FX
Mixing
```

---

# 61. Future Features

未來可以加入：

```text
DAW project export
Ableton export
Reaper export
Logic-compatible MIDI export

Synth patch approximation
Timbre similarity search

Interactive correction UI

Waveform editor

Piano-roll editor

Score editor

Automatic orchestration

Original vs reconstructed A/B comparison
```

---

# 62. 專案最終輸出範例

輸入：

```text
Road_Trip.flac
```

輸出：

```text
Road_Trip/
│
├── manifest.json
│
├── 01_normalized/
│
├── 02_stems/
│   ├── kick.wav
│   ├── snare.wav
│   ├── closed_hat.wav
│   ├── bass.wav
│   ├── sub_bass.wav
│   ├── piano.wav
│   ├── guitar.wav
│   ├── synth_pad.wav
│   ├── synth_arp.wav
│   └── strings.wav
│
├── 03_events/
│   ├── notes.jsonl
│   ├── drums.jsonl
│   ├── fx.jsonl
│   └── labels.json
│
├── 04_midi/
│   ├── Piano.mid
│   ├── Bass.mid
│   ├── Guitar.mid
│   ├── Synth.mid
│   ├── Strings.mid
│   ├── Drums.mid
│   └── FULL_SONG.mid
│
├── 05_score/
│   ├── Piano.musicxml
│   ├── Strings.musicxml
│   └── Full_Score.musicxml
│
├── 06_analysis/
│   ├── tempo.json
│   ├── beat.json
│   ├── key.json
│   ├── chords.json
│   ├── structure.json
│   ├── dynamics.json
│   └── instruments.json
│
├── 07_fx/
│   ├── riser_001.wav
│   ├── impact_001.wav
│   └── trigger_map.json
│
└── 08_reconstruction/
    ├── reconstructed.wav
    └── reconstructed.mp3
```

---

# 63. 最終系統定義

這個專案不應被定義為：

> AI Stem Separator

也不是：

> Audio to MIDI

正確定位是：

# AI Music Reverse Engineering System

其功能涵蓋：

```text
Source Separation
+
Instrument Recognition
+
Sound Classification
+
Musical Role Analysis
+
Automatic Music Transcription
+
Music Information Retrieval
+
Score Reconstruction
+
Audio Reconstruction
```

最終目標是：

> 在接受音訊逆向工程本身存在不可辨識性的前提下，把一個完成混音的音樂檔，盡可能轉換成結構化、可分析、可編輯、可看譜、可播放、可重新編曲的多軌音樂工程資料。

---

# 64. Coding Agent 最重要的一句指令

Codex、Claude Code 或其他 coding agent 開始開發之前，都必須遵循：

> **先建立可靠、可測試、可替換模型、可續跑的架構，再逐步提高 AI 模型品質；不得為了快速展示結果，把所有功能塞進單一 Python 腳本。**

第一個真正的成功標準不是「一次就能完美拆出所有樂器」，而是：

> **每加入一個更好的模型，都能直接替換 Adapter，而不需要重新設計整個系統。**