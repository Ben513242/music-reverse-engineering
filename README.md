# MREP — Music Reverse Engineering Pipeline

**AI 音樂逆向工程系統**：輸入一首已完成混音的音樂（`.wav` / `.flac` / `.mp3`），
盡可能逆向拆解成結構化、可分析、可編輯、可看譜、可播放、可重新編曲的多軌音樂工程資料。

```text
Audio → Source Separation → Instrument Recognition → Musical Role Recognition
      → Automatic Music Transcription → Music Analysis → MIDI / Score → Reconstruction
```

本專案不是「AI Stem Separator」，也不是「Audio to MIDI」，而是把以下能力串成一條
**可靠、可測試、可替換模型、可續跑**的 pipeline：

| 能力 | 產出 |
| --- | --- |
| Source separation（broad + 依偵測結果觸發的 specialist） | `02_stems/` |
| 多標籤樂器 / 音樂角色 / 頻率角色 / 音色分類 | `03_events/labels.json`, `06_analysis/instruments.json` |
| 鼓組 component 事件（kick / snare / hat …，含 velocity 與 confidence） | `03_events/drums.jsonl` |
| 音高轉譜（per-stem AMT + full-mix AMT + event fusion） | `03_events/notes.jsonl` |
| Tempo（含變速曲線）/ beat / downbeat / 拍號（3/4、4/4；6/8 於 max_quality 啟用）/ 調性分段（Viterbi）/ 段落 / 動態 / 響度 | `06_analysis/*.json` |
| Synth / 音色分析（ADSR、spectral、harmonicity、vibrato / tremolo / filter movement）與 patch 近似候選 | `06_analysis/sound_design.json` |
| Humanization（micro-timing、velocity、legato / staccato、pitch bend、vibrato） | `05_score/humanization.json`, MIDI pitch wheel, MusicXML articulations |
| 低信心標記與建議（labels / chords / key / fx / notes / drums） | `06_analysis/ambiguities.json` |
| DAW 匯出（DAW 無關的 session.json + Reaper .rpp：stem 音軌、markers、tempo envelope） | `08_reconstruction/daw/` |
| 品質評估（tempo / beats / key / notes / drums / FX / chords / separation SDR） | `mrep evaluate` → `06_analysis/evaluation.json` |
| 和弦（含 7th / sus / dim / aug / add9 / slash chord）與琶音型態 | `06_analysis/chords.json`, `arpeggios.json` |
| FX（riser / downlifter / impact / noise …）保存為音訊片段 + trigger MIDI | `07_fx/` |
| MIDI（保留 micro-timing 的 performance 版 + quantized 版）、MusicXML 樂譜 | `04_midi/`, `05_score/` |
| 重新演奏（MIDI → 合成 / SoundFont，FX 以原始片段回填） | `08_reconstruction/` |

> 設計規格請見 [`docs/SPEC.md`](docs/SPEC.md)；架構說明請見 [`ARCHITECTURE.md`](ARCHITECTURE.md)；
> 要在有 GPU 的機器接手模型整合，請讀 [`docs/HANDOVER.md`](docs/HANDOVER.md)。

## 安裝

```bash
git clone https://github.com/Ben513242/music-reverse-engineering.git
cd music-reverse-engineering
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

需求：Python 3.10+。核心只依賴 `numpy / scipy / soundfile / PyYAML / mido`，
**不需要 GPU、不需要下載任何模型**就能跑完整條 pipeline（內建 baseline DSP adapters）。
`ffmpeg`（選用）用於 mp3 輸出與非 libsndfile 格式解碼。

## 快速開始

```bash
# 產生一首合成的測試歌曲（鼓 / sub bass / 和弦 / 琶音 / riser）+ ground truth
mrep synth demo.wav --bars 8

# 跑完整 pipeline
mrep run demo.wav --profile standard --device auto

# 中斷後續跑（已完成的 stage 會 cache hit 跳過）
mrep run demo.wav --resume

# 只重跑 transcription 與其 downstream
mrep run demo.wav --resume --force-stage transcription

# 分段執行
mrep analyze song.flac          # ingest + normalize + tempo / key / structure
mrep separate song.flac         # + separation + classification
mrep transcribe output/song     # transcription + fusion
mrep score output/song          # MIDI + MusicXML
mrep reconstruct output/song    # 重新演奏成 WAV / MP3
mrep info output/song           # 檢視 manifest / 錯誤 / 警告
```

評估：

```bash
mrep synth demo.wav --bars 8 --stems ref_stems/        # 合成歌曲 + ground truth + 參考 stems
mrep run demo.wav --profile standard
mrep evaluate output/demo --truth demo.wav.truth.json --stems ref_stems/
```

其他：`mrep profiles`、`mrep adapters [--kind separation]`、`mrep models download <kind> <name>`。
任何設定都可用 `--set` 覆寫，例如 `--set stages.transcription.polyphony=6`。

## 輸出結構

```text
output/SONG_NAME/
├── manifest.json            # project id, SHA256, config, 每個 stage 的狀態 / 模型 / artifacts / confidence / errors
├── 01_normalized/           # audio.wav (44.1k float32 stereo) + analysis_mono.wav
├── 02_stems/                # drums.wav bass.wav other.wav + drums/{kick,snare,hats,percussion}.wav + stems.json
├── 03_events/               # notes.jsonl drums.jsonl fx.jsonl labels.json + raw/ (每個 adapter 的原始輸出)
├── 04_midi/                 # Bass.mid Other.mid Drums.mid FULL_SONG.mid + quantized/
├── 05_score/                # *.musicxml Full_Score.musicxml score_events.jsonl（量化後，另存，不覆蓋 raw）
├── 06_analysis/             # tempo beat key chords structure dynamics loudness instruments arpeggios
├── 07_fx/                   # riser_001.wav … + trigger_map.json (C1 = 24 起)
├── 08_reconstruction/       # reconstructed.wav / .mp3 + track_renders/ + reconstruction.json（每軌來源）+ daw/（session.json, .rpp）
└── logs/mrep.log.jsonl
```

## Hardware profiles

| profile | 用途 |
| --- | --- |
| `light` | CPU / 低 VRAM：chunked separation、較少 specialist、較低 polyphony |
| `standard` | 一般 GPU：broad + specialist separation、per-stem AMT + full-mix AMT |
| `max_quality` | 高 VRAM：更多 specialist、多個 AMT adapter 的 ensemble consensus |

設定檔在 `config/`（`default.yaml` 為基底，其餘 profile deep-merge 其上），鼓的 MIDI 對應在
`config/drum_map.yaml`、GM program 在 `config/gm_programs.yaml` —— 全部 config-driven，不寫死在程式裡。

## 核心原則（摘要）

1. **Source ≠ Musical Role ≠ Frequency Role ≠ Sound Design**：同一個聲音是 multi-label，
   例如 `{"source_family": "synthesizer", "musical_role": ["bassline", "pluck"], "frequency_role": ["sub_bass"]}`。
2. **Raw events 與 Score events 分開保存**：量化永遠不覆蓋模型輸出的原始秒數。
3. **FX 不硬轉樂譜**：riser / impact 等以音訊片段 + features + trigger note 保存。
4. **所有 AI 結果都有 `confidence` 與 `provenance`**；模型沒有 confidence 時誠實寫 `null` + `confidence_method: unavailable`。
5. **Synth preset 不假裝精確辨識**：只描述（analog-style pad、FM bell…）或給 candidate / probability。
6. **Adapter 架構**：核心不依賴任何特定模型；換模型 = 換 adapter + 改 config，不重寫系統。
7. **Resume / partial success / CPU fallback**：單一模型失敗 → 專案狀態 `PARTIAL`，其餘照常輸出。

## 目前的 baseline adapters

內建的 adapter 全是 model-free 的 DSP（numpy/scipy），目的是讓架構端到端可跑、可測，
品質遠不及神經網路模型，**預期被更好的 adapter 取代**（見 `ROADMAP.md`）：

| kind | name | 做法 |
| --- | --- | --- |
| separation | `hpss` | median-filter HPSS → drums / bass(低通) / other |
| separation | `drum_bandsplit` | 鼓 stem 依頻帶拆 kick / snare / hats / percussion（specialist） |
| discovery | `energy_heuristic` | 能量比例估計 drums / bass / harmonic 是否存在（其餘家族誠實回 `null`） |
| classification | `heuristic` | 頻譜 / 包絡特徵 → 多標籤 + candidates |
| drum_transcription | `onset_bands` | onset + 頻帶能量規則（kick / snare / closed_hat / open_hat / clap / tom） |
| transcription | `spectral_peaks` | iterative harmonic-summation 多音高偵測 + note segmentation |
| mir | `dsp` | autocorrelation tempo、DP beat tracking、KS key finding、novelty 段落、RMS 響度 |
| fx | `heuristic` | stationary 高頻噪音趨勢 → riser / downlifter / white_noise / impact |
| sound_analysis | `dsp` | ADSR、spectral、harmonicity、inharmonicity、vibrato / tremolo / filter movement → sound_design 描述與 patch 近似候選 |
| render | `additive` / `fluidsynth` | 加法合成（預設）/ SoundFont（需自行提供 .sf2，絕不自動下載） |

測試用的 `fake` / `failing` adapters 則用來驗證 pipeline、resume、cache 與 partial failure。

### 選用的神經模型 adapters（需自行安裝依賴）

| kind | name | 需求 | 啟用方式 |
| --- | --- | --- | --- |
| separation | `demucs` | `pip install torch demucs`，然後 `mrep models download separation demucs`（下載權重到 torch hub cache） | `--set stages.separation.broad_adapter=demucs` |
| transcription | `basic_pitch` | `pip install basic-pitch`（模型隨套件附帶） | `--set stages.transcription.pitched_adapters=[spectral_peaks,basic_pitch]` |

這兩個 adapter 以 lazy import 實作，沒裝依賴時 `available()` 為 False，pipeline 會回報 `AdapterUnavailable` 並使用 `fallback_adapter`。
對應的 integration tests 在依賴缺席時自動略過。

## 測試

```bash
pytest                      # unit + contract + e2e（不載任何模型，約 1–2 分鐘）
pytest tests/unit -q        # schema / quantization / fusion / MIDI mapping / manifest
pytest tests/contract -q    # pipeline + resume + cache + partial failure（fake adapters）
pytest tests/e2e -q         # 合成歌曲跑真實 baseline adapters，對照 ground truth
pytest -m integration       # 需要外部工具 / 真實模型（預設跳過）
```

所有測試音訊都由 `mrep.utils.synthetic` 現場生成；repository 不含任何商業歌曲。

## 授權

MIT，見 `LICENSE`。
