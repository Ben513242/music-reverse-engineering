# ARCHITECTURE

MREP 是一條**分層、可續跑、模型可替換**的 pipeline。本文件描述程式碼的實際結構；
需求來源是 [`docs/SPEC.md`](docs/SPEC.md)。

## 分層

```text
cli.py ──► pipeline/runner.py ──► pipeline/<stage>.py ──► adapters/<kind>/<name>.py
                │                        │                        │
             manifest.py            analysis/* (純規則)        ML / DSP 模型程式碼
                │                        │
             domain/* (labels, events, tracks, project)  ◄── 任何層都只透過 domain 物件溝通
```

| 層 | 路徑 | 規則 |
| --- | --- | --- |
| Domain | `src/mrep/domain/` | 純 Python 資料模型。**禁止 import torch / tensorflow / jax**。 |
| Analysis | `src/mrep/analysis/` | 不依賴模型的音樂分析：fusion 規則、和弦、琶音、量化。可單元測試。 |
| Pipeline | `src/mrep/pipeline/` | 每個 stage 一個模組，只讀寫磁碟上的 artifacts 與 domain 物件，透過 `ctx.adapter(kind, name)` 取得模型。 |
| Adapters | `src/mrep/adapters/` | 所有模型相關程式碼。以 `@register(kind, name)` 註冊，由 config 選擇。 |
| IO | `src/mrep/io/` | audio (soundfile / ffmpeg fallback)、JSON/JSONL、MIDI (mido)、MusicXML (自寫 writer)。 |
| Utils | `src/mrep/utils/` | DSP primitives、音樂理論、hash、device、synthetic audio。 |

## Stage 圖與依賴

```text
ingest → normalization ─┬→ global_analysis ─────────────┐
                        └→ separation → classification → transcription → fusion → scoring → reconstruction
```

`runner.py` 中的 `STAGE_DEPS` 定義依賴；`downstream_of()` / `upstream_closure()` 推導
`--force-stage` 與部分執行（`mrep analyze|separate|transcribe|score|reconstruct`）要跑哪些 stage。

| stage | 輸入 | 輸出 | adapter kinds |
| --- | --- | --- | --- |
| ingest | 原始檔 | `manifest.source`（SHA256、duration、sr、channels、project_id） | – |
| normalization | 原始檔 | `01_normalized/audio.wav`, `analysis_mono.wav`, `normalization.json`（含 gain_db） | – |
| global_analysis | normalized | `06_analysis/{tempo,beat,key,structure,dynamics,loudness}.json` | `mir` |
| separation | normalized | `02_stems/*.wav`, `02_stems/<source>/<component>.wav`, `stems.json`, `instruments.json`(discovery) | `discovery`, `separation`（broad + specialists） |
| classification | stems | `03_events/labels.json`（多標籤）, `instruments.json`(labels) | `classification` |
| transcription | stems, labels | `03_events/raw/*.jsonl`（每個 adapter / stem 一檔）, `07_fx/*.wav` | `drum_transcription`, `transcription`, `fx` |
| fusion | raw events | `03_events/{notes,drums,fx}.jsonl`, `06_analysis/{chords,arpeggios}.json`, labels 更新（arpeggio role） | –（`analysis/`） |
| scoring | fused events + global | `04_midi/*.mid`, `04_midi/quantized/`, `05_score/*.musicxml`, `score_events.jsonl`, `07_fx/trigger_map.json` | –（`io/midi`, `io/musicxml`） |
| reconstruction | fused events + fx clips | `08_reconstruction/reconstructed.{wav,mp3}`, `track_renders/`, `reconstruction.json` | `render` |

## Runner：resume、cache、partial failure

* 每個 stage 執行後在 manifest 記錄 `params_hash`（該 stage 的 config 區段 + audio 設定）、
  `upstream_fingerprint`（上游 stage 的 status / run_id / params_hash）與 artifacts 清單。
* `--resume`：stage 為 complete/partial、params_hash 相同、upstream fingerprint 相同、本次 invocation
  上游沒重跑、artifacts 檔案都在 → `cache_hit`，直接跳過。
* `--force-stage X`：X 與所有 downstream stage 被 invalidate。
* 修改 config 中某 stage 的參數 → 該 stage 與其 downstream 自動失效（fingerprint 連鎖）。
* Stage 內部的單一單元失敗（某個 specialist / 某個 AMT adapter / 某個 stem）→ `StageResult.add_error()`，
  stage 狀態 `partial`，專案狀態 `PARTIAL`，其餘照常。
* Stage 整體 crash → `failed`；downstream `skipped`；若是 required stage（ingest / normalization）→ 專案 `FAILED`。
* CLI exit code：0 = COMPLETE、2 = PARTIAL、1 = FAILED。

## Adapter 介面

`adapters/base.py`：

```python
class SeparationAdapter:       def separate(audio, sr, *, source_hint) -> SeparationResult
class DiscoveryAdapter:        def discover(audio, sr) -> DiscoveryResult
class ClassificationAdapter:   def classify(audio, sr, *, stem_name, source_hint, provenance) -> ClassificationResult
class TranscriptionAdapter:    def transcribe(audio, sr, *, track, instrument, provenance) -> list[NoteEvent]
class DrumTranscriptionAdapter:def transcribe_drums(audio, sr, *, track, class_hint, provenance) -> list[DrumEvent]
class MIRAdapter:              def analyze(audio, sr) -> MIRResult
class FXAdapter:               def detect(audio, sr, *, provenance) -> list[FXEvent]
class RenderAdapter:           def render(*, tracks, sample_rate, duration, fx_clips, tempo_bpm, midi_path) -> RenderResult
```

共同規則：

* `__init__(config: dict, device: str)`；`available()` 回報環境是否支援；
  `ensure_models(download=False)` 必須在沒有模型時 **raise `AdapterUnavailable`** 而不是偷偷下載。
* `info()` 提供 name / model / model_version 供 provenance 使用；pipeline 用
  `adapter.provenance(stage, run_id, input, params_hash)` 產生每個事件的 `Provenance`。
* 新增 adapter：建立 `adapters/<kind>/<name>.py`，加上 `@register("<kind>", "<name>")`，
  把模組加入 `registry._BUILTIN_MODULES`（或用 `"package.module:ClassName"` 於 config 直接指定外部 plug-in）。
  重型依賴（torch 等）**只能在方法內 lazy import**。
* 選擇 adapter 只在 config：`stages.separation.broad_adapter`, `stages.separation.specialists.<source>`,
  `stages.transcription.pitched_adapters[]`（多個 = ensemble）、`stages.reconstruction.adapter` 等。
  `fallback_adapter` 提供 CPU / 無模型時的退路。

## Event 模型（`domain/events.py`）

* `NoteEvent`（秒、midi、velocity、pitch_bend、track、instrument、sources）
* `DrumEvent`（time、class、velocity、duration、class_scores）
* `FXEvent`（class、start、end、clip_path、trigger_note、features）
* `ChordEvent`、`KeyEvent`、`TempoEvent`、`BeatEvent`、`SectionEvent`
* `ScoreNoteEvent`（beat 為單位的量化結果，`raw_event_id` 指回原始事件，`timing_offset_beats` 保留 micro-timing）

每個事件都有 `confidence: float|None`、`confidence_method: str`、`provenance: Provenance`。

## 標籤模型（`domain/labels.py`）

`SoundLabel` 是四個獨立軸的多標籤：`source_family`/`source_subtype`（+ `source_candidates`）、
`musical_role[]`、`frequency_role[]`、`sound_design[]`。Taxonomy 以常數定義，`validate()` 回報不合法值。

## 設定

`config/default.yaml` 是基底；`light / standard / max_quality.yaml` deep-merge 於其上；
`--config extra.yaml` 再疊加；`--set a.b.c=value` 最後。`drum_map.yaml`（GM 對應 + 鼓譜記譜）與
`gm_programs.yaml`（GM program）由 `config.load_resource_yaml()` 載入。
套件安裝後 `config/` 會隨 wheel 放到 `mrep/resources/config`；也可用 `MREP_CONFIG_DIR` 指定。

## 測試策略

* `tests/unit`：schema、taxonomy、量化、fusion 規則、MIDI mapping、MusicXML、manifest、config、DSP 基礎、metrics。不載任何模型。
* `tests/contract`：以 `fake` / `failing` adapters 跑整條 pipeline，驗證 artifacts、resume、cache 失效、force-stage、partial failure、CLI。
* `tests/e2e`：以 `mrep.utils.synthetic.demo_song` 產生有 ground truth 的歌曲，跑真實 baseline adapters，比對 tempo / key / downbeat / 鼓 / 音符 / FX。
* `tests/integration`：標記 `integration` + `slow`，需要外部工具或真實模型，預設跳過。
