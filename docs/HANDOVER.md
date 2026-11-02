# HANDOVER — 交接給實體機（GPU / torch）coding agent

> 本文件是給下一位在**有 GPU、可安裝 torch 的實體機**上接手的 coding agent（Claude Code / Codex / Cursor 等）。
> 它說明目前完成到哪裡、為什麼停在這裡、接下來要做什麼、怎麼驗收。
> 開始前請先讀 `AGENTS.md`（15 條硬性規範）與 `ARCHITECTURE.md`；需求來源是 `docs/SPEC.md`。

---

## 0. 一句話現況

**架構與所有 model-free 的功能已完成並通過 CI；剩下的工作全部是「把真正的 AI 模型接成 adapter 並實測」。**
目前 `main` 上的 pipeline 以 numpy/scipy baseline adapters 端到端可跑，68 個測試綠燈。

---

## 1. 目前完成的範圍（三個 PR）

| PR | 內容 |
| --- | --- |
| #1 | M0 架構 + M1 垂直切片：domain / config / manifest / 可續跑 runner / 九個 stage / baseline DSP adapters / CLI / 測試 / 文件 |
| #2 | 拍號偵測、`sound_analysis` adapter（ADSR、spectral、modulation、patch 近似候選）、pitch bend / vibrato、humanization、低信心標記、鼓 ensemble、`demucs` 與 `basic_pitch` adapter **骨架** |
| #3 | Viterbi 調性分段、tempo 曲線、humanization 套用到渲染、`mrep evaluate`、`mrep synth --stems`、DAW export（session.json + Reaper .rpp）、ogg/m4a/flac/48 kHz 輸入測試 |

里程碑對照（`ROADMAP.md`）：M0 ✅、M1 ✅、M4 ✅、M5 機制 ✅、M3 機制 🟡（缺分類模型）、M6 🟡（缺 SoundFont 以外的高品質渲染）、M2 🔜（缺 specialist 分離模型）。

### 為什麼停在這裡

開發環境是雲端容器，透過 proxy 無法安裝 torch（`pip install torch` 找不到任何版本），也不能下載模型權重。
因此 `separation/demucs` 與 `transcription/basic_pitch` 兩個 adapter **只依公開 API 撰寫、從未執行過**，
只有「缺依賴即略過」的 integration tests 守著。這是實體機上的第一件事。

---

## 2. 環境建置（實體機）

```bash
git clone https://github.com/Ben513242/music-reverse-engineering.git
cd music-reverse-engineering
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 先確認 68 個測試綠燈（不載模型，約 1 分鐘）
ruff check src tests

# 選用的模型依賴（依 CUDA 版本選 torch）
pip install torch torchaudio    # 依 https://pytorch.org 的指示挑選對應 CUDA 的指令
pip install demucs
pip install basic-pitch         # 依官方 README 選擇推論後端（tensorflow / onnxruntime / coreml / tflite）

# 模型權重只能明確下載（AGENTS.md 第 4 條）：
mrep models download separation demucs
mrep adapters                   # 應列出 separation/demucs 與 transcription/basic_pitch
```

GPU 選擇：`--device auto|cpu|cuda|mps`（`mrep.utils.device.resolve_device`）。CPU 永遠是合法 fallback。

---

## 3. 接手後的工作順序與驗收標準

### 3.1 實測 `separation/demucs`（最優先）

檔案：`src/mrep/adapters/separation/demucs.py`

已知的設計假設，實測時逐一確認：

1. `demucs.pretrained.get_model(name)` 與 `demucs.apply.apply_model(model, wav[None], device=..., shifts, split, overlap, progress=False)` 的簽名。
2. `model.samplerate`（預期 44100）、`model.sources`（htdemucs = drums/bass/other/vocals；htdemucs_6s 另有 guitar/piano）。
3. `ensure_models(download=False)` 以 `torch.hub.get_dir()/checkpoints/*.th` 是否存在判斷權重；若 demucs 改了快取位置，請改這個判斷而不是放寬規則。
4. 正規化：目前照 demucs `separate.py` 的做法（以 mix 的 mean/std 標準化再還原）。

驗收：

```bash
pytest -m integration tests/integration/test_ml_adapters.py::test_demucs_adapter_separates_demo -q
mrep synth demo.wav --bars 8 --stems ref_stems/
mrep run demo.wav --profile standard --set stages.separation.broad_adapter=demucs --set stages.separation.fallback_adapter=hpss
mrep evaluate output/demo --truth demo.wav.truth.json --stems ref_stems/
```

- `06_analysis/evaluation.json` 的 `separation.drums.si_sdr_db` 應明顯高於 HPSS baseline 的 **2.0 dB**（bass 9.8、other 11.7 為 baseline 參考值）。
- `manifest.json` 的 `stages.separation.adapters` 必須出現 `demucs` 與 `model_version`（provenance，AGENTS.md 第 8 條）。
- 沒有權重時 `mrep run` 必須走 `fallback_adapter`，專案狀態不可 FAILED。
- 通過後：把 `config/standard.yaml` / `max_quality.yaml` 的 `broad_adapter` 改為 `demucs`、`fallback_adapter: hpss`，README 的「選用的神經模型 adapters」表格改成「已實測」。

### 3.2 實測 `transcription/basic_pitch`

檔案：`src/mrep/adapters/transcription/basic_pitch.py`

已知假設：`basic_pitch.inference.predict(path, model_or_model_path=ICASSP_2022_MODEL_PATH, onset_threshold, frame_threshold, minimum_note_length(ms), minimum_frequency, maximum_frequency, multiple_pitch_bends, melodia_trick)` 回傳 `(model_output, midi_data, note_events)`，`note_events` 每筆為 `(start_s, end_s, pitch_midi, amplitude, bends|None)`，bends 單位為 1/3 半音。新版 API 若不同，改 adapter 不改 pipeline。

驗收：

```bash
pytest -m integration tests/integration/test_ml_adapters.py::test_basic_pitch_adapter_transcribes_tone -q
mrep run demo.wav --profile standard --set "stages.transcription.pitched_adapters=[spectral_peaks,basic_pitch]"
mrep evaluate output/demo --truth demo.wav.truth.json
```

- `notes.onset_pitch.f1` 應高於 baseline **0.61**（bass 子集的 recall 尤其要改善；HPSS 把 kick 尾音漏進 bass stem 造成的假音符是 baseline 的主要弱點）。
- fusion 的 consensus 已支援多 adapter（`analysis/fusion_rules.merge_note_runs`），檢查 `03_events/fusion.json` 的 `notes.<stem>.adapters == 2`。
- 通過後把 `standard` / `max_quality` 的 `pitched_adapters` 加入 `basic_pitch`。

### 3.3 鼓轉譜模型 adapter（新寫）

- 介面：`DrumTranscriptionAdapter.transcribe_drums(audio, sr, *, track, class_hint, provenance) -> list[DrumEvent]`，註冊 `@register("drum_transcription", "<name>")`，加入 `adapters/registry._BUILTIN_MODULES`。
- 類別名稱必須用 `domain/labels.py` 的 `DRUM_CLASSES`；模型沒有 confidence 時寫 `None` + `confidence_method="unavailable"`。
- 設定：`stages.transcription.drums_adapters: [onset_bands, <name>]`（ensemble），fusion 會做 consensus。
- 驗收：`evaluate` 的 `drums.onset_f1.f1` 高於 baseline **0.71**，且 `per_class.closed_hat.recall` 明顯高於目前（baseline 幾乎分不出 closed/open hat）。

### 3.4 學習型多標籤 classification adapter（新寫）

- 介面：`ClassificationAdapter.classify(...) -> ClassificationResult(label: SoundLabel, features)`。
- 輸出必須是 multi-label（`source_family` + `source_candidates` 機率、`musical_role[]`、`frequency_role[]`、`sound_design[]`），不可只給單一 instrument 字串（SPEC §2）。
- `energy_heuristic` discovery 目前對 piano/synth/strings/guitar/vocal 回 `null`，所以 specialist 需要 `force_specialists`；有了分類模型後請同時提供 `DiscoveryAdapter`（kind `discovery`），讓 `specialist_threshold` gating 真正生效。

### 3.5 Beat / downbeat / key 模型 adapter（`mir/`）

- 介面：`MIRAdapter.analyze(audio, sr) -> MIRResult`。可以只覆蓋部分欄位：例如用模型給 `beat_times`/`downbeat_times`，其餘沿用 `dsp` 的做法（可在 adapter 內部呼叫 `DSPMIR` 再覆寫欄位）。
- 拍號：模型若直接給拍號，填 `time_signature` 與 `time_signature_method="model"`；否則保留 `analysis/time_signature.estimate_time_signature`。
- 驗收：`evaluate` 的 `beats.f1`（baseline 0.96）與 `downbeats.f1` 不退步，並在真實音樂上測。

### 3.6 用真實音樂驗證（所有模型接上後）

- Repository **不得**放商業錄音（AGENTS.md 第 15 條）。請在本機建立 `benchmarks/`（已在 `.gitignore` 忽略音訊），放公有領域或自製錄音，必要時以 MIDI 渲染產生 ground truth。
- 每首跑 `mrep run` + `mrep evaluate`，把 `06_analysis/evaluation.json` 的 `summary` 彙整成表，比較 baseline 與模型 adapter。
- Heuristic 門檻都在 `config/*.yaml` 與各 adapter 的 `self.config`（AGENTS.md：不可寫死），調整時改 yaml。

---

## 4. 新增 adapter 的固定步驟（摘自 AGENTS.md）

1. `src/mrep/adapters/<kind>/<name>.py`，繼承對應 base class，`@register("<kind>", "<name>")`。
2. `available()` 檢查依賴；`ensure_models(download=False)` 缺模型就 raise `AdapterUnavailable`，只有 `download=True` 才下載。
3. 重型依賴（torch 等）只能在方法內 lazy import；`import mrep` 不得觸發下載。
4. 設定 `model` / `model_version` 類別屬性（provenance 自動帶上）。
5. 加入 `adapters/registry._BUILTIN_MODULES`。
6. `config/*.yaml` 以名稱選用；提供 `fallback_adapter`。
7. 測試：`tests/unit`（結果 schema）+ `tests/integration`（`@pytest.mark.integration`、缺依賴時 `pytest.skip`）。
8. 更新 README 的 adapters 表格、`TASKS.md`、`CHANGELOG.md`。

---

## 5. 程式碼地圖（接手最常碰的地方）

| 想做的事 | 看哪裡 |
| --- | --- |
| 改 stage 流程、resume / cache | `src/mrep/pipeline/runner.py` |
| 分離 / 分類 / 轉譜 / 融合 / 樂譜 / 重建 各 stage | `src/mrep/pipeline/<stage>.py` |
| adapter 介面與結果 dataclass | `src/mrep/adapters/base.py` |
| 註冊表 | `src/mrep/adapters/registry.py` |
| 事件 / 標籤 / provenance | `src/mrep/domain/` |
| 融合規則、和弦、琶音、量化、拍號、調性 Viterbi、pitch bend、humanization、低信心 | `src/mrep/analysis/` |
| 評估指標與 `mrep evaluate` | `src/mrep/metrics.py`, `src/mrep/evaluate.py` |
| 合成測試歌曲（含參考 stems） | `src/mrep/utils/synthetic.py` |
| MIDI / MusicXML / DAW 匯出 | `src/mrep/io/midi.py`, `musicxml.py`, `daw.py` |
| 設定與 profiles | `config/*.yaml` |

---

## 6. 已知限制與陷阱

- **Baseline 的頻率參數以 44.1 kHz 調校**；輸入會先重採樣到 44.1 kHz。不要把 `audio.target_sample_rate` 改成 22.05 kHz。
- **6/8 偵測是選用**（`max_quality` 的 `time_signature.candidates: [3, 4, 6]`）：在 downbeat 重音不明顯的素材上會過度擬合；預設只判 3/4 vs 4/4。
- **HPSS 會把 kick 尾音漏進 bass stem**，低音 AMT 因此有假音符；fusion 的 `drum_leak_filter` 只擋短音。接上 demucs 後此問題應大幅改善。
- **`onset_bands` 鼓分類**：pluck 類樂器 transient 漏進鼓 stem 時會誤判為 snare；open/closed hat 以高頻衰減時間區分，不穩定。
- **Reaper `.rpp`** 只有 stem 音軌、markers、tempo envelope；MIDI 請從 `04_midi/` 匯入（RPP 內嵌 MIDI 尚未實作）。
- **`raw` 與 `score` 事件分離**是硬性規範：量化永遠寫到 `05_score/score_events.jsonl`，不可改 `03_events/`。
- Contract tests 用 `fake` / `failing` adapters，不要把真實模型塞進去；真實模型只能在 `tests/integration`。

---

## 7. 交付檢查清單（每個 PR）

- [ ] `ruff check src tests` 通過
- [ ] `pytest -q` 通過（不載模型）；`pytest -m integration` 在有依賴的機器通過
- [ ] `mrep run` 在合成歌曲上狀態 `COMPLETE`，`mrep evaluate` 的 summary 貼進 PR 說明並與 baseline 比較
- [ ] 新 adapter 有 provenance（`model`, `model_version`）與 confidence（或 `null` + method）
- [ ] 缺模型 / 缺依賴時 pipeline 會 fallback，不會 FAILED
- [ ] `README.md` adapters 表、`TASKS.md`、`CHANGELOG.md` 已更新
- [ ] 沒有提交 `output/`、模型檔、音訊

---

## 8. Baseline 參考數字（合成歌曲 `mrep synth --bars 8`，profile `standard`）

| 指標 | baseline |
| --- | --- |
| tempo 誤差 | 0.2 bpm（4% 內） |
| beat F1 | 0.96 |
| 調性 / 拍號 | C major / 4/4 正確 |
| note onset+pitch F1 | 0.61 |
| drum onset F1 | 0.71 |
| FX F1 | 1.0 |
| chord root accuracy | 1.0 |
| SI-SDR（HPSS） | bass 9.8 dB、other 11.7 dB、drums 2.0 dB |

任何模型 adapter 接上後，先確認這張表不退步，再追求提升。
