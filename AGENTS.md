# AGENTS.md — 給 coding agents（Codex / Claude Code / Cursor / Copilot）

> **先建立可靠、可測試、可替換模型、可續跑的架構，再逐步提高 AI 模型品質；
> 不得為了快速展示結果，把所有功能塞進單一 Python 腳本。**
>
> 成功標準不是「一次就完美拆出所有樂器」，而是：**每加入一個更好的模型，都能直接替換 Adapter，
> 不需要重新設計整個系統。**

## 開始前

1. 讀 `docs/SPEC.md`（需求）與 `ARCHITECTURE.md`（實作結構）。
2. 安裝：`pip install -e ".[dev]"`；跑 `pytest -q` 確認綠燈再動手。
3. 看 `TASKS.md` 與 `ROADMAP.md` 了解目前里程碑；要接模型整合請讀 `docs/HANDOVER.md`。

## 硬性規範（違反即不合格）

1. 不建立 monolithic script；不把專案做成單一 Notebook。
2. `src/mrep/domain/` 與 `src/mrep/analysis/` 不得 import PyTorch / TensorFlow / JAX。
3. ML 模型程式碼只能放在 `src/mrep/adapters/`；重型依賴在方法內 lazy import。
4. `import mrep` 不得自動下載模型；下載只能發生在 `mrep models download <kind> <name>`（`ensure_models(download=True)`）。
5. Raw events（`03_events/*.jsonl`、`raw/`）不得被量化結果覆蓋；量化結果寫到 `05_score/score_events.jsonl`。
6. FX（riser / impact / noise…）不得硬轉為一般音符；保留 clip + features + trigger note。
7. Synth preset 不可假裝精確辨識；只能描述音色或給 candidate / probability / similarity。
8. 所有 AI 結果必須有 `provenance`（stage / adapter / model / model_version / run_id / input / parameters_hash）。
9. 所有 AI 結果必須有 `confidence`（0–1）；沒有就 `null` + `confidence_method: "unavailable"`，不可偽造。
10. Pipeline 必須可 `--resume`；參數改變或上游重跑要讓 cache 失效。
11. CPU fallback 必須存在（`fallback_adapter` / baseline adapters）。
12. 單一模型失敗 → `StageResult.add_error()` → 專案 `PARTIAL`，其餘照常。
13. 大型模型不得成為基本功能的必要條件。
14. 鼓 MIDI mapping / GM program 等對應表只能放在 `config/*.yaml`。
15. 不把商業歌曲放進 repository；測試音訊用 `mrep.utils.synthetic` 生成。

## 新增一個模型 adapter 的步驟

1. `src/mrep/adapters/<kind>/<name>.py`，繼承對應 base class，加 `@register("<kind>", "<name>")`。
2. 在 `available()` 檢查依賴；在 `ensure_models()` 檢查模型檔，缺檔 raise `AdapterUnavailable`（`download=True` 才允許下載）。
3. 設定 `model` / `model_version` 類別屬性，provenance 會自動帶上。
4. 把模組加入 `adapters/registry._BUILTIN_MODULES`，或在 config 用 `"my_pkg.module:MyAdapter"` 指定。
5. 在 `config/*.yaml` 以名稱選用；需要時加 `fallback_adapter`。
6. 加 `tests/unit`（結果 schema）與 `@pytest.mark.integration` 測試（真的載模型）。
7. 不要修改 pipeline / domain 來遷就模型；需要新欄位時先改 domain 並補測試。

## 技術稽核與驗收（2026-11-18）

開始修復或整合新模型前，先讀 [`docs/TECHNICAL_AUDIT_2026-11-18.md`](docs/TECHNICAL_AUDIT_2026-11-18.md)，與 [`TASKS.md`](TASKS.md) 的稽核清單。本節是**額外驗收指引**，不表示問題已修復：

- 先補回歸測試：量化版多軌 MIDI 的 channel／program 互不干擾；6/8 在 MIDI 與 MusicXML 一致；full-mix AMT 的多 Adapter 設定語意明確。
- 評估應將整體音高辨識與「正確樂器／軌道歸屬」分開，缺失軌不可自動用全部預測音符冒充；鼓組同時報 class-agnostic onset 與 class-aware F1。
- `confidence` 預設視為未校準分數；不可在沒有校準與標註評估的情況下當成真實機率。
- 任何宣稱 ML 已啟用、已改善品質，必須提供真正執行的 integration／benchmark、依賴版本、模型識別及基準對照。**被 skip 的 integration test 不算通過。**
- MREP 的音訊→MIDI 辨識誤差必須與未來 Minecraft／其他演奏器的時序與音域映射誤差分開報告；匯出器只應是可選擇模組。

## 工作習慣

* 小步提交，每個 PR 附測試；`ruff check src tests` 與 `pytest -q` 必須通過。
* 任何 heuristics 都要寫出 `confidence_method` 名稱並在 docstring 說明限制。
* 更新 `TASKS.md`（勾選完成項、新增 follow-ups）。
