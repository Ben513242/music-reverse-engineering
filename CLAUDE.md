# CLAUDE.md

本 repo 是 **MREP — Music Reverse Engineering Pipeline**（Python，`src/mrep`）。
規格：`docs/SPEC.md`。架構：`ARCHITECTURE.md`。給 agent 的規範：`AGENTS.md`（必讀，含 15 條硬性規範）。

## 常用指令

```bash
pip install -e ".[dev]"        # 安裝（core: numpy scipy soundfile PyYAML mido）
pytest -q                      # unit + contract + e2e（不載模型，1–2 分鐘）
pytest tests/unit -q           # 最快的回饋
ruff check src tests           # lint
mrep synth demo.wav && mrep run demo.wav --profile standard   # 手動端到端
mrep info output/demo          # 看 manifest / 錯誤
```

## 程式碼地圖

* `src/mrep/cli.py` — argparse CLI（run / analyze / separate / transcribe / score / reconstruct / info / synth / adapters / models）
* `src/mrep/pipeline/runner.py` — stage 依賴圖、resume、cache、force-stage、partial failure
* `src/mrep/pipeline/<stage>.py` — 九個 stage，各自只讀寫 artifacts
* `src/mrep/adapters/` — 模型程式碼（`base.py` 介面、`registry.py` 註冊；`<kind>/<name>.py` 實作；`fake.py` 給測試）
* `src/mrep/analysis/` — fusion 規則、和弦、琶音、量化（純規則，可單元測試）
* `src/mrep/domain/` — labels / events / tracks / project / provenance（不得 import ML 框架）
* `src/mrep/io/` — audio / json / midi / musicxml
* `config/` — default + profiles + drum_map + gm_programs

## 改動時注意

* 新欄位先改 `domain`，再補 `to_dict/from_dict` roundtrip 測試。
* 改 stage 輸出路徑時同步更新 `ARCHITECTURE.md` 的表格與 `tests/contract/test_pipeline_fake.py` 的檔案清單。
* Heuristic adapters 的門檻值請放在 `self.config`（可由 yaml 覆寫），不要寫死。
* 不要在 commit / PR / 程式碼註解寫入模型識別字串以外無關的內容；不要提交 `output/`、模型檔或音訊。
