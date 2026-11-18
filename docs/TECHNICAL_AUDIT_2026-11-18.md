# MREP 技術稽核與品質驗收計畫（2026-11-18）

> **性質：靜態原始碼／文件稽核；不是程式修正，也不是重新執行的實測報告。**  
> 審查基準：`main` commit [`1734ef1`](https://github.com/Ben513242/music-reverse-engineering/commit/1734ef1b7b7372da402ea6555982a8b0a9650178)（2026-11-02）；審查日期：2026-11-18。  
> 依據：`README.md`、`ARCHITECTURE.md`、`ROADMAP.md`、`TASKS.md`、`docs/HANDOVER.md`、`config/*.yaml`、`src/mrep/{pipeline,adapters,analysis,io,metrics}.py`、`tests/` 與 GitHub CI 記錄。**本次沒有在 GPU 上執行模型或以商業錄音重測。**

## 1. 核心判斷與能力邊界

MREP 目前是**具備端到端 baseline 的可續跑、模型可替換、可追溯的音樂逆向工程框架**。它不是已通過真實混音基準驗證的「任何歌曲精準還原系統」。

| 子系統 | 原始碼可確認的現況 | 不應宣稱的能力 |
| --- | --- | --- |
| Ingest、正規化、manifest、cache、resume、partial failure | 已實作，並有 unit/contract/e2e 測試 | 所有編碼格式、機器與失敗情境均實機驗證 |
| 分離（`hpss`、`drum_bandsplit`） | DSP baseline，可產生 drums/bass/other 與頻帶分解 | 鋼琴、弦樂、吉他、合成器已被準確分離 |
| 音符／鼓轉譜（`spectral_peaks`、`onset_bands`） | 已有起音、音高及鼓類別候選事件 | 複雜和弦、重疊泛音、open/closed hat 已達專業精度 |
| MIR／樂譜／重建 | 已有 tempo、beats、key、chords、MusicXML、MIDI、FX 音訊片段與 additive / SoundFont 路徑 | 樂譜、音色、音量、演奏法等同於原始製作工程 |
| `demucs`、`basic_pitch` | Adapter 程式碼存在，可選用；`docs/HANDOVER.md` 明示尚未在實體 GPU / 推論環境驗證 | 已完成模型整合或證明優於 DSP baseline |
| Quality evaluation | `mrep evaluate` 支援合成 ground truth 和多種評估指標 | CI 通過即代表真實歌曲解析準確 |

### 「任何音樂」的工程定義

支援廣泛的歌曲**輸入與產生候選分析結果**，不代表能從單一混音**唯一恢復**原始 DAW 軌道、各樂器 MIDI、預設音色及效果鏈。這是資訊不充分的逆問題；相同音高的重疊樂器、混響、sidechain、合成器調變與聲部同頻都可能存在多個合理解。輸出應分為「可觀測描述」「推定事件」「不可識別／低信心」，並保留原始音訊與來源證據。

## 2. 證據與指標的解讀

- `docs/HANDOVER.md` 記載 68 個 baseline 測試通過；提交 `1734ef1` 的 [GitHub Actions](https://github.com/Ben513242/music-reverse-engineering/actions/runs/41431304138) 為 `success`。此證據支持程式檢查、合成音訊端到端流程；**不支持真實音樂的一般化準確率宣稱**。
- 同份交接文件記載的合成音訊參考值：beat F1 `0.96`、note onset+pitch F1 `0.61`、drum onset F1 `0.71`；HPSS SI-SDR 分別為 drums `2.0 dB`、bass `9.8 dB`、other `11.7 dB`。這些數字是**既有文件的 baseline**，不是本次重算，也不能當成公開歌曲的績效。
- 目前 `max_quality` 仍使用 `spectral_peaks` 和 `onset_bands`，且未將 `demucs` 設為預設；profile 名稱不代表已使用高品質 ML 模型。
- `confidence` 目前可能是啟發式分數、模型 amplitude 或融合分數；除非做過可靠度校準，不應將其稱為實證機率。

## 3. 可從原始碼定位的問題與驗收條件

下表是**待修復／待驗證事項**。這份稽核文件不更動執行程式，因此任何項目都不得勾選「已修復」。

| ID／優先級 | 來源與觀察 | 風險與建議驗收 |
| --- | --- | --- |
| **A01／P1** | `src/mrep/pipeline/scoring.py`：量化 `FULL_SONG_quantized.mid` 建立各 pitched `MidiTrackSpec` 時一律 `channel=0`。 | 多聲部 Program Change 共用 MIDI channel，音色可能覆蓋。應分配合法且互不衝突的 pitched channel（避開 ch9 鼓），加入兩種以上樂器同時演奏的 round-trip 測試。 |
| **A02／P1** | `scoring.py` 對 `build_midi` 傳入 `(bpb, 4)`；`analysis/time_signature.py` 會把 6 tracked beats 定義為 `6/8`。 | 在 6/8 情境可能寫成 6/4。應明確傳遞分子／分母並用 MIDI meta event round-trip、MusicXML 拍號一致性測試。 |
| **A03／P1** | `pipeline/transcription.py`：`full_mix_amt` 路徑只跑 `pitched_adapters[0]`，per-stem 才遍歷所有 adapter。 | 多 Adapter 設定對 full-mix 不構成真正 ensemble；應明訂設計目標（單模型或多模型），若要求多模型則逐個執行並保留 raw/provenance，測試 fallback 與 fusion。 |
| **A04／P1** | `src/mrep/evaluate.py`：某真值軌沒有同名預測軌時，`cand = same_track or est` 改用所有預測音符。 | `per_truth_track` 不再衡量聲部歸屬，可能給出誤導分數。應分別報告「無對應軌道」、「全域音高偵測」及「正確樂器分配」三類指標。 |
| **A05／P1** | `src/mrep/metrics.py`：`drum_metrics().onset_f1` 只比時間；類別混淆與 per-class F1 另列。 | 不可將整體 onset F1 當成「鼓組種類辨識率」。應增加 class-aware micro/macro F1，清楚標示 class-agnostic 指標。 |
| **A06／P2** | `analysis/fusion_rules.py`：`combine_confidence` 使用 noisy-OR。 | 若 Adapter 高度相關，融合分數可能偏高。應保留原始分數並用校準集檢查 reliability / ECE / Brier；不要把未校準值稱為機率。 |
| **A07／P1（實測）** | `adapters/separation/demucs.py` 的權重存在檢查採 `checkpoints/*.th`。`docs/HANDOVER.md` 也列此為未驗證假設。 | 任意 .th 存在不等於**指定模型**權重就緒；需以實際載入與可離線推論驗證，確認無隱式下載、錯誤模型與 fallback 行為。 |
| **A08／P0（能力驗證）** | `config/default.yaml` 仍以 `hpss`／`spectral_peaks`／`onset_bands` 為主，`tests/integration/test_ml_adapters.py` 依賴缺席時 skip。 | 優先在有可用模型與 GPU 的環境測實際 Demucs、Basic Pitch，建立同一資料集下的 baseline 對照；不能以 skip 當作 ML 已驗證。 |

> P0／P1／P2 是**審查建議優先級**，不是已開立 issue 或已修復的標籤。A07 的權重問題屬需要實機驗證的風險，不預先判定在所有版本必然故障。

## 4. 建議的驗證里程碑

1. **QG-1：先修輸出正確性** — 為 A01–A05 新增最小失敗案例與 unit / contract tests；`ruff check src tests`、`pytest -q`、MIDI / MusicXML 寫出後重讀均須通過。
2. **QG-2：模型整合** — 依 `docs/HANDOVER.md` 在 GPU 機器實測 Demucs、Basic Pitch；無權重時能報錯且按設定 fallback；明確記錄 model/version、device、run ID、模型權重識別與 runtime。
3. **QG-3：基準資料集** — 使用可合法取得並遵循授權的分離／轉譜資料集（例如 Slakh2100、MUSDB18-HQ，視各自標註及授權可用範圍）；建立固定資料切分、版本、容忍時間窗與可重現評估。避免在測試資料上調參再報同一份測試分數。
4. **QG-4：真實歌曲案例** — 使用自製／合法授權音樂，加上本機保留的複雜混音案例，人工標註部分 Bass／Lead／Harmony／Kick／Snare／Hat，進行逐軌 A/B 比對。不可把商業錄音提交公開 repository。
5. **QG-5：編輯與重建** — 能指出低信心音符，人工更正後重生 performance MIDI、quantized MIDI、MusicXML 與重建音訊。記錄原曲與合成結果的同步、遺漏、錯音及殘差；FX 原始片段回填不能直接當成完全重建成功。

### 應固定輸出的品質欄位

| 項目 | 最低評估 |
| --- | --- |
| 分離 | SDR／SI-SDR（相同資料與來源定義）、不同源 leakage、缺失 stem |
| 音高事件 | note onset+pitch F1、onset+offset F1、每軌 precision／recall、八度錯誤 |
| 鼓 | class-agnostic onset F1 **與** class-aware per-class／macro F1 |
| MIR | BPM、beat／downbeat F1、調性與拍號（含 6/8） |
| 重建 | 原聲／重建對齊、分軌音色映射、主觀 A/B 盲聽或受控評分 |
| 系統 | 完成／PARTIAL／FAILED、fallback、resume、runtime、VRAM、provenance、可重現設定 |

沒有真值的真實歌曲，只能呈現**候選結果、信心區間及人工聽感**，不得自行宣稱「X% 還原準確率」。

## 5. 與 Minecraft 原版音符盒的後續整合（提案；尚未實作）

以 MREP `03_events/notes.jsonl`（raw 時間）、`03_events/drums.jsonl` 與 `04_midi/` 作為輸入，另建 `exporters/minecraft_noteblock` **可選擇匯出層**。獨立處理：

- 音域、樂器映射與音高移位：原版音符盒每種音色只提供 25 階音高，不應修改 raw 音符。
- 節奏：將實際秒數映射為遊戲 tick、礦車行駛時間與感測鐵軌觸發事件；需量測速度、曲線減速與累積漂移。
- 軌道：每個聲部一條平行礦車軌，全部使用同一基準時鐘／同步起點；礦車位置必須是起始軌道**方塊中心**，並測試生成、停止、第二圈重播與區塊載入。
- 驗證：分開保存「音檔 → MIDI」的辨識誤差與「MIDI → Minecraft」的紅石／遊戲時序誤差，避免把兩者混為一談。

這是**未來擴充提案**，不是 MREP `main` 已具備的 exporter，也不要求現在引入 Minecraft 遊戲依賴。

## 6. 交付與溝通規則

- 本文件記錄 **verified-in-source（原始碼可見）／reported-in-docs（文件記載）／needs-runtime-validation（需要實測）／proposal（建議）** 四種不同證據等級。
- 對外寫「能分析多類型音樂」而不是「可無損恢復任何歌曲」。所有性能主張要附模型、版本、資料集、分割方式、容差與指標。
- 工程修復須另開程式碼 PR；本稽核 PR 只修改文件及待辦，不修改模型、功能、演算法或測試結果。
- 本文件是一次審查快照，未來改動程式碼後須以新的 commit、實測結果更新狀態，不能把勾選測試當成永久精度保證。
