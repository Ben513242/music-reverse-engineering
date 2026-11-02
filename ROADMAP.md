# ROADMAP

| Milestone | 內容 | 狀態 |
| --- | --- | --- |
| **M0 — 骨架** | Repository、Python package、CLI、config profiles、manifest、stage runner、cache / resume、fake adapters、unit + contract tests | ✅ 完成 |
| **M1 — Vertical slice** | Audio → normalize → basic separation → tempo → kick/snare/hat → one pitched AMT → MIDI → 合成渲染 → WAV（全部以 baseline DSP adapters 跑通） | ✅ 完成（baseline 品質） |
| **M2 — Specialist separation** | Piano / Guitar / Synth / Strings / Bass / Vocal 專用分離模型 adapters（依 discovery 機率觸發） | 🔜 架構就緒（`specialists:` 設定 + gating 已實作，待接模型） |
| **M3 — 電子音樂細分類** | 808 / Sub Bass / Lead / Pad / Arp / Pluck / Chord / Drone；Riser / Downlifter / Impact / Noise / Glitch / Granular | 🟡 taxonomy、label 結構、FX events、琶音偵測、synth 音色分析 + patch 近似候選已有；分類模型待接 |
| **M4 — 樂譜** | MusicXML、鼓譜、和弦記號、轉調、動態記號、拍號 | ✅ 完成（writer + 量化 + harmony + dynamics + key change + 拍號偵測 + articulation / slur） |
| **M5 — Two-path AMT** | Per-stem AMT + Full-mix AMT → fusion；多 adapter ensemble consensus | ✅ 機制完成（`full_mix_amt`, `pitched_adapters[]`, `drums_adapters[]`）；`basic_pitch` adapter 骨架已加，待實測 |
| **M6 — Reconstruction 品質** | Velocity / dynamics / humanization / articulation / instrument mapping / FX / mixing | 🟡 velocity、micro-timing、articulation、pitch bend / vibrato、FX 回填已有；SoundFont adapter 需自備 .sf2；渲染端尚未套用 articulation |

## 近期優先（建議順序）

1. **實測 `basic_pitch` adapter**（已寫好骨架）並在 `standard` / `max_quality` 與 `spectral_peaks` 並列，由 fusion 做 consensus。
2. **實測 `demucs` adapter**（已寫好骨架）作為 `broad_adapter`；`hpss` 退為 `fallback_adapter`。
3. **鼓轉譜模型**（ADT）取代 `onset_bands`；保留 band 規則作為 fallback。
4. **學習型 classification**（instrument / role / timbre tagging）取代 `heuristic`，輸出 multi-label 機率。
5. **Beat / downbeat / key 模型**（madmom 類）取代 `dsp` MIR 的對應部分。
6. Synth sound analysis（ADSR、spectral、modulation）與 **patch approximation（candidate + similarity）**。
7. Humanization metadata → 渲染（legato / staccato / vibrato / pitch bend）。
8. Quality metrics 腳本：對合成與公有領域音訊計算 SDR / F1 / tempo error 並出報表。

## 長期

DAW project export（Ableton / Reaper / Logic MIDI）、timbre similarity search、互動修正 UI、
waveform / piano-roll / score editor、自動配器、原曲 vs 重建 A/B 比較。
