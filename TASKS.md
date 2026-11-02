# TASKS

## Done（M0 / M1 / M4 / M5 機制）

- [x] `pyproject.toml`（hatchling、`mrep` console script、pytest markers、ruff）
- [x] 設定系統：`default.yaml` + `light / standard / max_quality` + `drum_map.yaml` + `gm_programs.yaml`，`--set` 覆寫
- [x] Domain：多標籤 `SoundLabel`、events（Note / Drum / FX / Chord / Key / Tempo / Beat / Section / ScoreNote）、`Provenance`、`Project`
- [x] Manifest（project id、SHA256、config、stage 狀態、adapters、artifacts、confidence、errors、runs）
- [x] Stage runner：依賴圖、`--resume`、params / upstream fingerprint cache、`--force-stage`、partial failure、required stage 失敗中止
- [x] Stages：ingest、normalization、global_analysis、separation（discovery + broad + gated specialists + chunking）、classification、transcription（drums + per-stem AMT + full-mix AMT + ensemble + FX）、fusion、scoring、reconstruction
- [x] Baseline adapters：hpss、drum_bandsplit、energy_heuristic、heuristic classification、onset_bands、spectral_peaks、dsp MIR、heuristic FX、additive render、fluidsynth（opt-in）
- [x] Fake / failing adapters 供 contract tests
- [x] Analysis：fusion 規則（consensus、stem+fullmix、drum leak filter）、chords（template + bass/slash）、arpeggio（pattern / rhythm）、quantize（grid / triplets / swing，raw 與 score 分離）
- [x] IO：audio（soundfile + ffmpeg fallback）、JSON/JSONL、MIDI（tempo map、drum channel、FX trigger track）、MusicXML（voices、ties、chords、harmony、dynamics、percussion、key change）
- [x] Metrics：SDR / SI-SDR / leakage、note P/R/F1、drum onset F1 + confusion、tempo error、beat F1
- [x] CLI：run / analyze / separate / transcribe / score / reconstruct / info / synth / profiles / adapters / models download
- [x] Tests：unit、contract（pipeline / resume / cache / force / partial / CLI）、e2e（合成歌曲 vs ground truth）、integration placeholder
- [x] Docs：README、ARCHITECTURE、AGENTS、CLAUDE、ROADMAP、SPEC；CI workflow

## Done（第二輪：M3 / M6 機制）

- [x] 拍號偵測（3/4 vs 4/4，accent periodicity，含 confidence 與分數；可設 `mode: fixed`）
- [x] `sound_analysis` adapter kind：ADSR / spectral / harmonicity / inharmonicity / vibrato / tremolo / filter movement → `sound_design` 描述 + patch 近似候選（probability + similarity）
- [x] Pitch bend / vibrato：單音 stem 的 f0 tracking → `NoteEvent.pitch_bend` / `features.vibrato` → MIDI pitch wheel
- [x] Humanization：articulation（legato / staccato）、timing / velocity 統計 → `humanization.json`；MusicXML staccato / slur
- [x] 低信心處理：`ambiguities.json` + `SoundLabel.ambiguous`
- [x] 鼓 ensemble（`drums_adapters[]` + consensus）
- [x] `demucs`（separation）與 `basic_pitch`（transcription）adapter 骨架（lazy import、明確下載、integration tests 自動略過）

## Next

- [ ] 在有 GPU / torch 的環境實測 `demucs` 與 `basic_pitch` adapters（本次開發環境無法安裝 torch）
- [ ] 鼓轉譜模型 adapter（`drum_transcription/`）
- [ ] 學習型多標籤 classification adapter
- [ ] Beat / downbeat / key 模型 adapter（`mir/`）
- [ ] 6/8 等複拍子（目前候選只有 3/4 與 4/4）
- [ ] Humanization 套用到渲染（articulation 影響 release；vibrato 於 additive render）
- [ ] Key change 偵測更穩健（目前以 20 s 視窗 KS + 相對大小調規則）
- [ ] Tempo map 支援變速（目前 MIDI tempo map 依 beat grid，MusicXML 單一 bpm）
- [ ] Separation quality metrics 腳本（對合成 stems 算 SDR）
- [ ] `.m4a / .aac / .aiff / .ogg` 輸入測試（目前透過 ffmpeg fallback，未測）
- [ ] DAW export（Ableton / Reaper / Logic）

## Known limitations of the baseline adapters

- HPSS 會把 kick 尾音漏進 bass stem，造成低音 AMT 假音符（fusion 的 drum leak filter 只擋得住短音）。
- `onset_bands` 在 pluck 類樂器 transient 漏進鼓 stem 時會誤判為 snare；open / closed hat 以高頻衰減時間區分，不穩定。
- `spectral_peaks` 對低音的頻率解析度有限（已用較長 FFT + 動態鄰音抑制緩解），八度錯誤仍可能發生。
- `energy_heuristic` discovery 只能估 drums / bass / harmonic，其餘家族回 `null`（所以 specialist 需以 `force_specialists` 強制）。
- `heuristic` FX 偵測只處理 riser / downlifter / white_noise / impact。
- Baseline adapters 的頻率參數以 44.1 kHz 調校；`audio.target_sample_rate` 設成 22.05 kHz 時品質明顯下降（e2e 以 44.1 kHz 驗證）。
