"""Quality evaluation of a project against ground truth (synthetic songs or annotated audio).

``mrep evaluate output/<song> --truth song.wav.truth.json [--stems reference_stems/]``
writes ``06_analysis/evaluation.json`` and prints a summary table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mrep.io.audio import read_audio
from mrep.io.json import read_json, read_jsonl, write_json
from mrep.metrics import beat_f1, drum_metrics, leakage_db, note_metrics, sdr, si_sdr, tempo_error
from mrep.utils.dsp import to_mono

STEM_ALIASES = {"other": ("other", "harmonic", "piano", "guitar"), "drums": ("drums",), "bass": ("bass",), "vocals": ("vocals", "vocal"), "fx": ("fx",)}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def evaluate_project(project_dir: Path, truth: dict[str, Any], stems_dir: Path | None = None) -> dict[str, Any]:
    project_dir = Path(project_dir)
    analysis = project_dir / "06_analysis"
    events = project_dir / "03_events"
    report: dict[str, Any] = {"project": str(project_dir), "sections": {}}

    # ---- tempo / beats / downbeats / time signature / key ------------------------------------
    tempo = read_json(analysis / "tempo.json") if (analysis / "tempo.json").exists() else {}
    if "bpm" in truth and tempo.get("tempo"):
        report["sections"]["tempo"] = tempo_error(float(truth["bpm"]), float(tempo["tempo"]))
    beat = read_json(analysis / "beat.json") if (analysis / "beat.json").exists() else {}
    if truth.get("beats") and beat.get("beats"):
        report["sections"]["beats"] = beat_f1(truth["beats"], [b["time"] for b in beat["beats"]])
    if truth.get("downbeats") and beat.get("downbeats"):
        report["sections"]["downbeats"] = beat_f1(truth["downbeats"], beat["downbeats"], tol=0.1)
    if truth.get("time_signature") and tempo.get("time_signature"):
        report["sections"]["time_signature"] = {"truth": truth["time_signature"], "estimate": tempo["time_signature"], "correct": truth["time_signature"] == tempo["time_signature"], "confidence": tempo.get("time_signature_confidence")}
    key = read_json(analysis / "key.json") if (analysis / "key.json").exists() else {}
    if truth.get("key") and key.get("global_key"):
        report["sections"]["key"] = {"truth": truth["key"], "estimate": key["global_key"], "correct": truth["key"] == key["global_key"], "key_changes_estimated": max(0, len(key.get("key_events", [])) - 1)}

    # ---- notes -------------------------------------------------------------------------------
    if truth.get("notes") and (events / "notes.jsonl").exists():
        est = [n for n in read_jsonl(events / "notes.jsonl") if n.get("track") != "unassigned"]
        report["sections"]["notes"] = note_metrics(truth["notes"], est, onset_tol=0.08)
        per_track: dict[str, Any] = {}
        for track in sorted({n.get("track", "") for n in truth["notes"]}):
            ref = [n for n in truth["notes"] if n.get("track") == track]
            cand = [n for n in est if n.get("track") == track] or est
            per_track[track] = note_metrics(ref, cand, onset_tol=0.08)["onset_pitch"]
        report["sections"]["notes"]["per_truth_track"] = per_track

    # ---- drums -------------------------------------------------------------------------------
    if truth.get("drums") and (events / "drums.jsonl").exists():
        est_d = [{"time": d["time"], "class": d["class"]} for d in read_jsonl(events / "drums.jsonl")]
        report["sections"]["drums"] = drum_metrics([{"time": d["time"], "class": d["class"]} for d in truth["drums"]], est_d)

    # ---- fx ----------------------------------------------------------------------------------
    if truth.get("fx") is not None and (events / "fx.jsonl").exists():
        est_fx = read_jsonl(events / "fx.jsonl")
        hits = 0
        for t in truth["fx"]:
            dur = max(t["end"] - t["start"], 1e-6)
            if any(e["class"] == t["class"] and _overlap(t["start"], t["end"], e["start"], e["end"]) / dur >= 0.5 for e in est_fx):
                hits += 1
        n_ref, n_est = len(truth["fx"]), len(est_fx)
        p = hits / n_est if n_est else 0.0
        r = hits / n_ref if n_ref else 0.0
        report["sections"]["fx"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "hits": hits, "n_ref": n_ref, "n_est": n_est}

    # ---- chords (frame-wise) -------------------------------------------------------------------
    if truth.get("chords") and (analysis / "chords.json").exists():
        est_c = read_json(analysis / "chords.json").get("chords", [])
        step = 0.25
        total = root_ok = full_ok = covered = 0
        for tc in truth["chords"]:
            t = tc["start"] + step / 2
            while t < tc["end"]:
                total += 1
                ec = next((c for c in est_c if c["start"] <= t < c["end"]), None)
                if ec is not None:
                    covered += 1
                    if ec["root"] == tc["root"]:
                        root_ok += 1
                        if ec["quality"] == tc["quality"]:
                            full_ok += 1
                t += step
        report["sections"]["chords"] = {"root_accuracy": round(root_ok / total, 4) if total else 0.0, "root_quality_accuracy": round(full_ok / total, 4) if total else 0.0, "coverage": round(covered / total, 4) if total else 0.0, "frames": total}

    # ---- separation ---------------------------------------------------------------------------
    if stems_dir is not None and Path(stems_dir).is_dir():
        sep: dict[str, Any] = {}
        refs: dict[str, np.ndarray] = {}
        for ref_path in sorted(Path(stems_dir).glob("*.wav")):
            audio, _sr = read_audio(ref_path)
            refs[ref_path.stem] = to_mono(audio)
        est_dir = project_dir / "02_stems"
        for name, ref in refs.items():
            cand = None
            for alias in STEM_ALIASES.get(name, (name,)):
                if (est_dir / f"{alias}.wav").exists():
                    cand = est_dir / f"{alias}.wav"
                    break
            if cand is None:
                sep[name] = {"status": "no estimated stem"}
                continue
            est_audio, _sr = read_audio(cand)
            est = to_mono(est_audio)
            n = min(len(ref), len(est))
            others = [v[:n] for k, v in refs.items() if k != name]
            sep[name] = {"estimate": cand.name, "sdr_db": round(sdr(ref[:n], est[:n]), 2), "si_sdr_db": round(si_sdr(ref[:n], est[:n]), 2), "leakage_db": round(leakage_db(est[:n], others), 2) if others else None}
        report["sections"]["separation"] = sep

    s = report["sections"]
    report["summary"] = {
        "tempo_within_4pct": s.get("tempo", {}).get("within_4_percent"),
        "beat_f1": s.get("beats", {}).get("f1"),
        "key_correct": s.get("key", {}).get("correct"),
        "time_signature_correct": s.get("time_signature", {}).get("correct"),
        "note_onset_pitch_f1": s.get("notes", {}).get("onset_pitch", {}).get("f1"),
        "drum_onset_f1": s.get("drums", {}).get("onset_f1", {}).get("f1"),
        "fx_f1": s.get("fx", {}).get("f1"),
        "chord_root_accuracy": s.get("chords", {}).get("root_accuracy"),
        "separation_si_sdr_db": {k: v.get("si_sdr_db") for k, v in s.get("separation", {}).items()} or None,
    }
    return report


def format_summary(report: dict[str, Any]) -> str:
    lines = ["metric".ljust(28) + "value", "-" * 40]
    for k, v in report["summary"].items():
        lines.append(f"{k.ljust(28)}{v}")
    return "\n".join(lines)


def run_evaluation(project_dir: Path, truth_path: Path, stems_dir: Path | None = None, out_path: Path | None = None) -> dict[str, Any]:
    truth = read_json(truth_path)
    report = evaluate_project(project_dir, truth, stems_dir)
    report["truth"] = str(truth_path)
    out = out_path or (Path(project_dir) / "06_analysis" / "evaluation.json")
    write_json(out, report)
    report["output"] = str(out)
    return report
