"""Baseline reconstruction: additive synthesis for notes, synthesised drums, original FX clips."""

from __future__ import annotations

from typing import Any

import numpy as np

from mrep.adapters.base import RenderAdapter, RenderResult
from mrep.adapters.registry import register
from mrep.domain.events import FXEvent
from mrep.utils import synthetic as syn
from mrep.utils.music import midi_to_hz

PRESETS: dict[str, dict[str, Any]] = {
    "bass": {"harmonics": ((1, 1.0), (2, 0.4), (3, 0.15)), "adsr": (0.005, 0.05, 0.9, 0.08), "gain": 0.8},
    "piano": {"harmonics": ((1, 1.0), (2, 0.6), (3, 0.3), (4, 0.2), (5, 0.1)), "adsr": (0.003, 0.5, 0.25, 0.15), "gain": 0.55},
    "strings": {"harmonics": tuple((h, 1.0 / h) for h in range(1, 9)), "adsr": (0.15, 0.2, 0.8, 0.3), "gain": 0.4},
    "pad": {"harmonics": tuple((h, 1.0 / h) for h in range(1, 7)), "adsr": (0.2, 0.3, 0.8, 0.4), "gain": 0.35},
    "lead": {"harmonics": tuple((h, 1.0 / h) for h in range(1, 10, 2)), "adsr": (0.01, 0.1, 0.7, 0.1), "gain": 0.45},
    "pluck": {"harmonics": ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)), "adsr": (0.003, 0.15, 0.2, 0.08), "gain": 0.5},
    "default": {"harmonics": ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)), "adsr": (0.01, 0.1, 0.7, 0.1), "gain": 0.5},
}
DRUM_PAN = {"closed_hat": 0.3, "open_hat": 0.3, "ride": 0.4, "crash": -0.3, "high_tom": 0.2, "low_tom": -0.2}


def _preset_for(track: Any) -> dict[str, Any]:
    label = getattr(track, "label", None)
    if label is None:
        return PRESETS["default"]
    fam = label.source_family
    roles = set(label.musical_role)
    if fam == "bass" or "bassline" in roles:
        return PRESETS["bass"]
    if fam in ("strings", "brass", "woodwinds", "vocal"):
        return PRESETS["strings"]
    if "pad" in roles or "drone" in roles:
        return PRESETS["pad"]
    if "pluck" in roles or "arpeggio" in roles:
        return PRESETS["pluck"]
    if "lead" in roles:
        return PRESETS["lead"]
    if fam in ("piano", "electric_piano", "guitar", "harp"):
        return PRESETS["piano"]
    return PRESETS["default"]


def render_note(midi: int, duration: float, sr: int, *, harmonics, adsr: tuple[float, float, float, float], amp: float, pitch_bend: list | None = None, vibrato: dict | None = None, articulation: str = "normal") -> np.ndarray:
    """Additive tone with a time-varying frequency (pitch-bend curve + vibrato) and
    articulation-dependent envelope. ``pitch_bend`` is [[t_rel, semitones], ...]."""
    a, d, s_level, r = adsr
    if articulation == "staccato":
        duration, r = max(0.05, duration * 0.8), min(r, 0.03)
    elif articulation == "legato":
        duration, r = duration + 0.02, min(r, 0.08)
    n = int((duration + r) * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr
    semis = np.zeros(n)
    if pitch_bend:
        pts = np.asarray(pitch_bend, dtype=np.float64)
        if pts.ndim == 2 and len(pts):
            semis += np.interp(t, pts[:, 0], pts[:, 1], left=pts[0, 1], right=pts[-1, 1])
    if vibrato and vibrato.get("rate_hz") and vibrato.get("depth_cents"):
        onset_delay = 0.15
        ramp = np.clip((t - onset_delay) / 0.2, 0.0, 1.0)
        semis += ramp * (float(vibrato["depth_cents"]) / 200.0) * np.sin(2 * np.pi * float(vibrato["rate_hz"]) * t)
    f = midi_to_hz(midi) * 2.0 ** (semis / 12.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros(n)
    f0 = midi_to_hz(midi)
    for h, w in harmonics:
        if h * f0 < sr / 2:
            y += w * np.sin(h * phase)
    y *= syn.adsr(n, sr, a, d, s_level, r)
    peak = float(np.max(np.abs(y))) if n else 0.0
    if peak > 0:
        y /= peak
    return (amp * y).astype(np.float32)


def _drum_sample(cls: str, sr: int, cache: dict[str, np.ndarray]) -> np.ndarray:
    if cls in cache:
        return cache[cls]
    if cls == "kick":
        y = syn.kick(sr)
    elif cls in ("snare", "rimshot"):
        y = syn.snare(sr)
    elif cls == "clap":
        base = syn.snare(sr, dur=0.12, amp=0.6)
        y = np.zeros(int(0.25 * sr), dtype=np.float32)
        for k, off in enumerate((0.0, 0.012, 0.024)):
            i = int(off * sr)
            y[i : i + len(base)] += base * (0.8 if k else 1.0)
    elif cls in ("closed_hat", "pedal_hat", "shaker", "tambourine"):
        y = syn.hat(sr)
    elif cls in ("open_hat", "ride", "crash"):
        y = syn.hat(sr, open_=True, amp=0.45)
    elif cls in ("low_tom", "mid_tom", "high_tom", "conga"):
        n = int(0.3 * sr)
        t = np.arange(n) / sr
        f = {"low_tom": 90, "mid_tom": 130, "high_tom": 180, "conga": 220}[cls]
        y = (np.sin(2 * np.pi * (f * (1 + 0.4 * np.exp(-t / 0.05))) * t) * np.exp(-t / 0.15) * 0.8).astype(np.float32)
    else:
        y = syn.hat(sr, amp=0.3)
    cache[cls] = y
    return y


@register("render", "additive")
class AdditiveRender(RenderAdapter):
    model = "additive_synth"
    model_version = "1.0"

    def render(self, *, tracks: list[Any], sample_rate: int, duration: float, fx_clips: list[tuple[FXEvent, np.ndarray]], tempo_bpm: float, midi_path: str | None = None) -> RenderResult:
        sr = sample_rate
        n = int((duration + 1.5) * sr)
        mix = np.zeros((n, 2), dtype=np.float32)
        outs: dict[str, np.ndarray] = {}
        sources: dict[str, str] = {}
        cache: dict[str, np.ndarray] = {}
        pans = np.linspace(-0.5, 0.5, max(1, len([t for t in tracks if not t.is_drum])))
        pi = 0
        for tr in tracks:
            buf = np.zeros((n, 2), dtype=np.float32)
            if tr.is_drum:
                for ev in tr.drums:
                    clip = _drum_sample(ev.drum_class, sr, cache)
                    syn.place(buf, clip, ev.time, sr, gain=ev.velocity / 127.0, pan=DRUM_PAN.get(ev.drum_class, 0.0))
            else:
                preset = _preset_for(tr)
                a, d, s, r = preset["adsr"]
                pan = float(pans[pi % len(pans)])
                pi += 1
                for ev in tr.notes:
                    dur = max(ev.offset - ev.onset, 0.05)
                    clip = render_note(ev.midi_note, dur, sr, harmonics=preset["harmonics"], adsr=(a, d, s, r), amp=preset["gain"], pitch_bend=ev.pitch_bend, vibrato=ev.features.get("vibrato"), articulation=str(ev.features.get("articulation", "normal")))
                    syn.place(buf, clip, ev.onset, sr, gain=(ev.velocity / 127.0) ** 1.5, pan=pan)
            outs[tr.name] = buf
            sources[tr.name] = "midi_rendered"
            mix += buf
        for ev, clip in fx_clips:
            i = int(ev.start * sr)
            if i >= n:
                continue
            seg = clip[: n - i]
            if seg.ndim == 1:
                seg = np.repeat(seg[:, None], 2, axis=1)
            mix[i : i + len(seg)] += seg[:, :2]
            sources[f"fx/{ev.fx_class}_{ev.event_id[-4:]}"] = "original_audio_clip"
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.891:
            mix *= 0.891 / peak
            for k in outs:
                outs[k] *= 0.891 / peak
        return RenderResult(mix=mix, sample_rate=sr, tracks=outs, sources=sources)
