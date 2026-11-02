"""Synthetic audio generation for tests and demos (no copyrighted material needed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mrep.utils.dsp import butter_filter, istft, stft
from mrep.utils.music import midi_to_hz, midi_to_name


def adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    a, d, r = (max(1, int(x * sr)) for x in (attack, decay, release))
    s = max(0, n - a - d - r)
    env = np.concatenate(
        [
            np.linspace(0, 1, a, endpoint=False),
            np.linspace(1, sustain, d, endpoint=False),
            np.full(s, sustain),
            np.linspace(sustain, 0, r),
        ]
    )
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def sine(freq: float, dur: float, sr: int, amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def tone(
    midi: int,
    dur: float,
    sr: int,
    amp: float = 0.3,
    harmonics: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)),
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.05,
) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = midi_to_hz(midi)
    y = np.zeros(n)
    for h, w in harmonics:
        if h * f0 < sr / 2:
            y += w * np.sin(2 * np.pi * h * f0 * t)
    y *= adsr(n, sr, attack, decay, sustain, release)
    y /= max(np.max(np.abs(y)), 1e-9)
    return (amp * y).astype(np.float32)


def sub_bass(midi: int, dur: float, sr: int, amp: float = 0.5) -> np.ndarray:
    n = int(dur * sr)
    y = np.sin(2 * np.pi * midi_to_hz(midi) * np.arange(n) / sr)
    return (amp * y * adsr(n, sr, 0.005, 0.05, 0.9, 0.05)).astype(np.float32)


def chord(midis: list[int], dur: float, sr: int, amp: float = 0.25, **kw: Any) -> np.ndarray:
    y = sum(tone(m, dur, sr, amp=1.0, **kw) for m in midis)
    y /= max(np.max(np.abs(y)), 1e-9)
    return (amp * y).astype(np.float32)


def kick(sr: int, dur: float = 0.35, amp: float = 0.9) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = 45 + (160 - 45) * np.exp(-t / 0.05)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    y = np.sin(phase) * np.exp(-t / 0.12)
    return (amp * y).astype(np.float32)


def snare(sr: int, dur: float = 0.22, amp: float = 0.7, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = butter_filter(rng.standard_normal(n).astype(np.float32), sr, [300.0, 7000.0], "band")
    noise = noise / max(np.max(np.abs(noise)), 1e-9) * np.exp(-t / 0.06)
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.04)
    y = 0.7 * noise + 0.5 * body
    return (amp * y / max(np.max(np.abs(y)), 1e-9)).astype(np.float32)


def hat(sr: int, open_: bool = False, amp: float = 0.35, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dur = 0.35 if open_ else 0.06
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = butter_filter(rng.standard_normal(n).astype(np.float32), sr, 6500.0, "high")
    noise = noise / max(np.max(np.abs(noise)), 1e-9)
    y = noise * np.exp(-t / (0.12 if open_ else 0.018))
    return (amp * y).astype(np.float32)


def white_noise(sr: int, dur: float, amp: float = 0.3, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(dur * sr))).astype(np.float32)


def riser(sr: int, dur: float = 2.0, amp: float = 0.4, seed: int = 4, downlifter: bool = False) -> np.ndarray:
    """Noise with a low-pass cutoff sweeping upward (or downward) and rising amplitude."""
    n = int(dur * sr)
    noise = white_noise(sr, dur, 1.0, seed)
    n_fft, hop = 2048, 512
    S = stft(noise, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    T = S.shape[1]
    prog = np.linspace(0, 1, T)
    if downlifter:
        prog = prog[::-1]
    cutoff = 300.0 * (14000.0 / 300.0) ** prog
    mask = (1.0 / np.sqrt(1.0 + (freqs[:, None] / cutoff[None, :]) ** 8)).astype(np.float32)
    y = istft(S * mask, n_fft, hop, n)
    env = np.linspace(0.2, 1.0, n) if not downlifter else np.linspace(1.0, 0.2, n)
    y = y / max(np.max(np.abs(y)), 1e-9) * env
    return (amp * y).astype(np.float32)


def place(buffer: np.ndarray, clip: np.ndarray, start_sec: float, sr: int, gain: float = 1.0, pan: float = 0.0) -> None:
    """Add a mono clip into a stereo buffer at ``start_sec`` with constant-power panning."""
    start = int(round(start_sec * sr))
    if start >= len(buffer):
        return
    end = min(len(buffer), start + len(clip))
    seg = (clip[: end - start] * gain).astype(np.float32)
    fade = min(len(seg), int(0.005 * sr))
    if fade > 1:
        seg = seg.copy()
        seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    theta = (pan + 1.0) * np.pi / 4.0
    buffer[start:end, 0] += seg * np.cos(theta)
    buffer[start:end, 1] += seg * np.sin(theta)


def demo_song(sr: int = 44100, bpm: float = 120.0, bars: int = 8, seed: int = 0, lead_in: float = 0.25, return_stems: bool = False):
    """Render a small electronic loop with drums, sub bass, chords, arpeggio and a riser.

    Returns the stereo audio (n, 2) and a ground-truth dictionary (seconds). ``lead_in``
    seconds of silence precede the first downbeat (like a real recording). With
    ``return_stems=True`` a third value holds the reference stems (drums / bass / other / fx)
    that sum exactly to the mix, for separation metrics.
    """
    beat = 60.0 / bpm
    bar = 4 * beat
    total = lead_in + bars * bar + 1.0
    n = int(total * sr)
    stems = {name: np.zeros((n, 2), dtype=np.float32) for name in ("drums", "bass", "other", "fx")}
    mix = stems["drums"]  # drum hits are placed here first; other sources below
    truth: dict[str, Any] = {
        "bpm": bpm,
        "time_signature": "4/4",
        "key": "C major",
        "bars": bars,
        "notes": [],
        "drums": [],
        "chords": [],
        "fx": [],
    }
    progression = [
        (48, "C", "major", [60, 64, 67]),
        (45, "A", "minor", [57, 60, 64]),
        (41, "F", "major", [53, 57, 60]),
        (43, "G", "major", [55, 59, 62]),
    ]
    k, s, ch, oh = kick(sr), snare(sr), hat(sr), hat(sr, open_=True)
    truth["lead_in"] = lead_in
    for b in range(bars):
        t0 = lead_in + b * bar
        root_midi, root, quality, triad = progression[b % 4]
        # drums ------------------------------------------------------------------
        for i in range(4):
            t = t0 + i * beat
            if i in (0, 2):
                place(stems["drums"], k, t, sr)
                truth["drums"].append({"time": t, "class": "kick"})
            else:
                place(stems["drums"], s, t, sr, gain=0.8)
                truth["drums"].append({"time": t, "class": "snare"})
            for half in (0.0, 0.5):
                th = t + half * beat
                if i == 3 and half == 0.5:
                    place(stems["drums"], oh, th, sr, gain=0.5)
                    truth["drums"].append({"time": th, "class": "open_hat"})
                else:
                    place(stems["drums"], ch, th, sr, gain=0.45)
                    truth["drums"].append({"time": th, "class": "closed_hat"})
        # bass -------------------------------------------------------------------
        bass_midi = root_midi - 12
        for i in (0, 2):
            t = t0 + i * beat
            place(stems["bass"], sub_bass(bass_midi, 1.6 * beat, sr, amp=0.55), t, sr)
            truth["notes"].append(
                {"midi_note": bass_midi, "pitch": midi_to_name(bass_midi), "onset": t, "offset": t + 1.6 * beat, "track": "bass"}
            )
        # chords (pad) -----------------------------------------------------------
        pad = chord(triad, bar * 0.98, sr, amp=0.22, attack=0.12, decay=0.2, sustain=0.8, release=0.2)
        place(stems["other"], pad, t0, sr, pan=-0.3)
        for m in triad:
            truth["notes"].append({"midi_note": m, "pitch": midi_to_name(m), "onset": t0, "offset": t0 + bar * 0.98, "track": "chords"})
        truth["chords"].append({"start": t0, "end": t0 + bar, "root": root, "quality": quality})
        # arpeggio 1-5-3-5 in 8ths -------------------------------------------------
        pattern = [triad[0] + 12, triad[2] + 12, triad[1] + 12, triad[2] + 12] * 2
        for i, m in enumerate(pattern):
            t = t0 + i * beat / 2
            place(stems["other"], tone(m, beat * 0.45, sr, amp=0.25, attack=0.005, decay=0.08, sustain=0.5, release=0.03), t, sr, pan=0.4)
            truth["notes"].append({"midi_note": m, "pitch": midi_to_name(m), "onset": t, "offset": t + beat * 0.45, "track": "arp"})
    # riser into the last bar -----------------------------------------------------
    if bars >= 2:
        r_dur = min(2.0, bar)
        r_start = lead_in + (bars - 1) * bar - r_dur
        place(stems["fx"], riser(sr, r_dur, amp=0.35), r_start, sr)
        truth["fx"].append({"class": "riser", "start": r_start, "end": r_start + r_dur})
    mix = sum(stems.values())
    peak = float(np.max(np.abs(mix))) or 1.0
    gain = 0.9 / peak
    mix = (mix * gain).astype(np.float32)
    for name in stems:
        stems[name] = (stems[name] * gain).astype(np.float32)
    truth["duration"] = total
    truth["beats"] = [round(lead_in + i * beat, 6) for i in range(bars * 4)]
    truth["downbeats"] = [round(lead_in + b * bar, 6) for b in range(bars)]
    if return_stems:
        return mix, truth, stems
    return mix, truth
