"""Numpy/scipy signal-processing primitives shared by the baseline (model-free) adapters.

Everything here is deterministic DSP. ML adapters live in ``mrep.adapters`` and may use
their own feature extraction.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, uniform_filter1d

EPS = 1e-10


# ----------------------------------------------------------------------------- basic shapes
def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    return x.mean(axis=1).astype(np.float32)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[:, None] if x.ndim == 1 else x


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def db(x: float, floor: float = 1e-9) -> float:
    return 20.0 * math.log10(max(float(x), floor))


# ----------------------------------------------------------------------------- STFT
def _hann(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)


def frame_signal(x: np.ndarray, frame: int, hop: int, center: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if center:
        pad = frame // 2
        mode = "reflect" if len(x) > pad else "constant"
        x = np.pad(x, (pad, pad), mode=mode)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def stft(x: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Return a complex spectrogram of shape (n_fft//2 + 1, n_frames)."""
    frames = frame_signal(x, n_fft, hop, center=True) * _hann(n_fft)
    return np.fft.rfft(frames, axis=1).T


def istft(X: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    frames = np.fft.irfft(X.T, n=n_fft, axis=1)
    win = _hann(n_fft)
    n_frames = frames.shape[0]
    out_len = n_fft + hop * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    win_sq = win.astype(np.float64) ** 2
    for t in range(n_frames):
        s = t * hop
        y[s : s + n_fft] += frames[t] * win
        wsum[s : s + n_fft] += win_sq
    y /= np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    y = y[pad : pad + length]
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


def fft_freqs(sr: int, n_fft: int) -> np.ndarray:
    return np.fft.rfftfreq(n_fft, 1.0 / sr)


def frame_times(n_frames: int, sr: int, hop: int) -> np.ndarray:
    return np.arange(n_frames) * hop / sr


# ----------------------------------------------------------------------------- onsets
def log_band_matrix(freqs: np.ndarray, n_bands: int = 64, fmin: float = 30.0, fmax: float | None = None) -> np.ndarray:
    """(n_bands, n_bins) rectangular band aggregation on a log-frequency axis."""
    fmax = fmax or float(freqs[-1])
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    M = np.zeros((n_bands, len(freqs)))
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if m.any():
            M[i, m] = 1.0 / m.sum()
    return M


def onset_strength(S_mag: np.ndarray, smooth: int = 1, freqs: np.ndarray | None = None, n_bands: int = 0) -> np.ndarray:
    """Half-wave rectified spectral flux on a dB spectrogram, normalised to max 1.

    With ``freqs`` and ``n_bands`` the spectrum is first pooled into log-spaced bands so
    that a wide-band hi-hat does not out-vote a kick drum simply by covering more bins.
    """
    if freqs is not None and n_bands > 0:
        S_mag = log_band_matrix(freqs, n_bands) @ S_mag
    ref = max(float(S_mag.max()), EPS)
    L = 20.0 * np.log10(np.maximum(S_mag, 1e-6 * ref))
    diff = np.maximum(np.diff(L, axis=1), 0.0)
    env = diff.mean(axis=0)
    env = np.concatenate([[0.0], env])
    if smooth > 1:
        env = uniform_filter1d(env, smooth)
    m = env.max()
    return (env / m).astype(np.float32) if m > 0 else env.astype(np.float32)


def pick_peaks(
    env: np.ndarray,
    pre_max: int = 3,
    post_max: int = 3,
    pre_avg: int = 10,
    post_avg: int = 10,
    delta: float = 0.07,
    wait: int = 3,
) -> np.ndarray:
    """librosa-style peak picker returning frame indices."""
    env = np.asarray(env, dtype=np.float64)
    if env.size == 0:
        return np.zeros(0, dtype=int)
    local_max = maximum_filter1d(env, size=pre_max + post_max + 1, mode="nearest")
    local_avg = uniform_filter1d(env, size=pre_avg + post_avg + 1, mode="nearest")
    cand = np.where((env >= local_max - 1e-12) & (env >= local_avg + delta) & (env > 0))[0]
    peaks: list[int] = []
    last = -wait - 1
    for i in cand:
        if i - last > wait:
            peaks.append(int(i))
            last = int(i)
    return np.asarray(peaks, dtype=int)


def autocorrelate(x: np.ndarray) -> np.ndarray:
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, nfft)
    return np.fft.irfft(F * np.conj(F), nfft)[:n]


def estimate_tempo(
    onset_env: np.ndarray,
    fps: float,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
    prior_bpm: float = 120.0,
) -> tuple[float, float]:
    """Return (bpm, confidence) from autocorrelation of the onset envelope."""
    env = np.asarray(onset_env, dtype=np.float64)
    if env.size < 4:
        return float(prior_bpm), 0.0
    env = env - env.mean()
    acf = autocorrelate(env)
    if acf[0] <= 0:
        return float(prior_bpm), 0.0
    acf = acf / acf[0]
    lags = np.arange(len(acf), dtype=np.float64)
    bpms = np.full_like(lags, np.inf)
    bpms[1:] = 60.0 * fps / lags[1:]
    mask = (bpms >= min_bpm) & (bpms <= max_bpm)
    if not mask.any():
        return float(prior_bpm), 0.0
    prior = np.zeros_like(lags)
    prior[mask] = np.exp(-0.5 * (np.log2(bpms[mask] / prior_bpm)) ** 2)
    score = np.where(mask, acf * prior, -np.inf)
    best = int(np.argmax(score))
    shift = 0.0
    if 1 <= best < len(acf) - 1:
        a, b, c = acf[best - 1], acf[best], acf[best + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            shift = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    lag = best + shift
    bpm = 60.0 * fps / lag
    confidence = float(np.clip(acf[best], 0.0, 1.0))
    return float(bpm), confidence


def track_beats(onset_env: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis 2007). Returns beat frame indices."""
    env = np.asarray(onset_env, dtype=np.float64)
    period = 60.0 * fps / max(bpm, 1e-3)
    if env.size < 2 or period < 1.0:
        return np.zeros(0, dtype=int)
    env = gaussian_filter1d(env, sigma=max(period / 32.0, 0.5))
    env = env / (env.std() + EPS)
    lo, hi = -int(round(2 * period)), -int(round(period / 2))
    window = np.arange(lo, hi + 1)
    if window.size == 0:
        window = np.array([-int(round(period))])
    txwt = -tightness * (np.log(-window / period)) ** 2
    T = env.size
    cumscore = np.zeros(T)
    backlink = -np.ones(T, dtype=int)
    for i in range(T):
        cand = i + window
        valid = cand >= 0
        if not valid.any():
            cumscore[i] = env[i]
            continue
        sc = txwt[valid] + cumscore[cand[valid]]
        k = int(np.argmax(sc))
        cumscore[i] = env[i] + sc[k]
        backlink[i] = cand[valid][k]
    tail_len = min(T, int(round(period)))
    tail = cumscore[T - tail_len :]
    b = T - tail_len + int(np.argmax(tail))
    beats: list[int] = []
    while b >= 0:
        beats.append(b)
        b = backlink[b]
    return np.asarray(beats[::-1], dtype=int)


def choose_downbeat_phase(beat_frames: np.ndarray, weight_env: np.ndarray, beats_per_bar: int) -> int:
    best, best_score = 0, -np.inf
    for off in range(beats_per_bar):
        idx = beat_frames[off::beats_per_bar]
        idx = idx[idx < len(weight_env)]
        if idx.size == 0:
            continue
        score = float(weight_env[idx].mean())
        if score > best_score:
            best, best_score = off, score
    return best


# ----------------------------------------------------------------------------- chroma / key
def chroma_from_stft(
    S_mag: np.ndarray, freqs: np.ndarray, fmin: float = 65.0, fmax: float = 5000.0, a4: float = 440.0
) -> np.ndarray:
    """(12, n_frames) chroma, L2-normalised per frame."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    midi = 69.0 + 12.0 * np.log2(freqs[mask] / a4)
    pcs = np.round(midi).astype(int) % 12
    C = np.zeros((12, S_mag.shape[1]), dtype=np.float64)
    np.add.at(C, pcs, S_mag[mask])
    norm = np.linalg.norm(C, axis=0) + EPS
    return (C / norm).astype(np.float32)


def estimate_key(chroma_mean: np.ndarray) -> tuple[int, str, float, dict[str, float]]:
    """Krumhansl-Schmuckler key finding. Returns (tonic_pc, mode, confidence, all_scores)."""
    from mrep.utils.music import KS_MAJOR, KS_MINOR, key_name

    c = np.asarray(chroma_mean, dtype=np.float64)
    if c.std() < EPS:
        return 0, "major", 0.0, {}
    scores: dict[tuple[int, str], float] = {}
    for mode, profile in (("major", np.array(KS_MAJOR)), ("minor", np.array(KS_MINOR))):
        for tonic in range(12):
            prof = np.roll(profile, tonic)
            r = float(np.corrcoef(prof, c)[0, 1])
            scores[(tonic, mode)] = r if np.isfinite(r) else -1.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (tonic, mode), best = ranked[0]
    second = ranked[1][1]
    margin = max(best - second, 0.0)
    confidence = 1.0 - math.exp(-8.0 * margin)
    if best < 0.2:
        confidence *= max(best, 0.0) / 0.2
    named = {key_name(t, m): round(v, 4) for (t, m), v in ranked}
    return tonic, mode, float(np.clip(confidence, 0.0, 1.0)), named


# ----------------------------------------------------------------------------- HPSS
def hpss(S: np.ndarray, kernel_harm: int = 31, kernel_perc: int = 31, power: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Median-filter harmonic/percussive separation on a complex spectrogram."""
    mag = np.abs(S)
    H = median_filter(mag, size=(1, kernel_harm), mode="reflect")
    P = median_filter(mag, size=(kernel_perc, 1), mode="reflect")
    Hp = H**power
    Pp = P**power
    tot = Hp + Pp + EPS
    return S * (Hp / tot), S * (Pp / tot)


# ----------------------------------------------------------------------------- spectral features
def spectral_centroid(S_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    tot = S_mag.sum(axis=0) + EPS
    return (freqs[:, None] * S_mag).sum(axis=0) / tot


def spectral_flatness(S_mag: np.ndarray, floor_ratio: float = 1e-5) -> np.ndarray:
    """Wiener entropy per frame; magnitudes are floored relative to the global peak so that
    digitally silent bins do not collapse the geometric mean."""
    S = S_mag.astype(np.float64)
    floor = max(float(S.max()) * floor_ratio, 1e-12)
    p = np.maximum(S, floor) ** 2
    return np.exp(np.mean(np.log(p), axis=0)) / np.mean(p, axis=0)


def spectral_rolloff(S_mag: np.ndarray, freqs: np.ndarray, roll: float = 0.85) -> np.ndarray:
    p = S_mag.astype(np.float64) ** 2
    cum = np.cumsum(p, axis=0)
    thresh = roll * cum[-1:, :]
    idx = np.argmax(cum >= thresh, axis=0)
    return freqs[idx]


def band_energy(S_mag: np.ndarray, freqs: np.ndarray, bands: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Energy fraction per band over the whole spectrogram."""
    p = S_mag.astype(np.float64) ** 2
    total = p.sum() + EPS
    return {name: float(p[(freqs >= lo) & (freqs < hi)].sum() / total) for name, (lo, hi) in bands.items()}


def band_energy_frames(S_mag: np.ndarray, freqs: np.ndarray, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    p = S_mag.astype(np.float64) ** 2
    return {name: p[(freqs >= lo) & (freqs < hi)].sum(axis=0) for name, (lo, hi) in bands.items()}


def rms_frames(x: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    frames = frame_signal(x, frame, hop, center=True)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS).astype(np.float32)


# ----------------------------------------------------------------------------- filters / resampling
def butter_filter(x: np.ndarray, sr: int, cutoff, btype: str = "low", order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    if isinstance(cutoff, (list, tuple)):
        cutoff = [min(max(c, 1.0), nyq * 0.999) for c in cutoff]
    else:
        cutoff = min(max(float(cutoff), 1.0), nyq * 0.999)
    sos = signal.butter(order, cutoff, btype=btype, fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(int(sr), int(target_sr))
    return signal.resample_poly(x, target_sr // g, sr // g, axis=0).astype(np.float32)


# ----------------------------------------------------------------------------- structure
def novelty_curve(features: np.ndarray, kernel: int = 8) -> np.ndarray:
    """Checkerboard-kernel novelty on the self-similarity matrix of (T, D) features."""
    F = np.asarray(features, dtype=np.float64)
    T = F.shape[0]
    if T < 2:
        return np.zeros(T)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + EPS)
    ssm = Fn @ Fn.T
    L = max(1, min(kernel, T // 2))
    ramp = np.arange(-L, L)
    taper = np.exp(-0.5 * (ramp / (L / 2.0)) ** 2)
    sign = np.sign(ramp + 0.5)
    K = np.outer(sign * taper, sign * taper)
    nov = np.zeros(T)
    for i in range(L, T - L):
        nov[i] = np.sum(ssm[i - L : i + L, i - L : i + L] * K)
    nov = np.maximum(nov, 0.0)
    m = nov.max()
    return nov / m if m > 0 else nov


def segment_boundaries(novelty: np.ndarray, min_distance: int, rel_threshold: float = 0.3) -> np.ndarray:
    if novelty.size == 0:
        return np.zeros(0, dtype=int)
    delta = rel_threshold * float(novelty.max())
    return pick_peaks(novelty, pre_max=min_distance // 2, post_max=min_distance // 2, pre_avg=min_distance, post_avg=min_distance, delta=delta * 0.5, wait=min_distance)
