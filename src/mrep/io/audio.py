"""Audio reading/writing (libsndfile via soundfile, ffmpeg fallback for other codecs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mrep.utils.dsp import resample as _resample

FIRST_CLASS_FORMATS = {".wav", ".flac", ".mp3"}
EXTENDED_FORMATS = {".m4a", ".aac", ".aiff", ".aif", ".ogg", ".opus", ".wma"}
SUPPORTED_FORMATS = FIRST_CLASS_FORMATS | EXTENDED_FORMATS


class AudioFormatError(ValueError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def audio_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise AudioFormatError(f"unsupported audio format {ext!r} ({path.name})")
    try:
        info = sf.info(str(path))
        return {
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
            "decoder": "libsndfile",
        }
    except (RuntimeError, sf.LibsndfileError):
        if shutil.which("ffprobe") is None:
            raise AudioFormatError(f"cannot decode {path.name}: libsndfile failed and ffprobe is not installed") from None
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a:0", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        stream = json.loads(out)["streams"][0]
        sr = int(stream.get("sample_rate", 0))
        duration = float(stream.get("duration", 0.0))
        return {
            "format": stream.get("codec_name", ext.lstrip(".")).upper(),
            "subtype": stream.get("sample_fmt", ""),
            "sample_rate": sr,
            "channels": int(stream.get("channels", 0)),
            "frames": int(duration * sr),
            "duration": duration,
            "decoder": "ffprobe",
        }


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read audio as float32 (n_samples, n_channels)."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except (RuntimeError, sf.LibsndfileError):
        if not ffmpeg_available():
            raise AudioFormatError(f"cannot decode {path.name}: libsndfile failed and ffmpeg is not installed") from None
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "decoded.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)],
                check=True,
                capture_output=True,
            )
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            return data, int(sr)


def write_audio(path: Path, data: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    sf.write(str(path), data, int(sr), subtype=subtype)
    return path


def to_stereo(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] == 2:
        return data
    return data[:, :2]


def to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    return data if data.ndim == 1 else data.mean(axis=1).astype(np.float32)


def resample(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    return _resample(data, sr, target_sr)


def peak_normalize(data: np.ndarray, target_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    """Scale so the peak sits at ``target_dbfs``. Returns (audio, applied_gain_db)."""
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0:
        return data, 0.0
    target = 10 ** (target_dbfs / 20.0)
    gain = target / peak
    return (data * gain).astype(np.float32), 20.0 * float(np.log10(gain))


def encode_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> bool:
    if not ffmpeg_available():
        return False
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
        check=True,
        capture_output=True,
    )
    return True
