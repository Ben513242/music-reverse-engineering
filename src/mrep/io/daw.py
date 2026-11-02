"""DAW project export: a DAW-agnostic ``session.json`` plus a minimal Reaper ``.rpp`` project.

The Reaper file references the stem WAVs as audio items and carries tempo / time signature
and section markers. MIDI tracks are not embedded (Reaper stores MIDI inline); import the
files in ``04_midi/`` by drag-and-drop instead. Everything is plain text, no DAW needed here.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def _q(text: str) -> str:
    return '"' + str(text).replace('"', "'") + '"'


def write_session_json(path: Path, session: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_reaper_rpp(
    path: Path,
    *,
    bpm: float,
    time_signature: tuple[int, int],
    duration: float,
    audio_tracks: list[dict[str, Any]],
    markers: list[tuple[float, str]],
    tempo_changes: list[tuple[float, float]] | None = None,
    title: str = "MREP",
) -> Path:
    """``audio_tracks``: [{"name": str, "file": relative path, "volume": float, "mute": bool}]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f'<REAPER_PROJECT 0.1 "6.0/mrep" {int(time.time())}')
    lines.append(f"  TITLE {_q(title)}")
    lines.append(f"  TEMPO {bpm:.3f} {time_signature[0]} {time_signature[1]}")
    for i, (t, name) in enumerate(markers, start=1):
        lines.append(f"  MARKER {i} {t:.6f} {_q(name)} 0")
    if tempo_changes:
        lines.append("  <TEMPOENVEX")
        for t, b in tempo_changes:
            lines.append(f"    PT {t:.6f} {b:.3f} 0")
        lines.append("  >")
    for tr in audio_tracks:
        guid = "{" + str(uuid.uuid4()).upper() + "}"
        lines.append(f"  <TRACK {guid}")
        lines.append(f"    NAME {_q(tr['name'])}")
        lines.append(f"    VOLPAN {float(tr.get('volume', 1.0)):.4f} 0 -1 -1 1")
        lines.append(f"    MUTESOLO {1 if tr.get('mute') else 0} 0 0")
        lines.append("    <ITEM")
        lines.append("      POSITION 0")
        lines.append(f"      LENGTH {duration:.6f}")
        lines.append(f"      NAME {_q(Path(tr['file']).name)}")
        lines.append("      <SOURCE WAVE")
        lines.append(f"        FILE {_q(tr['file'])}")
        lines.append("      >")
        lines.append("    >")
        lines.append("  >")
    lines.append(">")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
