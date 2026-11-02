"""Project: the output folder layout for one song."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

STAGE_DIRS = {
    "normalized": "01_normalized",
    "stems": "02_stems",
    "events": "03_events",
    "midi": "04_midi",
    "score": "05_score",
    "analysis": "06_analysis",
    "fx": "07_fx",
    "reconstruction": "08_reconstruction",
    "logs": "logs",
}


def slugify(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return name or "song"


@dataclass
class Project:
    root: Path
    song_name: str
    project_id: str = ""

    @classmethod
    def for_audio(cls, audio_path: Path, output_root: Path) -> Project:
        song = slugify(Path(audio_path).stem)
        return cls(root=Path(output_root) / song, song_name=song)

    @classmethod
    def open(cls, project_dir: Path) -> Project:
        project_dir = Path(project_dir)
        return cls(root=project_dir, song_name=project_dir.name)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def dir(self, key: str) -> Path:
        return self.root / STAGE_DIRS[key]

    @property
    def normalized_dir(self) -> Path:
        return self.dir("normalized")

    @property
    def stems_dir(self) -> Path:
        return self.dir("stems")

    @property
    def events_dir(self) -> Path:
        return self.dir("events")

    @property
    def midi_dir(self) -> Path:
        return self.dir("midi")

    @property
    def score_dir(self) -> Path:
        return self.dir("score")

    @property
    def analysis_dir(self) -> Path:
        return self.dir("analysis")

    @property
    def fx_dir(self) -> Path:
        return self.dir("fx")

    @property
    def reconstruction_dir(self) -> Path:
        return self.dir("reconstruction")

    @property
    def logs_dir(self) -> Path:
        return self.dir("logs")

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for d in STAGE_DIRS.values():
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        """Path relative to the project root (as stored in manifests/events)."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def abs(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else self.root / p
