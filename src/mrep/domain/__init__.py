"""Domain layer: pure-Python data model (labels, events, tracks, project).

Rule: this package must never import PyTorch / TensorFlow / JAX or any model library.
"""

from mrep.domain.events import (  # noqa: F401
    BeatEvent,
    ChordEvent,
    DrumEvent,
    Event,
    FXEvent,
    KeyEvent,
    NoteEvent,
    ScoreNoteEvent,
    SectionEvent,
    TempoEvent,
    event_from_dict,
)
from mrep.domain.labels import SoundLabel  # noqa: F401
from mrep.domain.project import Project  # noqa: F401
from mrep.domain.provenance import Provenance  # noqa: F401
from mrep.domain.tracks import Stem, Track  # noqa: F401
