"""Model adapters. All model-specific code lives here; the rest of MREP only sees interfaces.

Built-in baseline adapters are pure DSP (numpy/scipy) so the pipeline runs on any CPU.
ML adapters register themselves with ``@register(kind, name)`` and are selected via config.
"""

from mrep.adapters.base import (  # noqa: F401
    AdapterError,
    AdapterInfo,
    AdapterUnavailable,
    BaseAdapter,
    ClassificationAdapter,
    ClassificationResult,
    DiscoveryAdapter,
    DiscoveryResult,
    DrumTranscriptionAdapter,
    FXAdapter,
    MIRAdapter,
    MIRResult,
    RenderAdapter,
    RenderResult,
    SeparationAdapter,
    SeparationResult,
    SoundAnalysisAdapter,
    SoundAnalysisResult,
    TranscriptionAdapter,
)
from mrep.adapters.registry import (  # noqa: F401
    create_adapter,
    get_adapter_class,
    list_adapters,
    register,
)
