"""framelift — upscale and restore videos with Real-ESRGAN, with a human touch.

Quick start::

    from framelift import enhance_video

    result = enhance_video("old_clip.mp4", "old_clip_hd.mp4", scale=2, profile="old_tv")
    print(result.frames_written, "frames written to", result.output_path)

Importing framelift is cheap: PyTorch and Real-ESRGAN are only loaded when
you touch something that needs them (``enhance_video``, ``VideoEnhancer``…).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

__version__ = "1.2.0"

from .analysis import AIEffect, estimate_noise, softness, suggest_ai_strength, suggest_size
from .console import setup_console
from .errors import (
    BlackFrameError,
    EncodingError,
    FrameliftError,
    InvalidOptionError,
    MissingDependencyError,
    NothingToDoError,
    OutOfMemoryError,
    VideoReadError,
)
from .filters import PROFILES, Profile, apply_profile
from .models import MODELS, ModelSpec
from .options import EnhanceOptions
from .planning import EnhanceResult, RunPlan, plan_run
from .video import VideoInfo, probe_video

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import VideoEnhancer, enhance_video
    from .tuning import Preset, TuneReport, tune_video

# Names that live in modules importing PyTorch; resolved on first access.
_LAZY = {
    "enhance_video": "pipeline",
    "VideoEnhancer": "pipeline",
    "tune_video": "tuning",
    "TuneReport": "tuning",
    "Preset": "tuning",
}

__all__ = [
    "__version__",
    # main API
    "enhance_video",
    "VideoEnhancer",
    "EnhanceOptions",
    "EnhanceResult",
    "RunPlan",
    "plan_run",
    "tune_video",
    "TuneReport",
    "Preset",
    "AIEffect",
    "estimate_noise",
    "softness",
    "suggest_ai_strength",
    "suggest_size",
    # catalogs
    "MODELS",
    "ModelSpec",
    "PROFILES",
    "Profile",
    "apply_profile",
    # utilities
    "probe_video",
    "VideoInfo",
    "setup_console",
    # errors
    "FrameliftError",
    "MissingDependencyError",
    "InvalidOptionError",
    "VideoReadError",
    "EncodingError",
    "OutOfMemoryError",
    "BlackFrameError",
    "NothingToDoError",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'framelift' has no attribute '{name}'")
