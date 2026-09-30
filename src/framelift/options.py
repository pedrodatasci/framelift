"""All the knobs of an enhancement run, in one place.

:class:`EnhanceOptions` mirrors the command-line flags one-to-one (``--ai-strength``
becomes ``ai_strength`` and so on) and uses exactly the same defaults, so
anything you can do from the terminal you can do from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from .errors import InvalidOptionError
from .filters import PROFILES
from .models import MODELS

DEVICES = ("auto", "cpu", "cuda")
ENCODERS = ("cpu", "nvenc")
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")

DEFAULT_MODEL = "realesr-animevideov3"


@dataclass
class EnhanceOptions:
    """Settings for :func:`framelift.enhance_video`.

    The defaults are a balanced starting point: a fast model, a gentle 1.5x
    upscale and an AI blend of 45% so the result still looks like the
    original footage rather than a painting.
    """

    # --- What the output should look like ---------------------------------
    model: str = DEFAULT_MODEL
    """Real-ESRGAN model to use. See :data:`framelift.MODELS`."""

    scale: float = 1.5
    """Final size relative to the input (1.5 = 150%). Ignored with ``same_resolution``."""

    same_resolution: bool = False
    """Keep the exact input resolution — clean the image up without resizing it."""

    output_fps: float | None = None
    """Output frame rate. ``None`` keeps the input's. Higher values duplicate frames."""

    ai_strength: float = 0.45
    """How much of the AI result to keep, from 0.0 (none) to 1.0 (all of it)."""

    profile: str = "none"
    """Pre-cleaning filter applied before the AI. See :data:`framelift.PROFILES`."""

    # --- Which part of the video to process -------------------------------
    start_frame: int = 1
    """1-based frame to start from — handy for resuming an interrupted run."""

    max_frames: int | None = None
    """Process at most this many frames (great for quick previews)."""

    # --- Hardware ----------------------------------------------------------
    device: str = "auto"
    """``auto`` (CUDA if available), ``cpu`` or ``cuda``."""

    gpu_id: int = 0
    """Which CUDA GPU to use when there is more than one."""

    half: bool = False
    """Use FP16 on CUDA — faster and lighter on VRAM. Ignored on CPU."""

    torch_threads: int = 0
    """PyTorch CPU threads. ``0`` lets PyTorch decide."""

    tile: int = 256
    """Split frames into tiles of this size to save memory. ``0`` disables tiling."""

    tile_pad: int = 10
    """Overlap between tiles, in pixels, to hide seams."""

    pre_pad: int = 0
    """Padding added around the whole frame before inference."""

    prefetch: int = 2
    """How many pre-processed frames the reader thread keeps ready."""

    weights_dir: str | Path = "weights"
    """Where model weights are stored (and downloaded to on first use)."""

    # --- Encoding ------------------------------------------------------------
    audio: bool = True
    """Copy the input's audio into the output (trimmed to the enhanced part)."""

    encoder: str = "cpu"
    """``cpu`` (libx264) or ``nvenc`` (NVIDIA hardware, falls back to cpu)."""

    crf: int = 20
    """Output quality: lower = better and bigger. 18–22 is the sweet spot."""

    x264_preset: str = "veryfast"
    """libx264 speed/size trade-off (only used by the ``cpu`` encoder)."""

    # --- Diagnostics -----------------------------------------------------------
    debug_timings: bool = False
    """Print a per-stage timing breakdown while processing."""

    timing_every: int = 20
    """With ``debug_timings``: report every N frames."""

    gc_every: int = 0
    """With ``debug_timings``: force garbage collection every N frames (0 = never)."""

    # ---------------------------------------------------------------------------

    @property
    def effective_scale(self) -> float:
        """The scale actually used, taking ``same_resolution`` into account."""
        return 1.0 if self.same_resolution else self.scale

    @property
    def clamped_ai_strength(self) -> float:
        """``ai_strength`` forced into the valid 0.0–1.0 range."""
        return max(0.0, min(1.0, float(self.ai_strength)))

    def validate(self) -> None:
        """Raise :class:`InvalidOptionError` if any option can't possibly work.

        Checks that need the actual video (like "start frame is past the end")
        happen later, once the video has been opened.
        """
        _require_choice("model", self.model, MODELS)
        _require_choice("profile", self.profile, PROFILES)
        _require_choice("device", self.device, DEVICES)
        _require_choice("encoder", self.encoder, ENCODERS)
        _require_choice("x264_preset", self.x264_preset, X264_PRESETS)

        if self.start_frame < 1:
            raise InvalidOptionError(
                "The start frame must be 1 or greater (frames are counted from 1)."
            )
        if self.max_frames is not None and self.max_frames < 1:
            raise InvalidOptionError(
                "max frames must be at least 1 (leave it out to process everything)."
            )
        if not self.same_resolution and self.scale <= 0:
            raise InvalidOptionError(f"scale must be greater than 0, got {self.scale}.")
        if self.output_fps is not None and self.output_fps <= 0:
            raise InvalidOptionError("The output fps must be greater than 0.")
        if self.tile < 0 or self.tile_pad < 0 or self.pre_pad < 0:
            raise InvalidOptionError("tile, tile_pad and pre_pad can't be negative.")
        if self.gpu_id < 0:
            raise InvalidOptionError("gpu_id can't be negative.")

    @classmethod
    def field_names(cls) -> list[str]:
        """Names of every option (used by the CLI to build options from args)."""
        return [f.name for f in fields(cls)]


def _require_choice(name: str, value: str, allowed) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise InvalidOptionError(f"Unknown {name} '{value}'. Pick one of: {choices}.")
