"""Working out what a run will do, and describing what it did.

Pure bookkeeping — no PyTorch, no FFmpeg — so it's fast to import and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidOptionError
from .options import EnhanceOptions
from .video import VideoInfo

# -- planning ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunPlan:
    """What a run is going to do, worked out before any heavy lifting starts."""

    video: VideoInfo
    start_frame: int
    frames_to_process: int
    output_fps: float
    scale: float
    width: int
    height: int

    @property
    def last_frame(self) -> int:
        return self.start_frame + self.frames_to_process - 1

    @property
    def expected_output_frames(self) -> int:
        return round(self.frames_to_process * (self.output_fps / self.video.fps))

    @property
    def duplicates_frames(self) -> bool:
        return self.output_fps > self.video.fps


def plan_run(video: VideoInfo, options: EnhanceOptions) -> RunPlan:
    """Validate ``options`` against ``video`` and compute sizes, ranges and rates."""
    output_fps = options.output_fps if options.output_fps is not None else video.fps

    if output_fps <= 0:
        raise InvalidOptionError("--output-fps must be greater than 0.")

    if output_fps < video.fps:
        raise InvalidOptionError(
            f"--output-fps ({output_fps:g}) is lower than the input's {video.fps:.3f} fps. "
            "framelift can keep the frame rate or raise it (by repeating frames), "
            "but not lower it. Leave --output-fps out to keep the original rate."
        )

    if options.start_frame > video.frame_count:
        raise InvalidOptionError(
            f"--start-frame ({options.start_frame}) is past the end of the video, "
            f"which has {video.frame_count} frames."
        )

    scale = options.effective_scale
    if options.same_resolution:
        width, height = video.width, video.height
    else:
        width = int(round(video.width * scale))
        height = int(round(video.height * scale))

    frames_available = video.frame_count - options.start_frame + 1
    frames_to_process = frames_available
    if options.max_frames is not None:
        frames_to_process = min(frames_available, options.max_frames)

    return RunPlan(
        video=video,
        start_frame=options.start_frame,
        frames_to_process=frames_to_process,
        output_fps=output_fps,
        scale=scale,
        width=width,
        height=height,
    )


# -- results -----------------------------------------------------------------------


@dataclass(frozen=True)
class EnhanceResult:
    """What a run actually produced."""

    output_path: Path
    first_frame: int
    """First original frame included in the output (1-based)."""

    last_frame: int
    """Last original frame included in the output (1-based)."""

    frames_processed: int
    """Input frames that went through the AI in this run."""

    frames_written: int
    """Frames in the output file (more than processed when raising the fps)."""

    total_frames: int
    """Frame count of the whole input video."""

    interrupted: bool
    """True if the run was stopped early (e.g. Ctrl+C) and saved partially."""

    elapsed_seconds: float = 0.0

    audio: str = "disabled"
    """``"copied"``, ``"converted"`` (to AAC), ``"none"`` (the input has no audio),
    ``"failed"`` (the video was saved silent) or ``"disabled"``."""

    @property
    def is_complete(self) -> bool:
        """True if the output reaches the end of the input video."""
        return self.last_frame >= self.total_frames

    @property
    def resume_from(self) -> int | None:
        """The ``start_frame`` to use to continue where this run stopped."""
        return None if self.is_complete else self.last_frame + 1
