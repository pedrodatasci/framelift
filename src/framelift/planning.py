"""Working out what a run will do, and describing what it did.

Pure bookkeeping — no PyTorch, no FFmpeg — so it's fast to import and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
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
    if options.interpolate and options.output_fps is None:
        output_fps = video.fps * 2  # --interpolate alone: twice the frame rate
    elif options.interpolate:
        # A rate typed with a few decimals (59.94 for 29.97 x 2) means the exact multiple;
        # keeping it exact keeps the in-between moments from drifting off the frames.
        exact = simple_multiple(video.fps, output_fps)
        if exact is not None and abs(exact - output_fps) < 0.01:
            output_fps = exact

    if output_fps <= 0:
        raise InvalidOptionError("--output-fps must be greater than 0.")

    if output_fps < video.fps:
        raise InvalidOptionError(
            f"--output-fps ({output_fps:g}) is lower than the input's {video.fps:.3f} fps. "
            "framelift can keep the frame rate or raise it (by repeating frames), "
            "but not lower it. Leave --output-fps out to keep the original rate."
        )

    if options.interpolate and output_fps <= video.fps:
        raise InvalidOptionError(
            f"--interpolate needs an --output-fps above the input's {video.fps:.3f} fps: "
            "it draws the frames in between. Leave --output-fps out to double the rate."
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


# -- frame rates for interpolation -------------------------------------------------

SNAP = 0.05
"""In-between moments this close to an original frame (as a fraction of the time between
two frames) show that frame itself: the difference is invisible, and it saves a RIFE run."""


def simple_multiple(input_fps: float, output_fps: float) -> float | None:
    """The simple multiple of ``input_fps`` (2x, 2.5x, 3x, 4x…) within 0.5% of
    ``output_fps``, or ``None`` if there isn't one.

    >>> round(simple_multiple(29.99, 60), 2)
    59.98
    """
    ratio = Fraction(output_fps / input_fps).limit_denominator(4)
    if ratio <= 1:
        return None
    exact = input_fps * float(ratio)
    return exact if abs(exact - output_fps) <= 0.005 * exact else None


def in_between_count(input_fps: float, output_fps: float, frames: int) -> int:
    """How many in-between frames RIFE draws for ``frames`` input frames (fewer if some
    pairs are scene cuts or identical). Follows :class:`framelift.interpolation.Retimer`."""
    ratio = output_fps / input_fps
    count = slot = 0
    for index in range(frames):
        while slot <= index * ratio + 1e-9:
            t = slot / ratio - (index - 1)
            count += index > 0 and SNAP < t < 1 - SNAP
            slot += 1
    return count


def interpolation_advice(input_fps: float, output_fps: float, frames: int) -> str | None:
    """A tip when a slightly different ``output_fps`` means much less work for RIFE.

    Between rates like 29.99 and 60 (almost, but not exactly, 2x) the in-between moments
    slowly drift off the original frames, until nearly every output frame has to be
    drawn. The exact multiple (59.98) looks the same and needs far fewer.
    """
    exact = simple_multiple(input_fps, output_fps)
    if exact is None or abs(exact - output_fps) < 0.01:
        return None
    now, then = (
        in_between_count(input_fps, output_fps, frames),
        in_between_count(input_fps, exact, frames),
    )
    if then > now * 0.9:
        return None
    multiple = Fraction(exact / input_fps).limit_denominator(4)
    how = f"--output-fps {exact:.3f}"
    if multiple == 2:
        how = f"leave --output-fps out (it doubles the rate) or use {how}"
    return (
        f"--output-fps {output_fps:g} isn't an exact multiple of the input's {input_fps:.3f} fps, "
        f"so the in-between moments drift and RIFE will draw {now:,} frames. At {exact:.3f} fps "
        f"({float(multiple):g}x, the same to the eye) it would draw {then:,}, "
        f"{100 * (now - then) / now:.0f}% less: {how}."
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

    frames_interpolated: int = 0
    """In-between frames drawn by RIFE (``interpolate``); the rest were written as is."""

    frames_reused: int = 0
    """Input frames identical to the one before them, whose result was reused instead of
    running the AI again."""

    @property
    def is_complete(self) -> bool:
        """True if the output reaches the end of the input video."""
        return self.last_frame >= self.total_frames

    @property
    def resume_from(self) -> int | None:
        """The ``start_frame`` to use to continue where this run stopped."""
        return None if self.is_complete else self.last_frame + 1
