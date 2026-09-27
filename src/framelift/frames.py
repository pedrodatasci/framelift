"""Small, pure helpers that operate on single frames or frame counts.

Nothing here touches PyTorch or FFmpeg, which keeps it easy to test.
"""

from __future__ import annotations

import cv2

from .filters import Frame


def resize_exact(frame: Frame, width: int, height: int) -> Frame:
    """Resize with Lanczos (sharp, high quality) — or do nothing if already that size."""
    current_height, current_width = frame.shape[:2]

    if current_width == width and current_height == height:
        return frame

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def blend_with_classic_upscale(
    source: Frame,
    ai_frame: Frame,
    width: int,
    height: int,
    ai_strength: float,
) -> Frame:
    """Mix the AI result with a plain Lanczos upscale of ``source``.

    Full-strength Real-ESRGAN can look plasticky on real footage. Blending it
    with a conventional resize keeps the AI's crisp edges while bringing back
    some of the original texture. ``ai_strength`` is the AI's share:
    ``1.0`` is pure AI, ``0.0`` is a plain resize.
    """
    ai_strength = max(0.0, min(1.0, float(ai_strength)))
    ai_frame = resize_exact(ai_frame, width, height)

    if ai_strength >= 0.999:
        return ai_frame

    classic = resize_exact(source, width, height)
    return cv2.addWeighted(ai_frame, ai_strength, classic, 1.0 - ai_strength, 0)


def mean_brightness(frame: Frame) -> float:
    """Average luminance of a BGR frame, from 0 (black) to 255 (white)."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def repeat_count(position: int, input_fps: float, output_fps: float) -> int:
    """How many times the ``position``-th input frame (1-based) must be written.

    To turn 24 fps into 60 fps without changing the duration, each input frame
    is written 2 or 3 times in a repeating pattern. Computing it from running
    totals (instead of a fixed ratio per frame) spreads the duplicates evenly
    and guarantees the total never drifts.

    >>> [repeat_count(i, 24, 60) for i in range(1, 5)]
    [2, 3, 3, 2]
    """
    if input_fps <= 0:
        raise ValueError("input_fps must be > 0")
    if output_fps <= 0:
        raise ValueError("output_fps must be > 0")

    ratio = output_fps / input_fps
    written_before = round((position - 1) * ratio)
    written_after = round(position * ratio)

    return max(0, written_after - written_before)
