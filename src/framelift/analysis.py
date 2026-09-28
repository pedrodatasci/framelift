"""Look at a video and suggest settings: noise level, target size, and a visual
comparison sheet to pick the look by eye.

Everything here is plain OpenCV/NumPy (no PyTorch), so it's quick and testable.
The suggestions are heuristics — good starting points, not ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .filters import Frame

# -- noise ------------------------------------------------------------------------

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_EDGE_PERCENTILE = 90

# (upper bound of measured noise, label, profile). Calibrated on real H.264 output:
# compression already wipes out faint grain, so these describe what *survives* in
# the file — which is exactly the noise the AI would otherwise amplify.
NOISE_LEVELS: tuple[tuple[float, str, str], ...] = (
    (1.0, "very clean", "none"),
    (2.5, "clean", "minimal"),
    (5.0, "light grain", "soft_camera"),
    (9.0, "noticeable noise", "old_tv"),
    (float("inf"), "heavy noise", "heavy_noise"),
)


def estimate_noise(frame: Frame) -> float:
    """Estimate the noise level of a frame (roughly a standard deviation, 0–255 scale).

    Uses Immerkær's fast estimator: a 3x3 filter that cancels smooth image content
    and keeps pixel-level noise. Strong edges and texture would fool it, so the
    10% of pixels with the strongest gradients are ignored.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    response = np.abs(cv2.filter2D(gray, -1, _NOISE_KERNEL))[1:-1, 1:-1]

    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))[
        1:-1, 1:-1
    ]
    flat_areas = gradient <= np.percentile(gradient, _EDGE_PERCENTILE)

    return float(np.sqrt(np.pi / 2) * response[flat_areas].mean() / 6)


def describe_noise(noise: float) -> tuple[str, str]:
    """Map a noise estimate to ``(human label, suggested profile name)``."""
    for upper_bound, label, profile in NOISE_LEVELS:
        if noise < upper_bound:
            return label, profile
    raise AssertionError("unreachable: the last level has no upper bound")


# -- size --------------------------------------------------------------------------

MAX_SUGGESTED_SCALE = 3.0
TARGET_HEIGHTS = (1080, 720)


@dataclass(frozen=True)
class SizeSuggestion:
    same_resolution: bool
    scale: float
    target_height: int
    reason: str


def suggest_size(width: int, height: int) -> SizeSuggestion:
    """Suggest a scale that lands on a common resolution without over-stretching.

    Full HD when it's within 3x of the source, otherwise 720p, otherwise 3x.
    Sources that are already Full HD or bigger are restored at their own size.
    """
    if height >= 1080:
        return SizeSuggestion(
            True, 1.0, height, "already Full HD or larger: restore, don't enlarge"
        )

    for target in TARGET_HEIGHTS:
        if target / height <= MAX_SUGGESTED_SCALE:
            scale = round(target / height, 2)
            return SizeSuggestion(False, scale, target, f"{height}p → {target}p")

    target = int(round(height * MAX_SUGGESTED_SCALE))
    return SizeSuggestion(
        False, MAX_SUGGESTED_SCALE, target, f"{height}p → {target}p (3x is the useful maximum)"
    )


# -- choosing what to look at ------------------------------------------------------


def sample_positions(frame_count: int, samples: int) -> list[int]:
    """Evenly spaced 1-based frame numbers, skipping the first and last 5%.

    Intros and endings are often black or titles, which say little about the footage.
    """
    if frame_count < 1:
        return []

    samples = max(1, min(samples, frame_count))
    first = max(1, int(frame_count * 0.05))
    last = max(first, int(frame_count * 0.95))

    if samples == 1:
        return [(first + last) // 2]

    step = (last - first) / (samples - 1)
    return sorted({int(round(first + i * step)) for i in range(samples)})


def detail_score(frame: Frame) -> float:
    """How much fine detail a frame has (variance of the Laplacian)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def crop(self, frame: Frame) -> Frame:
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


def most_detailed_region(frame: Frame, width: int, height: int, grid: int = 4) -> Region:
    """Find the ``width`` x ``height`` window with the most detail.

    Differences between settings show up in detail (edges, faces, text), not in
    flat sky, so that's where the comparison sheet zooms in.
    """
    frame_height, frame_width = frame.shape[:2]
    width, height = min(width, frame_width), min(height, frame_height)

    best, best_score = Region(0, 0, width, height), -1.0
    for row in range(grid):
        for col in range(grid):
            x = round((frame_width - width) * col / max(1, grid - 1))
            y = round((frame_height - height) * row / max(1, grid - 1))
            candidate = Region(x, y, width, height)
            score = detail_score(candidate.crop(frame))
            if score > best_score:
                best, best_score = candidate, score

    return best


# -- the comparison sheet ------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BACKGROUND = (24, 24, 24)
_TEXT = (235, 235, 235)
_MUTED = (150, 150, 150)
_GAP = 8
_LABEL_HEIGHT = 30
_ROW_LABEL_WIDTH = 150
_TITLE_HEIGHT = 44


def build_comparison_sheet(
    rows: Sequence[str],
    columns: Sequence[str],
    render: Callable[[int, int], Frame],
    title: str = "",
) -> Frame:
    """Assemble a labelled grid: ``render(row, col)`` returns each tile.

    All tiles must have the same size. Labels use plain ASCII (OpenCV's fonts
    have no accents).
    """
    tiles = [[render(r, c) for c in range(len(columns))] for r in range(len(rows))]
    tile_height, tile_width = tiles[0][0].shape[:2]

    sheet_width = _ROW_LABEL_WIDTH + len(columns) * (tile_width + _GAP) + _GAP
    sheet_height = _TITLE_HEIGHT + len(rows) * (tile_height + _LABEL_HEIGHT + _GAP) + _GAP
    sheet = np.full((sheet_height, sheet_width, 3), _BACKGROUND, dtype=np.uint8)

    if title:
        cv2.putText(sheet, title, (_GAP * 2, 29), _FONT, 0.7, _TEXT, 1, cv2.LINE_AA)

    for r, row_name in enumerate(rows):
        top = _TITLE_HEIGHT + r * (tile_height + _LABEL_HEIGHT + _GAP)
        middle = top + _LABEL_HEIGHT + tile_height // 2
        cv2.putText(sheet, "profile", (_GAP * 2, middle - 12), _FONT, 0.5, _MUTED, 1, cv2.LINE_AA)
        cv2.putText(sheet, row_name, (_GAP * 2, middle + 14), _FONT, 0.6, _TEXT, 1, cv2.LINE_AA)

        for c, column_name in enumerate(columns):
            left = _ROW_LABEL_WIDTH + _GAP + c * (tile_width + _GAP)
            cv2.putText(sheet, column_name, (left, top + 21), _FONT, 0.55, _TEXT, 1, cv2.LINE_AA)
            y = top + _LABEL_HEIGHT
            sheet[y : y + tile_height, left : left + tile_width] = tiles[r][c]

    return sheet
