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


# -- softness -----------------------------------------------------------------------

# (upper bound, label). Calibrated on clips from crisp synthetic video (~0.15) to
# visibly soft old footage (~0.45–0.55).
SOFTNESS_LEVELS: tuple[tuple[float, str], ...] = (
    (0.25, "sharp"),
    (0.40, "slightly soft"),
    (0.55, "soft"),
    (float("inf"), "very soft"),
)


def softness(frame: Frame) -> float:
    """How blurry a frame is, from 0 (sharp) to 1 (very blurry).

    Crété-Roffet's no-reference blur metric: blur the frame a bit more and see how
    much contrast between neighbouring pixels is lost. A sharp image loses a lot;
    an already-blurry one barely changes. Fairly independent of content.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    blur_per_axis = []
    for axis, kernel in ((0, (1, 9)), (1, (9, 1))):
        reblurred = cv2.blur(gray, kernel)
        original_contrast = np.abs(np.diff(gray, axis=axis))
        remaining_contrast = np.abs(np.diff(reblurred, axis=axis))
        lost = np.maximum(0, original_contrast - remaining_contrast).sum()
        total = max(original_contrast.sum(), 1e-9)
        blur_per_axis.append((total - lost) / total)
    return float(max(blur_per_axis))


def describe_softness(value: float) -> str:
    for upper_bound, label in SOFTNESS_LEVELS:
        if value < upper_bound:
            return label
    raise AssertionError("unreachable: the last level has no upper bound")


def motion(frame: Frame, next_frame: Frame) -> float:
    """Average brightness change between two frames (0–255). High = fast motion or a cut."""
    small = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)) for f in (frame, next_frame)
    ]
    return float(np.abs(small[0].astype(np.float64) - small[1]).mean())


def structural_similarity(a: Frame, b: Frame) -> float:
    """SSIM between two same-sized frames: 1.0 = identical structure."""
    x, y = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in (a, b))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def smooth(image):
        return cv2.GaussianBlur(image, (0, 0), 1.5)

    mean_x, mean_y = smooth(x), smooth(y)
    var_x = smooth(x * x) - mean_x**2
    var_y = smooth(y * y) - mean_y**2
    covariance = smooth(x * y) - mean_x * mean_y
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim_map.mean())


# -- what the AI does to this footage -----------------------------------------------


@dataclass(frozen=True)
class AIEffect:
    """How a model changes *this* footage, compared with a plain resize.

    * ``sharpening``: softness removed (plain resize minus AI). ~0 = adds nothing.
    * ``noise_change``: noise after the AI minus before, measured at the source size.
      Negative = cleans noise; positive = invents grain or texture.
    * ``structure``: SSIM between the AI result (shrunk back) and the source.
      Below ~0.93 the AI is redrawing details rather than restoring them.
    """

    sharpening: float
    noise_change: float
    structure: float

    @classmethod
    def measure(cls, source: Frame, classic: Frame, enhanced: Frame) -> AIEffect:
        """``classic`` and ``enhanced`` are the same crop upscaled without / with AI."""
        height, width = source.shape[:2]
        shrunk = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_AREA)
        return cls(
            sharpening=softness(classic) - softness(enhanced),
            noise_change=estimate_noise(shrunk) - estimate_noise(source),
            structure=structural_similarity(shrunk, source),
        )

    @classmethod
    def average(cls, effects: Sequence[AIEffect]) -> AIEffect:
        return cls(
            sharpening=float(np.mean([e.sharpening for e in effects])),
            noise_change=float(np.mean([e.noise_change for e in effects])),
            structure=float(np.mean([e.structure for e in effects])),
        )


def suggest_ai_strength(
    effect: AIEffect, source_softness: float | None = None
) -> tuple[float, list[str]]:
    """Pick an AI strength from what the AI actually does, with the reasons why.

    The more real sharpness the AI brings back, the more of it we keep. Invented
    texture, redrawn structure or an already-sharp source pull it back down.
    Rules calibrated on a handful of clips (crisp, blurred and grainy synthetic
    video, plus real old footage).
    """
    strength, reasons = 0.45, []

    if source_softness is not None and source_softness < SOFTNESS_LEVELS[0][0]:
        strength -= 0.10
        reasons.append("source is already sharp")

    if effect.sharpening >= 0.25:
        strength += 0.25
        reasons.append("recovers a lot of sharpness")
    elif effect.sharpening >= 0.12:
        strength += 0.15
        reasons.append("recovers sharpness well")
    elif effect.sharpening >= 0.06:
        strength += 0.05
        reasons.append("recovers some sharpness")
    elif effect.sharpening < 0.05:
        strength -= 0.15
        reasons.append("adds little sharpness (source is already crisp)")

    if effect.noise_change <= -0.25:
        strength += 0.05
        reasons.append("cleans up noise")
    elif effect.noise_change >= 0.30:
        strength -= 0.10
        reasons.append("invents grain or texture")

    if effect.structure < 0.90:
        strength -= 0.15
        reasons.append("redraws details noticeably")
    elif effect.structure < 0.93:
        strength -= 0.05
        reasons.append("redraws some details")

    strength = min(0.85, max(0.20, strength))
    return round(strength * 20) / 20, reasons


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
    """How much real detail a frame has (variance of the Laplacian).

    A light blur first keeps pixel noise from passing for detail.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rank_test_frames(
    samples: Sequence[tuple[int, Frame, Frame | None]],
) -> list[tuple[int, Frame]]:
    """Order ``(number, frame, next_frame)`` samples from best to worst test material.

    Frames during fast motion or right at a scene cut are blurred or mixed, so
    anything moving much more than is typical for this video goes last. The rest
    are ordered sharpest first.
    """
    moves = [motion(frame, nxt) if nxt is not None else 0.0 for _, frame, nxt in samples]
    limit = max(2 * float(np.median(moves)), 3.0)

    def badness(index: int) -> tuple[bool, float]:
        return moves[index] > limit, softness(samples[index][1])

    order = sorted(range(len(samples)), key=badness)
    return [(samples[i][0], samples[i][1]) for i in order]


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
_ROW_LABEL_WIDTH = 200
_TITLE_HEIGHT = 44


def build_comparison_sheet(
    rows: Sequence[str | Sequence[str]],
    columns: Sequence[str],
    render: Callable[[int, int], Frame],
    title: str = "",
) -> Frame:
    """Assemble a labelled grid: ``render(row, col)`` returns each tile.

    A row label is either a profile name or ``(caption, main, detail)`` lines.
    All tiles must have the same size. Labels use plain ASCII (OpenCV's fonts
    have no accents).
    """
    rows = [("profile", row) if isinstance(row, str) else tuple(row) for row in rows]
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
        caption, main, *detail = row_name
        cv2.putText(sheet, caption, (_GAP * 2, middle - 16), _FONT, 0.5, _MUTED, 1, cv2.LINE_AA)
        cv2.putText(sheet, main, (_GAP * 2, middle + 10), _FONT, 0.6, _TEXT, 1, cv2.LINE_AA)
        for i, line in enumerate(detail):
            y = middle + 32 + i * 20
            cv2.putText(sheet, line, (_GAP * 2, y), _FONT, 0.45, _MUTED, 1, cv2.LINE_AA)

        for c, column_name in enumerate(columns):
            left = _ROW_LABEL_WIDTH + _GAP + c * (tile_width + _GAP)
            cv2.putText(sheet, column_name, (left, top + 21), _FONT, 0.55, _TEXT, 1, cv2.LINE_AA)
            y = top + _LABEL_HEIGHT
            sheet[y : y + tile_height, left : left + tile_width] = tiles[r][c]

    return sheet
