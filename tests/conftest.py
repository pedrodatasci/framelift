"""Shared fixtures. Nothing here needs PyTorch, so the suite runs anywhere."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def frame() -> np.ndarray:
    """A deterministic, textured 64x48 BGR frame."""
    rng = np.random.default_rng(seed=7)
    image = rng.integers(40, 216, size=(48, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (40, 30), (255, 255, 255), -1)
    return image


@pytest.fixture
def make_video(tmp_path: Path):
    """Factory writing a small video where frame *n* has brightness ``n * 10`` (capped at 250)."""

    def _make(frames: int = 12, fps: float = 24.0, size=(64, 48)) -> Path:
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for n in range(1, frames + 1):
            writer.write(np.full((size[1], size[0], 3), min(n * 10, 250), dtype=np.uint8))
        writer.release()
        return path

    return _make


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
