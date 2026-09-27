"""Classic (non-AI) image filters used to prepare frames before the AI sees them.

Real-ESRGAN is great at inventing detail, but it also happily "enhances"
noise and compression artifacts. A light clean-up pass first usually gives
a more natural result. Each :class:`Profile` is a small, declarative recipe:
optional denoise → tone adjustment → gentle sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

Frame = np.ndarray
"""A BGR ``uint8`` image of shape ``(height, width, 3)``, as OpenCV uses."""


def adjust_tone(
    frame: Frame,
    contrast: float = 1.0,
    brightness: float = 0,
    saturation: float = 1.0,
) -> Frame:
    """Tweak contrast, brightness and saturation of a BGR frame."""
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return adjusted


def unsharp_mask(frame: Frame, amount: float = 0.25, radius: float = 3) -> Frame:
    """Sharpen by subtracting a blurred copy (the classic darkroom trick)."""
    if amount <= 0:
        return frame

    blurred = cv2.GaussianBlur(frame, (0, 0), radius)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


class Denoiser(Protocol):
    def __call__(self, frame: Frame) -> Frame: ...


@dataclass(frozen=True)
class BilateralDenoise:
    """Edge-preserving smoothing. Cheap, good for light grain."""

    diameter: int
    sigma: float

    def __call__(self, frame: Frame) -> Frame:
        return cv2.bilateralFilter(
            frame, d=self.diameter, sigmaColor=self.sigma, sigmaSpace=self.sigma
        )


@dataclass(frozen=True)
class NonLocalMeansDenoise:
    """Strong denoiser that compares patches across the image. Slow but thorough."""

    strength: float = 4
    color_strength: float = 4
    template_window: int = 7
    search_window: int = 21

    def __call__(self, frame: Frame) -> Frame:
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=self.strength,
            hColor=self.color_strength,
            templateWindowSize=self.template_window,
            searchWindowSize=self.search_window,
        )


@dataclass(frozen=True)
class Profile:
    """A pre-cleaning recipe: denoise, then adjust tone, then sharpen."""

    name: str
    description: str
    denoise: Denoiser | None = None
    contrast: float = 1.0
    brightness: float = 0
    saturation: float = 1.0
    sharpen_amount: float = 0.0
    sharpen_radius: float = 2

    @property
    def is_passthrough(self) -> bool:
        """True when this profile leaves frames untouched."""
        return (
            self.denoise is None
            and self.contrast == 1.0
            and self.brightness == 0
            and self.saturation == 1.0
            and self.sharpen_amount <= 0
        )

    def apply(self, frame: Frame) -> Frame:
        if self.is_passthrough:
            return frame

        cleaned = self.denoise(frame) if self.denoise else frame
        cleaned = adjust_tone(
            cleaned,
            contrast=self.contrast,
            brightness=self.brightness,
            saturation=self.saturation,
        )
        return unsharp_mask(cleaned, amount=self.sharpen_amount, radius=self.sharpen_radius)


PROFILES: dict[str, Profile] = {
    profile.name: profile
    for profile in (
        Profile(
            name="none",
            description="No pre-processing. Fastest; the AI sees the raw frame.",
        ),
        Profile(
            name="minimal",
            description="A touch of contrast, color and sharpness. Almost free.",
            contrast=1.02,
            brightness=1,
            saturation=1.02,
            sharpen_amount=0.10,
        ),
        Profile(
            name="soft_camera",
            description="Light smoothing for phone/webcam footage with fine grain.",
            denoise=BilateralDenoise(diameter=3, sigma=18),
            contrast=1.04,
            brightness=2,
            saturation=1.05,
            sharpen_amount=0.18,
        ),
        Profile(
            name="old_tv",
            description="Stronger smoothing and a color lift for VHS/TV captures.",
            denoise=BilateralDenoise(diameter=5, sigma=25),
            contrast=1.06,
            brightness=3,
            saturation=1.08,
            sharpen_amount=0.22,
        ),
        Profile(
            name="heavy_noise",
            description="Non-local-means denoising for very noisy video. Slowest.",
            denoise=NonLocalMeansDenoise(),
            contrast=1.05,
            brightness=2,
            saturation=1.04,
            sharpen_amount=0.15,
        ),
    )
}


def get_profile(name: str) -> Profile:
    """Look up a profile by name, with a helpful error for typos."""
    try:
        return PROFILES[name]
    except KeyError:
        choices = ", ".join(PROFILES)
        raise ValueError(f"Unknown profile '{name}'. Available: {choices}.") from None


def apply_profile(frame: Frame, name: str) -> Frame:
    """Convenience wrapper: ``apply_profile(frame, "old_tv")``."""
    return get_profile(name).apply(frame)
