import cv2
import numpy as np
import pytest

from framelift.analysis import (
    NOISE_LEVELS,
    build_comparison_sheet,
    describe_noise,
    estimate_noise,
    most_detailed_region,
    sample_positions,
    suggest_size,
)


@pytest.fixture
def scene() -> np.ndarray:
    """Smooth gradient + shapes + text: detail that must NOT be mistaken for noise."""
    image = np.zeros((180, 320, 3), np.uint8)
    for y in range(180):
        image[y, :] = (40 + y // 2, 80 + y // 3, 150 - y // 3)
    cv2.circle(image, (90, 90), 50, (30, 160, 220), -1)
    for i in range(6):
        cv2.putText(image, "detail", (190, 40 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    return image


def _add_noise(image, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def test_texture_is_not_mistaken_for_noise(scene):
    assert estimate_noise(scene) < 1.0


def test_noise_estimate_grows_with_real_noise(scene):
    estimates = [estimate_noise(_add_noise(scene, sigma)) for sigma in (0, 4, 10, 20)]
    assert estimates == sorted(estimates)
    assert estimates[-1] > 10


@pytest.mark.parametrize(
    ("noise", "profile"),
    [(0.2, "none"), (1.5, "minimal"), (3, "soft_camera"), (6, "old_tv"), (30, "heavy_noise")],
)
def test_describe_noise(noise, profile):
    assert describe_noise(noise)[1] == profile


def test_noise_levels_are_ordered_and_cover_everything():
    bounds = [upper for upper, _, _ in NOISE_LEVELS]
    assert bounds == sorted(bounds)
    assert bounds[-1] == float("inf")


@pytest.mark.parametrize(
    ("height", "same", "scale"),
    [(2160, True, 1.0), (1080, True, 1.0), (720, False, 1.5), (480, False, 2.25),
     (360, False, 3.0), (240, False, 3.0), (144, False, 3.0)],
)  # fmt: skip
def test_suggest_size(height, same, scale):
    suggestion = suggest_size(height * 16 // 9, height)
    assert (suggestion.same_resolution, suggestion.scale) == (same, scale)


def test_sample_positions_avoid_intro_and_ending():
    positions = sample_positions(1000, 3)
    assert positions == sorted(positions)
    assert len(positions) == 3
    assert positions[0] >= 50 and positions[-1] <= 950


@pytest.mark.parametrize(("count", "samples"), [(1, 3), (2, 5), (10, 1)])
def test_sample_positions_small_videos(count, samples):
    positions = sample_positions(count, samples)
    assert positions and all(1 <= p <= count for p in positions)


def test_sample_positions_without_frames():
    assert sample_positions(0, 3) == []


def test_most_detailed_region_finds_the_texture():
    image = np.full((200, 200, 3), 128, np.uint8)
    image[150:200, 150:200] = _add_noise(np.zeros((50, 50, 3)), 60)
    region = most_detailed_region(image, 50, 50)
    assert (region.x, region.y) == (150, 150)
    assert region.crop(image).shape == (50, 50, 3)


def test_region_never_exceeds_the_frame():
    image = np.zeros((40, 60, 3), np.uint8)
    region = most_detailed_region(image, 500, 500)
    assert (region.width, region.height) == (60, 40)


def test_comparison_sheet_layout():
    tile = np.full((30, 40, 3), 200, np.uint8)
    sheet = build_comparison_sheet(["a", "b"], ["x", "y", "z"], lambda r, c: tile, "title")
    assert sheet.ndim == 3
    assert sheet.shape[1] > 3 * 40 and sheet.shape[0] > 2 * 30
    assert (sheet == 200).sum() >= 6 * tile.size  # all six tiles were pasted
