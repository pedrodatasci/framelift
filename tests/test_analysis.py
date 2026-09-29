import cv2
import numpy as np
import pytest

from framelift.analysis import (
    NOISE_LEVELS,
    AIEffect,
    build_comparison_sheet,
    describe_noise,
    describe_softness,
    estimate_noise,
    most_detailed_region,
    motion,
    rank_test_frames,
    sample_positions,
    softness,
    structural_similarity,
    suggest_ai_strength,
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


# -- softness, motion, structure ---------------------------------------------------


def test_softness_separates_sharp_from_blurry(scene):
    sharp = softness(scene)
    blurry = softness(cv2.GaussianBlur(scene, (0, 0), 2.5))
    assert sharp < 0.25 < blurry
    assert describe_softness(sharp) == "sharp"
    assert describe_softness(blurry) in {"soft", "very soft", "slightly soft"}


def test_motion_is_zero_for_identical_frames_and_high_for_a_cut(scene):
    assert motion(scene, scene) == 0
    assert motion(scene, 255 - scene) > 20


def test_structural_similarity(scene):
    assert structural_similarity(scene, scene) == pytest.approx(1.0)
    assert structural_similarity(scene, _add_noise(scene, 30)) < 0.8


def test_ai_effect_measures_sharpening_and_noise(scene):
    source = cv2.GaussianBlur(scene, (0, 0), 1.5)
    big = (scene.shape[1] * 2, scene.shape[0] * 2)
    classic = cv2.resize(source, big, interpolation=cv2.INTER_LANCZOS4)
    crisp = cv2.resize(scene, big, interpolation=cv2.INTER_LANCZOS4)  # a "perfect" AI

    effect = AIEffect.measure(source, classic, crisp)
    assert effect.sharpening > 0.1
    assert effect.structure > 0.9

    noisy = AIEffect.measure(source, classic, _add_noise(classic, 20))
    assert noisy.noise_change > 1


def test_ai_effect_average():
    average = AIEffect.average([AIEffect(0.1, -1, 0.9), AIEffect(0.3, 1, 1.0)])
    assert average == AIEffect(pytest.approx(0.2), pytest.approx(0), pytest.approx(0.95))


# (sharpening, noise_change, structure, source softness) -> expected strength.
# These are the measurements from the calibration clips.
@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        ((0.045, -0.05, 0.983, 0.15), 0.2),  # crisp source: the AI adds little
        ((0.294, 0.02, 0.976, 0.54), 0.7),  # blurry source: the AI brings back a lot
        ((0.381, 0.15, 0.907, 0.54), 0.65),  # ...but the heavier model redraws a bit
        ((0.100, -0.27, 0.952, 0.47), 0.55),  # old concert footage, video model
        ((0.169, 0.38, 0.962, 0.47), 0.5),  # same footage, x4plus invents texture
    ],
)
def test_suggest_ai_strength(measured, expected):
    *effect, source_softness = measured
    strength, reasons = suggest_ai_strength(AIEffect(*effect), source_softness)
    assert strength == pytest.approx(expected)
    assert reasons


def test_suggest_ai_strength_stays_in_range():
    lowest, _ = suggest_ai_strength(AIEffect(0.0, 5.0, 0.5), 0.1)
    highest, _ = suggest_ai_strength(AIEffect(0.9, -5.0, 1.0), 0.9)
    assert 0.2 <= lowest < highest <= 0.85


def test_rank_test_frames_prefers_sharp_and_steady(scene):
    blurry = cv2.GaussianBlur(scene, (0, 0), 3)
    samples = [
        (1, blurry, blurry),  # steady but blurry
        (2, scene, 255 - scene),  # sharp but at a scene cut
        (3, scene, scene),  # sharp and steady: the one we want
    ]
    ranked = [number for number, _ in rank_test_frames(samples)]
    assert ranked == [3, 1, 2]


def test_sheet_accepts_multi_line_row_labels():
    tile = np.full((30, 40, 3), 200, np.uint8)
    rows = [("fast preset", "none", "animevideov3"), "old_tv"]
    sheet = build_comparison_sheet(rows, ["a", "b"], lambda r, c: tile)
    assert sheet.shape[0] > 2 * 30
