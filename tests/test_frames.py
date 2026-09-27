import doctest

import numpy as np
import pytest

from framelift import frames
from framelift.frames import blend_with_classic_upscale, mean_brightness, repeat_count, resize_exact


def test_doctests():
    assert doctest.testmod(frames).failed == 0


@pytest.mark.parametrize(
    ("input_fps", "output_fps", "frames_in"),
    [(24, 24, 100), (24, 60, 100), (23.976, 60, 1001), (29.97, 30, 500), (25, 50, 37)],
)
def test_repeat_count_never_drifts(input_fps, output_fps, frames_in):
    total = sum(repeat_count(i, input_fps, output_fps) for i in range(1, frames_in + 1))
    assert total == round(frames_in * output_fps / input_fps)


def test_same_fps_writes_each_frame_once():
    assert {repeat_count(i, 30, 30) for i in range(1, 50)} == {1}


@pytest.mark.parametrize("bad", [0, -1])
def test_repeat_count_rejects_non_positive_fps(bad):
    with pytest.raises(ValueError):
        repeat_count(1, bad, 30)
    with pytest.raises(ValueError):
        repeat_count(1, 30, bad)


def test_resize_exact_is_a_no_op_at_same_size(frame):
    assert resize_exact(frame, 64, 48) is frame
    assert resize_exact(frame, 96, 72).shape == (72, 96, 3)


def test_blend_full_strength_returns_the_ai_frame(frame):
    ai = np.zeros((96, 128, 3), np.uint8)
    assert blend_with_classic_upscale(frame, ai, 128, 96, ai_strength=1.0) is ai


def test_blend_zero_strength_is_a_plain_resize(frame):
    ai = np.zeros((96, 128, 3), np.uint8)
    out = blend_with_classic_upscale(frame, ai, 128, 96, ai_strength=0.0)
    np.testing.assert_array_equal(out, resize_exact(frame, 128, 96))


def test_blend_clamps_strength_and_resizes_ai_output(frame):
    ai = np.zeros((192, 256, 3), np.uint8)  # e.g. native 4x output
    out = blend_with_classic_upscale(frame, ai, 96, 72, ai_strength=5)
    assert out.shape == (72, 96, 3)
    assert mean_brightness(out) == 0


def test_mean_brightness():
    assert mean_brightness(np.full((4, 4, 3), 200, np.uint8)) == pytest.approx(200)
