"""RIFE in-between frames: where they go (fast, with a stand-in network) and what the real
network draws (slow: downloads its weights once, 25 MB)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from framelift.frames import repeat_count  # noqa: E402
from framelift.interpolation import (  # noqa: E402
    SCENE_CUT_SIMILARITY,
    FrameInterpolator,
    Retimer,
    _thumbnail,
    ensure_rife_weights,
    similarity,
)


class FakeInterpolator:
    """Stands in for RIFE: in-between frames are tagged with the frames and the moment."""

    def __init__(self):
        self.calls = []

    def prepare(self, image):
        return image

    def between(self, before, after, t):
        self.calls.append((int(before[0, 0, 0]), int(after[0, 0, 0]), round(t, 4)))
        return ("between", int(before[0, 0, 0]), int(after[0, 0, 0]), round(t, 4))


def frame(tag, noise=None):
    """A frame tagged in its first pixel; ``noise`` gives it unrelated content (a cut)."""
    image = np.full((48, 64, 3), 128, np.uint8)
    if noise is not None:
        image = np.random.default_rng(noise).integers(0, 256, image.shape, dtype=np.uint8)
    image[0, 0, 0] = tag % 256
    return image


def run(input_fps, output_fps, count, frames=None):
    fake = FakeInterpolator()
    retimer = Retimer(input_fps, output_fps, fake)
    written = []
    for n in range(count):
        image = frames[n] if frames else frame(n)
        written += retimer.push(image, image)
    last, copies = retimer.flush()
    written += [last] * copies
    return written, fake.calls, retimer


def label(item):
    return item if isinstance(item, tuple) else int(item[0, 0, 0])


def test_doubling_puts_one_new_frame_between_each_pair():
    written, calls, retimer = run(30, 60, 4)
    assert [label(x) for x in written] == [
        0, ("between", 0, 1, 0.5), 1, ("between", 1, 2, 0.5), 2, ("between", 2, 3, 0.5), 3, 3,
    ]  # fmt: skip
    assert retimer.interpolated == 3


def test_24_to_60_uses_the_right_moments():
    _, calls, _ = run(24, 60, 3)
    assert calls == [(0, 1, 0.4), (0, 1, 0.8), (1, 2, 0.2), (1, 2, 0.6)]


RATES = [(24, 60), (23.976, 60), (25, 60), (29.97, 60), (30, 144), (12, 30)]


@pytest.mark.parametrize(("input_fps", "output_fps"), RATES)
@pytest.mark.parametrize("count", [1, 2, 7, 50, 301])
def test_same_frame_count_as_repeating(input_fps, output_fps, count):
    """Same number of frames as repeating gives, so the duration and audio don't change."""
    written, _, _ = run(input_fps, output_fps, count)
    repeated = sum(repeat_count(n, input_fps, output_fps) for n in range(1, count + 1))
    assert len(written) == repeated


def test_moments_next_to_an_original_frame_show_it():
    """29.97 → 60: half the moments sit right on (or a hair off) an original frame."""
    _, calls, _ = run(29.97, 60, 40)
    assert calls and all(0.05 < t < 0.95 for *_, t in calls)
    assert len(calls) < 50  # far fewer than the 79 frames after the first one


def test_scene_cuts_are_repeated_not_blended():
    frames = [frame(0, noise=1), frame(1, noise=1), frame(2, noise=2), frame(3, noise=2)]
    frames[1][1:, :] = frames[0][1:, :]  # 0 and 1: same shot
    frames[3][1:, :] = frames[2][1:, :]  # 2 and 3: same shot, a different one
    written, calls, retimer = run(30, 60, 4, frames)
    assert [c[:2] for c in calls] == [(0, 1), (2, 3)]  # nothing drawn across the cut
    assert retimer.scene_cuts == 1
    assert [label(x) for x in written][3:5] == [1, 2]


def test_similarity_tells_a_shot_from_a_cut():
    still = frame(0, noise=5)
    shifted = np.roll(still, 1, axis=1)
    assert similarity(_thumbnail(still), _thumbnail(still)) == pytest.approx(1.0)
    assert similarity(_thumbnail(still), _thumbnail(shifted)) > SCENE_CUT_SIMILARITY
    assert similarity(_thumbnail(still), _thumbnail(frame(0, noise=6))) < SCENE_CUT_SIMILARITY


# -- the real network --------------------------------------------------------------------


def square_at(x, size=(96, 160)):
    """A gray frame with a white 24x24 square whose left edge is at ``x``."""
    image = np.full((*size, 3), 40, np.uint8)
    image[36:60, x : x + 24] = 255
    return image


def square_x(image):
    columns = np.where(image[48, :, 0] > 150)[0]
    return (columns.min() + columns.max() + 1) / 2 - 12  # left edge


@pytest.fixture(scope="module")
def rife_weights():
    try:
        return ensure_rife_weights(Path("weights"))
    except OSError as exc:  # no network
        pytest.skip(f"RIFE weights not available: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_rife_moves_things_to_the_moment_asked(rife_weights, t):
    from framelift.device import Device

    rife = FrameInterpolator(rife_weights, Device("cpu", "CPU"), 160, 96)
    middle = rife.between(rife.prepare(square_at(40)), rife.prepare(square_at(80)), t)
    assert middle.shape == (96, 160, 3)
    assert square_x(middle) == pytest.approx(40 + 40 * t, abs=2)


@pytest.mark.slow
def test_openvino_draws_the_same_frames(rife_weights, tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    from framelift import device as devices
    from framelift.device import Device

    # Pretend the CPU is the Intel GPU, and keep this test's converted models apart.
    monkeypatch.setattr(devices, "OPENVINO_DEVICE", "CPU")
    weights = tmp_path / rife_weights.name
    shutil.copy(rife_weights, weights)

    before, after = square_at(40), square_at(80)
    results = []
    for kind in ("cpu", "igpu"):
        rife = FrameInterpolator(weights, Device(kind, kind), 160, 96)
        results.append(rife.between(rife.prepare(before), rife.prepare(after), 0.5))
    assert np.abs(results[0].astype(int) - results[1].astype(int)).max() <= 1
    assert any((tmp_path / "openvino_cache").glob("rife-*.xml"))  # saved for next time


@pytest.mark.parametrize(("input_fps", "output_fps"), RATES + [(29.99, 60)])
@pytest.mark.parametrize("count", [1, 2, 7, 50, 301])
def test_the_estimate_matches_what_rife_draws(input_fps, output_fps, count):
    from framelift.planning import in_between_count

    _, calls, retimer = run(input_fps, output_fps, count)
    assert in_between_count(input_fps, output_fps, count) == retimer.interpolated == len(calls)


def test_nothing_is_drawn_between_identical_frames():
    still = frame(7)
    prepared = []
    fake = FakeInterpolator()
    fake.prepare = lambda image: prepared.append(image) or image
    retimer = Retimer(30, 60, fake)
    written = []
    for image in (still, still.copy(), still.copy(), frame(9)):
        written += retimer.push(image, image)
    assert len(fake.calls) == 1  # only between the last still and the new frame
    assert retimer.unchanged == 2
    assert len(prepared) == 2  # the copies of the still weren't prepared again
