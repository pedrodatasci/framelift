"""Full pipeline on a tiny clip. Needs PyTorch, Real-ESRGAN, FFmpeg and network
(or a pre-downloaded ``weights/realesr-animevideov3.pth``). Run with ``pytest -m slow``."""

import importlib.util
import threading

import pytest

# Skip the whole module *before* importing anything that pulls in PyTorch.
pytest.importorskip("torch")
if not (importlib.util.find_spec("basicsr") and importlib.util.find_spec("realesrgan")):
    pytest.skip("Real-ESRGAN not installed", allow_module_level=True)

from framelift import (  # noqa: E402
    EnhanceOptions,
    Profile,
    VideoEnhancer,
    enhance_video,
    probe_video,
)

pytestmark = pytest.mark.slow


@pytest.fixture
def options(tmp_path):
    return EnhanceOptions(device="cpu", tile=0, weights_dir="weights")


def test_enhance_video_upscales_and_counts_frames(make_video, tmp_path):
    output = tmp_path / "out.mp4"
    result = enhance_video(
        make_video(frames=6), output, device="cpu", scale=2, max_frames=4, show_progress=False
    )

    assert result.frames_processed == 4
    assert result.frames_written == 4
    assert result.resume_from == 5
    info = probe_video(output)
    assert (info.width, info.height) == (128, 96)


def test_enhancer_reuses_the_model_and_raises_fps(make_video, tmp_path, options):
    options.output_fps = 48
    enhancer = VideoEnhancer(options)
    clip = make_video(frames=4, fps=24)

    first = enhancer.enhance(clip, tmp_path / "a.mp4", show_progress=False)
    model = enhancer._upscaler
    second = enhancer.enhance(clip, tmp_path / "b.mp4", show_progress=False)

    assert enhancer._upscaler is model
    assert first.frames_written == second.frames_written == 8
    assert second.is_complete


def test_stop_event_saves_a_partial_video(make_video, tmp_path, options, monkeypatch):
    """Simulates a Ctrl+C that lands mid-run (after the reader has read 3 frames)."""
    stop = threading.Event()
    frames_read = 0
    real_apply = Profile.apply

    def apply_then_maybe_stop(self, frame):
        nonlocal frames_read
        frames_read += 1
        if frames_read == 3:
            stop.set()
        return real_apply(self, frame)

    monkeypatch.setattr(Profile, "apply", apply_then_maybe_stop)

    result = VideoEnhancer(options).enhance(
        make_video(frames=10), tmp_path / "p.mp4", stop_event=stop, show_progress=False
    )

    assert result.interrupted
    assert 1 <= result.frames_processed < 10
    assert result.resume_from == result.last_frame + 1
    assert probe_video(result.output_path).frame_count == result.frames_written
