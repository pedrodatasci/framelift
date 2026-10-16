"""Full pipeline on a tiny clip. Needs PyTorch, FFmpeg and network (or a pre-downloaded
``weights/realesr-animevideov3.pth``). Run with ``pytest -m slow``."""

import threading

import pytest

# Skip the whole module *before* importing anything that pulls in PyTorch.
pytest.importorskip("torch")

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


def test_tune_video_builds_runnable_presets(make_video, tmp_path, options):
    from framelift import tune_video

    report = tune_video(
        make_video(frames=12), options, keep={"weights_dir"}, sheet_path=tmp_path / "s.png"
    )

    assert set(report.presets) == {"fast", "best"}
    fast, best = report.presets["fast"], report.presets["best"]
    assert fast.options.model == "realesr-animevideov3"
    assert best.options.model == "realesrgan-x4plus"
    assert best.estimated_seconds > fast.estimated_seconds > 0
    assert [test.model for test in report.model_tests] == [fast.options.model, best.options.model]
    assert report.recommended is report.presets[report.recommended_preset].options
    assert report.sheet_path.exists()
    for preset in report.presets.values():
        preset.options.validate()
        assert 0.2 <= preset.options.ai_strength <= 0.85


def test_tune_video_respects_pinned_settings(make_video, options):
    from framelift import tune_video

    options.profile, options.scale, options.ai_strength = "heavy_noise", 2, 0.33
    report = tune_video(
        make_video(frames=12),
        options,
        keep={"profile", "scale", "ai_strength"},
        benchmark=False,
        try_best=False,
    )

    assert list(report.presets) == ["fast"]
    assert (report.recommended.profile, report.recommended.scale) == ("heavy_noise", 2)
    assert report.recommended.ai_strength == 0.33
    assert report.model_tests[0].reasons == ["your choice"]
    assert report.runs == [] and report.estimated_seconds is None


def test_tune_video_anime_uses_the_anime_model(make_video, options):
    from framelift import tune_video

    report = tune_video(make_video(frames=8), options, content="anime", benchmark=False)
    assert report.presets["best"].options.model == "realesrgan-x4plus-anime-6B"


def test_enhance_video_keeps_the_audio(tmp_path, options):
    import subprocess

    source = tmp_path / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
        check=True,
    )  # fmt: skip

    with_audio = enhance_video(source, tmp_path / "a.mp4", options, max_frames=6,
                               show_progress=False)  # fmt: skip
    without = enhance_video(source, tmp_path / "b.mp4", options, max_frames=6, audio=False,
                            show_progress=False)  # fmt: skip

    assert with_audio.audio == "copied"
    assert without.audio == "disabled"


def test_input_without_audio_reports_none(make_video, tmp_path, options):
    result = enhance_video(make_video(frames=4), tmp_path / "o.mp4", options, show_progress=False)
    assert result.audio == "none"


def _frame_md5s(path):
    import hashlib

    import cv2

    capture, hashes = cv2.VideoCapture(str(path)), []
    while True:
        ok, image = capture.read()
        if not ok:
            return hashes
        hashes.append(hashlib.md5(image.tobytes()).hexdigest())


def test_workers_change_speed_not_output(make_video, tmp_path):
    """1 or 2 AI workers: the same frames, in the same order."""
    clip = make_video(frames=6)
    outputs = []
    for workers in (1, 2):
        output = tmp_path / f"w{workers}.mp4"
        result = enhance_video(clip, output, device="cpu", tile=32, workers=workers, audio=False,
                               show_progress=False)  # fmt: skip
        assert result.frames_written == 6
        outputs.append(_frame_md5s(output))
    assert outputs[0] == outputs[1]


def test_upscaler_matches_realesrgan_exactly(make_video):
    """AIUpscaler re-implements RealESRGANer.enhance with its own copy of the networks;
    the output must not change. Needs the original packages, which framelift itself no
    longer installs: ``pip install realesrgan basicsr``."""
    import contextlib
    import importlib.util
    import io
    import sys
    import types

    from framelift.device import resolve_device
    from framelift.models import ensure_weights, get_model
    from framelift.upscaler import AIUpscaler, build_network
    from framelift.video import read_frames

    if importlib.util.find_spec("realesrgan") is None:
        pytest.skip("the original Real-ESRGAN packages aren't installed")
    # basicsr imports a torchvision module removed in torchvision 0.17.
    shim = "torchvision.transforms.functional_tensor"
    try:
        __import__(shim)
    except ImportError:
        import torchvision.transforms.functional as functional

        sys.modules[shim] = types.SimpleNamespace(rgb_to_grayscale=functional.rgb_to_grayscale)
    from realesrgan import RealESRGANer

    frame = read_frames(make_video(frames=2), [1])[0][1]
    spec = get_model("realesr-animevideov3")
    weights = ensure_weights(spec, "weights")
    for tile, pre_pad, scale in [(32, 0, 1.5), (0, 10, 2.0), (40, 10, 4.0)]:
        ours = AIUpscaler(spec, weights, resolve_device("cpu"), tile=tile, pre_pad=pre_pad)
        theirs = RealESRGANer(scale=4, model_path=str(weights), model=build_network(spec),
                              tile=tile, tile_pad=10, pre_pad=pre_pad, half=False,
                              device="cpu")  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()):
            expected, _ = theirs.enhance(frame, outscale=scale)
        assert (ours.upscale(frame, scale) == expected).all()


def test_openvino_backend(make_video, monkeypatch):
    """OpenVINO on the CPU, and the Intel GPU path pointed at the CPU (no GPU in CI)."""
    pytest.importorskip("openvino")
    import framelift.device as devices
    from framelift.models import ensure_weights, get_model
    from framelift.upscaler import AIUpscaler
    from framelift.video import read_frames

    monkeypatch.setattr(devices, "OPENVINO_DEVICE", "CPU")
    frame = read_frames(make_video(frames=2), [1])[0][1]
    spec = get_model("realesr-animevideov3")
    weights = ensure_weights(spec, "weights")
    reference = AIUpscaler(spec, weights, devices.resolve_device("cpu"), tile=32).upscale(frame, 2)

    for device, backend in [("cpu", "openvino"), ("igpu", "torch")]:
        upscaler = AIUpscaler(spec, weights, devices.resolve_device(device), tile=32,
                              backend=backend)  # fmt: skip
        assert upscaler.backend == "openvino"
        difference = abs(upscaler.upscale(frame, 2).astype(int) - reference.astype(int))
        assert difference.max() <= 2


def test_tune_video_tries_two_workers_on_a_multicore_cpu(make_video, options, monkeypatch):
    import framelift.tuning as tuning

    monkeypatch.setattr(tuning.torch, "get_num_threads", lambda: 8)  # pretend: 8 threads
    monkeypatch.setattr(tuning, "_openvino_tests", lambda *args: [])
    report = tuning.tune_video(make_video(frames=12), options, keep={"tile"}, try_best=False)
    assert [run.workers for run in report.real_runs] == [1, 2]
    assert report.real_run in report.real_runs
    assert report.presets["fast"].options.workers == report.real_run.workers


@pytest.mark.parametrize("ai_off", [{"ai": False}, {"ai_strength": 0}])
def test_without_the_ai_no_model_is_loaded(make_video, tmp_path, monkeypatch, ai_off):
    """--no-ai (or --ai-strength 0): resize and frame-rate change only. No model, and the
    device doesn't matter (igpu would fail here without an Intel GPU)."""

    def no_model(*args):
        raise AssertionError("no model should be loaded without the AI")

    monkeypatch.setattr(VideoEnhancer, "_get_upscaler", no_model)
    output = tmp_path / "out.mp4"
    result = enhance_video(make_video(frames=6, fps=24), output, device="igpu", scale=2,
                           output_fps=60, audio=False, show_progress=False, **ai_off)  # fmt: skip

    assert (result.frames_processed, result.frames_written) == (6, 15)
    info = probe_video(output)
    assert (info.width, info.height, info.fps) == (128, 96, 60)


def test_without_the_ai_frames_are_only_resized(make_video, tmp_path):
    """Same size and no pre-cleaning: the frames reach the encoder untouched."""
    from framelift.video import read_frames

    clip = make_video(frames=3)
    seen = []
    enhancer = VideoEnhancer(EnhanceOptions(ai=False, same_resolution=True, audio=False))
    real_finish = enhancer._finish_frame

    def spy(frame, *args, **kwargs):
        output = real_finish(frame, *args, **kwargs)
        seen.append((frame.image, output))
        return output

    enhancer._finish_frame = spy
    enhancer.enhance(clip, tmp_path / "same.mp4", show_progress=False)

    originals = [image for _, image in read_frames(clip, [1, 2, 3])]
    assert len(seen) == 3
    for original, (source, output) in zip(originals, seen):
        assert (source == original).all() and (output == original).all()


@pytest.mark.parametrize("with_ai", [False, True])
def test_interpolate_writes_in_between_frames(make_video, tmp_path, with_ai):
    """--interpolate (with or without the AI): new frames, same frame count as repeating."""
    output = tmp_path / "smooth.mp4"
    result = enhance_video(make_video(frames=6, fps=24), output, device="cpu", tile=32,
                           output_fps=60, interpolate=True, ai=with_ai, audio=False,
                           show_progress=False)  # fmt: skip

    assert (result.frames_processed, result.frames_written) == (6, 15)
    assert result.frames_interpolated == 10  # 2 per pair of frames, 5 pairs
    info = probe_video(output)
    assert info.fps == 60 and info.frame_count == 15


def test_identical_frames_reuse_the_result(make_video, tmp_path):
    """Frames 25-30 of the test clip are identical: 5 of them skip the AI, and the video
    comes out exactly the same as running the AI on every frame."""
    clip = make_video(frames=30)
    outputs = []
    for reuse in (True, False):
        enhancer = VideoEnhancer(EnhanceOptions(device="cpu", tile=32, audio=False))
        enhancer.reuse_repeated_frames = reuse
        result = enhancer.enhance(clip, tmp_path / f"reuse_{reuse}.mp4", show_progress=False)
        assert result.frames_reused == (5 if reuse else 0)
        outputs.append(_decoded(tmp_path / f"reuse_{reuse}.mp4"))
    assert len(outputs[0]) == 30
    assert all((a == b).all() for a, b in zip(*outputs))


def _decoded(path):
    import cv2

    capture, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, image = capture.read()
        if not ok:
            return frames
        frames.append(image)
