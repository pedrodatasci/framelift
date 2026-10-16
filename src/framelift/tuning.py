"""``framelift tune``: find good settings for *this* video on *this* machine.

What it does, in order:

1. **Looks at the footage.** Noise, softness and resolution are measured on frames
   spread across the video, and one sharp, steady frame is picked for the tests.
2. **Measures the machine.** A real frame is upscaled with each candidate tile size
   (plus FP16 on NVIDIA, and thread counts on CPU); the fastest setup that fits in
   memory wins.
3. **Measures what the AI does to this footage.** Small crops (with some context
   around them, so the AI doesn't see artificial borders) are upscaled with and without
   AI, and the difference is measured: how much sharpness comes back, whether noise goes
   down or texture gets invented, and whether the structure is preserved. That decides
   the AI strength, separately for each model.
4. **Builds two presets**: ``fast`` (the light video model) and ``best`` (a heavier,
   slower model with more careful encoding).
5. **Times a short real run.** A few frames go through the whole pipeline with the
   preset's settings (reading, pre-cleaning, AI, blend and encoding, as in a real run),
   which gives time estimates that include model loading and encoding. On CPU, ``best``
   is extrapolated from it instead, because a single frame can take minutes.
6. **Renders a comparison sheet** so the final call can be made by eye.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import statistics
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import torch

from .analysis import (
    AIEffect,
    SizeSuggestion,
    build_comparison_sheet,
    describe_noise,
    describe_softness,
    estimate_noise,
    most_detailed_region,
    rank_test_frames,
    sample_positions,
    softness,
    suggest_ai_strength,
    suggest_size,
)
from .device import (
    DEFAULT_CPU_THREADS,
    Device,
    configure_torch,
    gpu_memory_gb,
    intel_gpu_name,
    release_memory,
    resolve_device,
    synchronize,
)
from .encoder import nvenc_available, require_binary
from .errors import FrameliftError, OutOfMemoryError
from .filters import Frame, get_profile
from .frames import blend_with_classic_upscale, mean_brightness, resize_exact
from .models import ensure_weights, get_model
from .options import DEFAULT_MODEL, EnhanceOptions
from .pipeline import VideoEnhancer
from .planning import RunPlan, plan_run
from .tune_cli import run_label, workers_label
from .upscaler import AIUpscaler
from .video import VideoInfo, probe_video, read_frame_pairs

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BEST_MODELS = {"live": "realesrgan-x4plus", "anime": "realesrgan-x4plus-anime-6B"}
BEST_TIME_BUDGET_SECONDS = 10 * 60
"""``best`` is recommended over ``fast`` when it's estimated to finish within this."""

GPU_TILES = (0, 1024, 768, 512, 256)
CPU_TILES = (256, 128)
INTEL_GPU_TILES = (0, 512, 256)  # memory is shared with the system, so smaller tiles too
WORKER_TEST_FRAMES = 5
"""Frames per real run when comparing 1 and 2 AI workers (two finish at a time)."""
WORKER_GAIN = 0.95
"""2 workers are kept only if they're at least 5% faster: within noise, simpler wins."""
SHEET_STRENGTHS = (0.0, 0.3, 0.7, 1.0)
SHEET_TILE_SIZE = (400, 300)  # width, height of each crop, in output pixels
EFFECT_CROP_SIZE = (160, 120)  # source pixels used to measure what the AI does
EFFECT_FRAMES = 3  # test frames (one crop each) the AI's effect is averaged over
REAL_RUN_FRAMES = 3
"""Frames pushed through the whole pipeline to time a preset like a real run would."""
_CROP_MARGIN = 12  # source pixels of context around each crop, trimmed afterwards


@dataclass(frozen=True)
class BenchmarkRun:
    """One configuration tried during the speed test."""

    tile: int
    half: bool
    seconds_per_frame: float | None = None
    error: str | None = None
    threads: int | None = None
    """CPU threads used, when this run was part of the thread test."""

    device: str = "cpu"
    """Where it ran: ``"cpu"``, ``"cuda"`` or ``"igpu"`` (Intel GPU, through OpenVINO)."""

    backend: str = "torch"
    """What ran the AI: ``"torch"`` or ``"openvino"``."""

    @property
    def ok(self) -> bool:
        return self.seconds_per_frame is not None


@dataclass(frozen=True)
class ModelTest:
    """What one model does to this footage, and the AI strength that follows."""

    model: str
    effect: AIEffect
    ai_strength: float
    reasons: list[str]
    relative_cost: float
    """Time per frame relative to the ``fast`` model (1.0 = the same)."""


@dataclass(frozen=True)
class RealRun:
    """A few frames pushed through the whole pipeline, timed like a real run."""

    frames: int
    seconds_per_frame: float
    """Steady pace: reading, pre-cleaning, AI, blend and encoding, all included."""

    overhead_seconds: float
    """One-time costs: probing, loading the model, starting and finishing FFmpeg."""

    workers: int = 1
    """Frames the AI worked on at the same time."""

    threads: int | None = None
    """CPU threads per worker (``None`` on a GPU)."""


@dataclass(frozen=True)
class Preset:
    """A ready-to-use set of options, with its expected cost."""

    name: str
    options: EnhanceOptions
    seconds_per_frame: float | None
    estimated_seconds: float | None
    description: str
    basis: str = ""
    """How the time was worked out: ``"measured"`` (a short real run of this preset),
    ``"extrapolated"`` (from another preset's real run) or ``""`` (no estimate)."""


@dataclass(frozen=True)
class TuneReport:
    """Everything ``tune_video`` found out, plus the presets it recommends."""

    video: VideoInfo
    device: Device
    gpu_memory_gb: float | None
    nvenc: bool
    noise: float
    noise_label: str
    suggested_profile: str
    softness: float
    softness_label: str
    size: SizeSuggestion
    plan: RunPlan
    runs: list[BenchmarkRun]
    preprocess_seconds: float
    model_tests: list[ModelTest]
    presets: dict[str, Preset]
    recommended_preset: str
    sample_frame: int
    sheet_path: Path | None = None
    default_threads: int | None = None
    real_run: RealRun | None = None
    """The short real run of the ``fast`` preset (``None`` without the speed test)."""
    real_runs: tuple[RealRun, ...] = ()
    """Every real run of the ``fast`` preset: with 1 and 2 workers, when that was tested."""

    @property
    def recommended(self) -> EnhanceOptions:
        """Options of the recommended preset."""
        return self.presets[self.recommended_preset].options

    @property
    def best_run(self) -> BenchmarkRun | None:
        """The fastest successful speed-test run."""
        successful = [run for run in self.runs if run.ok]
        return min(successful, key=lambda run: run.seconds_per_frame) if successful else None

    @property
    def seconds_per_frame(self) -> float | None:
        return self.presets[self.recommended_preset].seconds_per_frame

    @property
    def estimated_seconds(self) -> float | None:
        """Estimated total time for the recommended preset (model loading and encoding
        included; the audio pass, usually a few seconds, is not)."""
        return self.presets[self.recommended_preset].estimated_seconds


def tune_video(
    input_path: PathLike,
    options: EnhanceOptions | None = None,
    *,
    keep: Iterable[str] = (),
    samples: int = 6,
    benchmark: bool = True,
    try_best: bool = True,
    content: str = "live",
    sheet_path: PathLike | None = None,
) -> TuneReport:
    """Analyse a video and recommend settings for enhancing it.

    ``options`` is the starting point. Option names listed in ``keep`` are
    treated as your decision and never overridden (e.g. ``keep={"profile"}``).
    ``content`` is ``"live"`` or ``"anime"`` and picks the model of the ``best``
    preset; ``try_best=False`` skips that heavier model entirely. Pass
    ``sheet_path`` to also render a comparison sheet PNG there.
    """
    options = options or EnhanceOptions()
    options.validate()
    keep = set(keep)
    unknown = keep - set(EnhanceOptions.field_names())
    if unknown:
        raise ValueError(f"Unknown option(s) in keep: {', '.join(sorted(unknown))}")
    if content not in BEST_MODELS:
        raise ValueError(f"content must be one of {', '.join(BEST_MODELS)}")

    require_binary("ffmpeg")
    video = probe_video(input_path)

    # 1. Look at the footage ---------------------------------------------------------
    positions = sample_positions(video.frame_count, samples) or [1]
    pairs = read_frame_pairs(video.path, positions)
    noise = statistics.median(estimate_noise(frame) for _, frame, _ in pairs)
    soft = statistics.median(softness(frame) for _, frame, _ in pairs)
    noise_label, suggested_profile = describe_noise(noise)
    size = suggest_size(video.width, video.height)
    test_frames = rank_test_frames(pairs)
    sample_number, sample = test_frames[0]

    log.info(
        "Checked %d frames: %s (noise %.1f), %s (softness %.2f), %dp. Testing on frame %d.",
        len(pairs), noise_label, noise, describe_softness(soft), soft, video.height,
        sample_number,
    )  # fmt: skip

    looks = {}
    if "profile" not in keep:
        looks["profile"] = suggested_profile
    if not keep & {"scale", "same_resolution"}:
        looks.update(same_resolution=size.same_resolution, scale=size.scale)
    looked = dataclasses.replace(options, **looks)
    plan = plan_run(video, looked)
    profile = get_profile(looked.profile)

    # 2. Look at the machine -----------------------------------------------------------
    configure_torch(options.torch_threads)
    device = resolve_device(options.device, options.gpu_id)
    memory = gpu_memory_gb(device)
    nvenc = device.is_cuda and nvenc_available()
    default_threads = None if device.is_cuda else torch.get_num_threads()

    fast_model = options.model if "model" in keep else DEFAULT_MODEL
    best_model = None if ("model" in keep or not try_best) else BEST_MODELS[content]
    if best_model == fast_model:
        best_model = None

    fast_spec = get_model(fast_model)
    fast_weights = ensure_weights(fast_spec, Path(options.weights_dir).resolve())
    cleaned = profile.apply(sample)
    preprocess_seconds = _time_it(lambda: profile.apply(sample))

    # 3. Speed test (fast model, full frame) --------------------------------------------
    runs: list[BenchmarkRun] = []
    if benchmark:
        log.info("Speed test: %s on %s (a real %dx%d frame)…", fast_spec.name, device.name,
                 video.width, video.height)  # fmt: skip
        runs = _benchmark(fast_spec, fast_weights, device, cleaned, plan.scale, options, keep)
        if not device.is_gpu and "torch_threads" not in keep:
            runs += _thread_test(fast_spec, fast_weights, device, cleaned, plan.scale, runs)
        if device.kind == "cpu":
            runs += _openvino_tests(fast_spec, fast_weights, cleaned, plan.scale, options, keep,
                                    runs, default_threads)  # fmt: skip

    speed = _speed_settings(options, keep, device, runs, nvenc, default_threads)
    if speed.get("device", device.kind) != device.kind:
        device = resolve_device(speed["device"], options.gpu_id)
        log.info("  → the AI will run on the %s", device.name)
    configure_torch(speed.get("torch_threads", options.torch_threads))

    # 4. What the AI does to this footage, per model --------------------------------------
    crops = [
        _with_context(frame, most_detailed_region(frame, *EFFECT_CROP_SIZE))
        for _, frame in test_frames[:EFFECT_FRAMES]
    ]
    upscalers: dict[str, AIUpscaler] = {}
    model_tests: list[ModelTest] = []
    fast_crop_seconds = None
    for model in filter(None, (fast_model, best_model)):
        spec = get_model(model)
        weights = ensure_weights(spec, Path(options.weights_dir).resolve())
        log.info("Measuring what %s does to this footage…", spec.name)
        upscalers[model] = AIUpscaler(spec, weights, device, tile=0, half=speed.get("half", False),
                                      backend=speed.get("backend", options.backend))  # fmt: skip
        test, crop_seconds = _test_model(model, upscalers[model], crops, profile, plan.scale,
                                         keep, looked.ai_strength, fast_crop_seconds,
                                         soft)  # fmt: skip
        fast_crop_seconds = fast_crop_seconds or crop_seconds
        model_tests.append(test)
        release_memory(device)

    # 5. Presets, timed with a short real run ----------------------------------------------
    base = dataclasses.replace(looked, **speed)
    fast_spf = _per_frame(runs, preprocess_seconds)
    fast = _fast_options(base, model_tests[0], keep, fast_spf)
    best = _best_options(base, model_tests[1], keep, speed) if len(model_tests) > 1 else None

    real_runs: list[RealRun] = []
    if fast_spf is not None:
        if _worker_test_applies(device, keep, fast, default_threads):
            log.info("Timing short real runs with 1 and 2 AI workers (%d frames each, encoding "
                     "included)…", WORKER_TEST_FRAMES)  # fmt: skip
            fast, real_runs = _worker_test(video, fast, sample_number, default_threads)
            if best is not None and "workers" not in keep:
                best = dataclasses.replace(best, workers=fast.workers,
                                           torch_threads=fast.torch_threads)  # fmt: skip
        else:
            log.info("Timing a short real run (%d frames, encoding included)…", REAL_RUN_FRAMES)
            real_runs = [run for run in [_real_run(video, fast, sample_number)] if run]
    real_run = next((run for run in real_runs if run.workers == fast.workers), None)

    presets = {"fast": _preset("fast", fast, real_run, plan, "light video model")}
    if best is not None:
        describe = "heavier model, more careful encoding"
        if device.is_gpu and real_run is not None:
            # GPUs are fast enough to time the real thing for this preset too.
            presets["best"] = _preset("best", best, _real_run(video, best, sample_number),
                                      plan, describe)  # fmt: skip
        else:
            # On CPU a single frame can take minutes: time one tile of it instead and
            # put it together with the fast run's reading, blending and encoding.
            ai_fast = min((run.seconds_per_frame for run in runs if run.ok), default=None)
            ai_best = None
            if real_run is not None:
                log.info("Timing one %s tile…", get_model(best.model).name)
                ai_best = _tiled_seconds(upscalers[best.model], cleaned, best)
            presets["best"] = _extrapolated_preset("best", best, real_run, ai_fast, ai_best, plan,
                                                   describe)  # fmt: skip

    best_preset = presets.get("best")
    recommended = (
        "best"
        if best_preset
        and best_preset.estimated_seconds
        and best_preset.estimated_seconds <= BEST_TIME_BUDGET_SECONDS
        else "fast"
    )

    # 6. Comparison sheet ------------------------------------------------------------------
    written_sheet = None
    if sheet_path is not None:
        rows = _sheet_rows(looked.profile, "profile" in keep, presets, upscalers)
        image = _render_sheet(video, sample_number, sample, rows, presets, plan)
        written_sheet = _write_png(Path(sheet_path), image)
        log.info("Comparison sheet saved to %s", written_sheet)

    return TuneReport(
        video=video,
        device=device,
        gpu_memory_gb=memory,
        nvenc=nvenc,
        noise=noise,
        noise_label=noise_label,
        suggested_profile=suggested_profile,
        softness=soft,
        softness_label=describe_softness(soft),
        size=size,
        plan=plan,
        runs=runs,
        preprocess_seconds=preprocess_seconds,
        model_tests=model_tests,
        presets=presets,
        recommended_preset=recommended,
        sample_frame=sample_number,
        sheet_path=written_sheet,
        default_threads=default_threads,
        real_run=real_run,
        real_runs=tuple(real_runs),
    )


# -- speed test ----------------------------------------------------------------------------


def _candidates(device: Device, frame: Frame, options: EnhanceOptions, keep: set[str]):
    """Which tile sizes (and FP16 or not) are worth trying on this device."""
    if "tile" in keep:
        tiles = [options.tile]
    else:
        # A tile as big as the frame is the same as no tiling, so skip duplicates.
        longest_side = max(frame.shape[:2])
        pool = {"cuda": GPU_TILES, "igpu": INTEL_GPU_TILES}.get(device.kind, CPU_TILES)
        tiles = [tile for tile in pool if 0 < tile < longest_side]
        if device.is_gpu or not tiles:
            tiles.insert(0, 0)

    half = options.half if "half" in keep else device.is_gpu
    return tiles, bool(half and device.is_gpu)


def _benchmark(spec, weights, device, frame, scale, options, keep) -> list[BenchmarkRun]:
    tiles, half = _candidates(device, frame, options, keep)
    repeats = 2 if device.is_gpu else 1
    backend = "openvino" if device.kind == "igpu" else options.backend

    def measure(tile, half):
        return _measure(spec, weights, device, frame, scale, tile, half, repeats, backend=backend)

    runs = [measure(tile, half) for tile in tiles]

    # FP16 is a big win on modern NVIDIA cards but *slower* on some older ones
    # (e.g. GTX 10xx), so double-check the winner in full precision too.
    successful = [run for run in runs if run.ok]
    if half and "half" not in keep and successful:
        best = min(successful, key=lambda run: run.seconds_per_frame)
        runs.append(measure(best.tile, False))

    return runs


def _openvino_tests(spec, weights, frame, scale, options, keep, runs, default_threads):
    """OpenVINO's two chances on a CPU-only machine: the Intel GPU, and the CPU itself.

    Neither is chosen blindly: OpenVINO is often faster than PyTorch, but not always,
    and an integrated GPU can be slower than a strong CPU. Both are simply timed.
    """
    best = min((run for run in runs if run.ok), key=lambda r: r.seconds_per_frame, default=None)
    if best is None or import_openvino_quietly() is None:
        return []

    results = []
    if "backend" not in keep and options.backend == "torch":
        threads = best.threads or default_threads
        cpu = Device("cpu", "CPU")
        with _cpu_threads(threads):
            results.append(_measure(spec, weights, cpu, frame, scale, best.tile, False, 1,
                                    threads=best.threads, backend="openvino"))  # fmt: skip

    intel_gpu = intel_gpu_name()
    if intel_gpu and "device" not in keep:
        log.info("Speed test on the Intel GPU (%s, through OpenVINO)…", intel_gpu)
        results += _benchmark(spec, weights, Device("igpu", intel_gpu), frame, scale, options, keep)
    return results


def import_openvino_quietly():
    """The ``openvino`` module, or ``None`` if it isn't installed (it's optional)."""
    try:
        import openvino
    except ImportError:
        return None
    return openvino


@contextlib.contextmanager
def _cpu_threads(threads: int | None):
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def thread_candidates(default: int) -> list[int]:
    """Thread counts worth trying on CPU: all of them, half, and 4.

    Laptops that mix performance and efficiency cores are often *faster* with
    fewer threads, because the slow cores hold the fast ones back.
    """
    return sorted({default, max(1, default // 2), max(1, min(4, default))}, reverse=True)


def _thread_test(spec, weights, device, frame, scale, runs) -> list[BenchmarkRun]:
    default = torch.get_num_threads()
    candidates = [n for n in thread_candidates(default) if n != default]
    best = min((run for run in runs if run.ok), key=lambda r: r.seconds_per_frame, default=None)
    if best is None or not candidates:
        return []

    results = []
    try:
        for threads in candidates:
            torch.set_num_threads(threads)
            run = _measure(spec, weights, device, frame, scale, best.tile, False, 1, threads)
            results.append(run)
    finally:
        torch.set_num_threads(default)
    return results


def _measure(spec, weights, device, frame, scale, tile, half, repeats, threads=None,
             backend="torch"):  # fmt: skip
    if device.kind == "igpu":
        backend = "openvino"
    label = run_label(device.kind, backend, tile, half, threads)
    last_output = []
    describe = {"device": device.kind, "backend": backend, "threads": threads}

    try:
        upscaler = AIUpscaler(spec, weights, device, tile=tile, half=half, backend=backend)

        def upscale_once() -> None:
            last_output[:] = [upscaler.upscale(frame, scale)]

        # The first call pays one-time setup costs. On GPU, cuDNN also tunes itself
        # for this exact frame size, and OpenVINO compiles each tile size, so warm up
        # with the full frame; for PyTorch on CPU a small crop is enough and much cheaper.
        warm_up = frame if device.is_gpu or backend == "openvino" else frame[:64, :64]
        _timed(device, lambda: upscaler.upscale(warm_up, scale))
        seconds = min(_timed(device, upscale_once) for _ in range(repeats))
    except OutOfMemoryError:
        log.info("  %-36s out of memory", label)
        return BenchmarkRun(tile, half, error="out of memory", **describe)
    except (FrameliftError, RuntimeError) as exc:  # e.g. an Intel GPU driver giving up
        log.info("  %-36s failed (%s)", label, str(exc).splitlines()[0][:60])
        return BenchmarkRun(tile, half, error="failed", **describe)
    finally:
        release_memory(device)

    if mean_brightness(last_output[0]) < 3:
        log.info("  %-36s produced a black frame", label)
        return BenchmarkRun(tile, half, error="black output", **describe)

    log.info("  %-36s %.2f s/frame", label, seconds)
    return BenchmarkRun(tile, half, seconds_per_frame=seconds, **describe)


def _timed(device: Device, work) -> float:
    """Wall time of ``work()``, waiting for the GPU so the number is real."""
    synchronize(device)
    started = time.perf_counter()
    work()
    synchronize(device)
    return time.perf_counter() - started


def _time_it(work, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        timings.append(time.perf_counter() - started)
    return min(timings)


def _speed_settings(options, keep, device, runs, nvenc, default_threads) -> dict:
    settings = {}
    if "device" not in keep:
        settings["device"] = device.kind

    best = min((run for run in runs if run.ok), key=lambda run: run.seconds_per_frame, default=None)
    if best is not None:
        if "device" not in keep:
            settings["device"] = best.device
        if "backend" not in keep and best.device == "cpu":
            settings["backend"] = best.backend
        if "tile" not in keep:
            settings["tile"] = best.tile
        if "half" not in keep:
            settings["half"] = best.half
        if best.threads is not None and best.threads != default_threads:
            settings["torch_threads"] = best.threads

    if "encoder" not in keep:
        settings["encoder"] = "nvenc" if nvenc and settings.get("device") == "cuda" else "cpu"

    return settings


def _per_frame(runs, preprocess_seconds) -> float | None:
    best = min((run for run in runs if run.ok), key=lambda run: run.seconds_per_frame, default=None)
    if best is None:
        return None
    # The reader pre-cleans frames in parallel with the AI: the slower one sets the pace.
    return max(best.seconds_per_frame, preprocess_seconds)


# -- what the AI does ------------------------------------------------------------------------


def _with_context(frame: Frame, region) -> tuple[Frame, tuple[int, int, int, int]]:
    """``region`` cut out with a margin of real picture around it.

    Returns the bigger crop and where ``region`` sits inside it (x, y, width, height).
    The AI and the clean-up filters behave differently at an image border, so they get
    to see real context there, and the border is trimmed off afterwards.
    """
    height, width = frame.shape[:2]
    left, top = max(0, region.x - _CROP_MARGIN), max(0, region.y - _CROP_MARGIN)
    right = min(width, region.x + region.width + _CROP_MARGIN)
    bottom = min(height, region.y + region.height + _CROP_MARGIN)
    inner = (region.x - left, region.y - top, region.width, region.height)
    return frame[top:bottom, left:right], inner


def _test_model(model, upscaler, crops, profile, scale, keep, kept_strength, fast_seconds,
                source_softness):  # fmt: skip
    """Measure the AI's effect on the test crops, and how long the crops take."""
    effects, seconds = [], 0.0
    for index, (crop, (x, y, width, height)) in enumerate(crops):
        cleaned = profile.apply(crop)
        crop_height, crop_width = cleaned.shape[:2]
        size = (round(crop_width * scale), round(crop_height * scale))
        started = time.perf_counter()
        enhanced = resize_exact(upscaler.upscale(cleaned, scale), *size)
        if index > 0 or len(crops) == 1:  # the first call includes one-time setup
            seconds += time.perf_counter() - started
        classic = resize_exact(cleaned, *size)

        # Measure only the region itself, without the context around it.
        left, top = round(x * scale), round(y * scale)
        right, bottom = left + round(width * scale), top + round(height * scale)
        effects.append(
            AIEffect.measure(
                cleaned[y : y + height, x : x + width],
                classic[top:bottom, left:right],
                enhanced[top:bottom, left:right],
            )
        )

    effect = AIEffect.average(effects)
    if "ai_strength" in keep:
        strength, reasons = kept_strength, ["your choice"]
    else:
        strength, reasons = suggest_ai_strength(effect, source_softness)

    relative = 1.0 if not fast_seconds else seconds / fast_seconds
    return ModelTest(model, effect, strength, reasons, relative), seconds


# -- presets ------------------------------------------------------------------------------------


def _fast_options(base, test, keep, per_frame) -> EnhanceOptions:
    changes = {"model": test.model, "ai_strength": test.ai_strength}
    # When the AI is slow, libx264 has time to spare: "medium" makes smaller files
    # at the same quality for practically free.
    if per_frame is not None and per_frame >= 0.25 and "x264_preset" not in keep:
        changes["x264_preset"] = "medium"
    return dataclasses.replace(base, **_unkept(changes, keep))


def _best_options(base, test, keep, speed) -> EnhanceOptions:
    changes = {
        "model": test.model,
        "ai_strength": test.ai_strength,
        "pre_pad": 10,  # less risk of artifacts at the frame edges
        "crf": 18,  # the encoder smooths away less of the recovered detail
    }
    if speed.get("encoder", base.encoder) == "cpu":
        changes["x264_preset"] = "slow"
    return dataclasses.replace(base, **_unkept(changes, keep))


def _preset(name, options, run: RealRun | None, plan, description) -> Preset:
    """A preset timed by its own short real run (no estimate if there was none)."""
    if run is None:
        return Preset(name, options, None, None, description)
    total = run.overhead_seconds + run.seconds_per_frame * plan.frames_to_process
    return Preset(name, options, run.seconds_per_frame, total, description, "measured")


def _extrapolated_preset(name, options, fast_run, ai_fast, ai_best, plan, description):
    """A preset timed from the fast preset's real run, with this preset's AI time."""
    if fast_run is None or ai_fast is None or ai_best is None:
        return Preset(name, options, None, None, description)
    per_frame = max(fast_run.seconds_per_frame - ai_fast + ai_best, ai_best)
    total = fast_run.overhead_seconds + per_frame * plan.frames_to_process
    return Preset(name, options, per_frame, total, description, "extrapolated")


def tiled_area(width: int, height: int, tile: int, tile_pad: int, pre_pad: int = 0) -> int:
    """Pixels the AI really processes for one frame, overlaps between tiles included.

    Real-ESRGAN pads the frame by ``pre_pad`` and then cuts it into ``tile``-sized tiles
    that each overlap their neighbours by ``tile_pad``.
    """
    width, height = width + pre_pad, height + pre_pad
    if tile <= 0:
        return width * height

    def covered(length: int) -> int:
        return sum(
            min(start + tile + tile_pad, length) - max(start - tile_pad, 0)
            for start in range(0, length, tile)
        )

    return covered(width) * covered(height)


def _tiled_seconds(upscaler: AIUpscaler, frame: Frame, options: EnhanceOptions) -> float:
    """AI time per frame for ``options``, from timing a single tile-sized piece.

    A heavy model can take minutes per frame on CPU; one tile takes a fraction of that,
    and the cost grows with the pixels processed.
    """
    height, width = frame.shape[:2]
    side = options.tile + 2 * options.tile_pad if options.tile > 0 else max(width, height)
    piece = frame[: min(side, height), : min(side, width)]
    if upscaler.backend == "openvino":
        upscaler.upscale(piece, 1.0)  # compiles this size first, which a real run does once
    started = time.perf_counter()
    upscaler.upscale(piece, 1.0)
    seconds = time.perf_counter() - started
    area = tiled_area(width, height, options.tile, options.tile_pad, options.pre_pad)
    return seconds * area / (piece.shape[0] * piece.shape[1])


# -- short real run ------------------------------------------------------------------------------


class _TimedEnhancer(VideoEnhancer):
    """A :class:`VideoEnhancer` that notes when each frame is ready to be written."""

    reuse_repeated_frames = False  # measure the AI on every frame, stills included

    def __init__(self, options: EnhanceOptions) -> None:
        super().__init__(options)
        self.marks: list[float] = []

    def _finish_frame(self, *args, **kwargs) -> Frame:
        output = super()._finish_frame(*args, **kwargs)
        self.marks.append(time.perf_counter())
        return output


@contextlib.contextmanager
def _quiet(*names: str):
    """Hide a real run's own messages (plan table, warnings): the tuner reports instead."""
    loggers = [logging.getLogger(name) for name in names]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def _worker_test_applies(device, keep, options, default_threads) -> bool:
    """2 workers only make sense on a CPU with cores to share between them."""
    threads = options.torch_threads or default_threads or 1
    return device.kind == "cpu" and not keep & {"workers", "torch_threads"} and threads >= 4


def _worker_test(video, options, around_frame, default_threads):
    """Short real runs with 1 and 2 AI workers; returns the faster options and both runs.

    The same CPU threads are shared out: 1 worker × N threads against 2 × N/2. Each
    worker then runs its own frame, and while one waits on memory or finishes a frame
    the other keeps the cores busy, which a single, wider worker can't always do.
    """
    threads = options.torch_threads or default_threads
    one = dataclasses.replace(options, workers=1)
    # Leaving torch_threads at 0 lets the run split PyTorch's default between workers.
    split = max(1, threads // 2) if options.torch_threads else 0
    two = dataclasses.replace(options, workers=2, torch_threads=split)
    runs = [
        _real_run(video, one, around_frame, WORKER_TEST_FRAMES),
        _real_run(video, two, around_frame, WORKER_TEST_FRAMES),
    ]
    if runs[0] and runs[1] and runs[1].seconds_per_frame < runs[0].seconds_per_frame * WORKER_GAIN:
        chosen = two
    else:
        chosen = one
    return chosen, [run for run in runs if run]


def _real_run(video: VideoInfo, options: EnhanceOptions, around_frame: int,
              frames: int = REAL_RUN_FRAMES) -> RealRun | None:  # fmt: skip
    """Enhance a few frames for real (into a throwaway file) and time it.

    The frames go through the same pipeline as a real run, with the same settings, so
    the pace includes what the speed test leaves out: reading and pre-cleaning, the
    blend, encoding (which competes with the AI for the CPU) and the model loading.
    """
    frames = max(1, min(frames, video.frame_count))
    start = max(1, min(around_frame, video.frame_count - frames + 1))
    options = dataclasses.replace(
        options, start_frame=start, max_frames=frames, audio=False, debug_timings=False
    )
    enhancer = _TimedEnhancer(options)

    with tempfile.TemporaryDirectory(prefix="framelift-tune-") as folder:
        started = time.perf_counter()
        try:
            with _quiet("framelift.pipeline", "framelift.encoder"):
                enhancer.enhance(video.path, Path(folder) / "real_run.mp4", show_progress=False)
        except (FrameliftError, RuntimeError) as exc:
            log.info("  the real run didn't work (%s), so there are no time estimates", exc)
            return None
        total = time.perf_counter() - started

    marks = enhancer.marks
    # Frame to frame, so the first frame's own one-time costs are left out.
    per_frame = (marks[-1] - marks[0]) / (len(marks) - 1) if len(marks) > 1 else total
    overhead = max(0.0, total - per_frame * len(marks))

    workers = options.workers
    threads = None if resolve_device(options.device, options.gpu_id).is_gpu else (
        options.torch_threads or max(1, DEFAULT_CPU_THREADS // workers))  # fmt: skip
    log.info("  %s: %.2f s/frame, plus %.1f s to start and finish",
             workers_label(workers, threads), per_frame, overhead)  # fmt: skip
    return RealRun(len(marks), per_frame, overhead, workers=workers, threads=threads)


def _unkept(changes: dict, keep: set[str]) -> dict:
    return {name: value for name, value in changes.items() if name not in keep}


# -- comparison sheet ---------------------------------------------------------------------------


def _short_model_name(model: str) -> str:
    return model.replace("realesrgan-", "").replace("realesr-", "")


def _sheet_rows(profile, keep_profile, presets, upscalers):
    """(label lines, profile, upscaler) for each row of the sheet."""
    fast = presets["fast"].options
    rows = []
    if profile != "none" and not keep_profile:
        rows.append((("no clean-up", "none", _short_model_name(fast.model)), "none",
                     upscalers[fast.model]))  # fmt: skip
    for name, preset in presets.items():
        model = preset.options.model
        rows.append(((f"{name} preset", profile, _short_model_name(model)), profile,
                     upscalers[model]))  # fmt: skip
    return rows


def _sheet_columns(presets) -> tuple[list[float], list[str]]:
    marks: dict[float, list[str]] = {}
    for name, preset in presets.items():
        marks.setdefault(round(preset.options.ai_strength, 2), []).append(name)
    values = sorted(set(SHEET_STRENGTHS) | set(marks))

    labels = []
    for value in values:
        label = "no AI (plain resize)" if value == 0 else f"--ai-strength {value:g}"
        if value in marks:
            label += f"  ({', '.join(marks[value])})"
        labels.append(label)
    return values, labels


def _render_sheet(video, frame_number, frame, rows, presets, plan) -> Frame:
    scale = plan.width / video.width
    tile_width = min(SHEET_TILE_SIZE[0], plan.width)
    tile_height = min(SHEET_TILE_SIZE[1], plan.height)

    # The same region in source pixels, plus a little context so the AI doesn't
    # see artificial borders; the context is trimmed off after upscaling.
    region = most_detailed_region(
        frame, max(1, round(tile_width / scale)), max(1, round(tile_height / scale))
    )
    context, (inner_x, inner_y, _, _) = _with_context(frame, region)
    offset_x = round(inner_x * scale)
    offset_y = round(inner_y * scale)
    out_width = round(context.shape[1] * scale)
    out_height = round(context.shape[0] * scale)

    rendered = []
    for _, profile_name, upscaler in rows:
        cleaned = get_profile(profile_name).apply(context)
        rendered.append((cleaned, upscaler.upscale(cleaned, scale)))

    strengths, column_labels = _sheet_columns(presets)

    def render(row: int, col: int) -> Frame:
        cleaned, enhanced = rendered[row]
        mixed = blend_with_classic_upscale(cleaned, enhanced, out_width, out_height, strengths[col])
        tile = mixed[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width]
        # Rounding can leave a crop a pixel short; pad so every tile matches.
        return cv2.copyMakeBorder(
            tile, 0, tile_height - tile.shape[0], 0, tile_width - tile.shape[1],
            cv2.BORDER_REPLICATE,
        )  # fmt: skip

    name = video.path.name.encode("ascii", "replace").decode()
    title = f"{name} - frame {frame_number} - 100% crop of the {plan.width}x{plan.height} output"
    return build_comparison_sheet([label for label, _, _ in rows], column_labels, render, title)


def _write_png(path: Path, image: Frame) -> Path:
    # cv2.imwrite can't handle non-ASCII paths on Windows; encoding in memory can.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Couldn't encode the comparison sheet as PNG.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path.resolve()
