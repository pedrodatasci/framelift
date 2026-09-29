"""``framelift tune``: find good settings for *this* video on *this* machine.

What it does, in order:

1. **Looks at the footage.** Noise, softness and resolution are measured on frames
   spread across the video, and one sharp, steady frame is picked for the tests.
2. **Measures the machine.** A real frame is upscaled with each candidate tile size
   (plus FP16 on NVIDIA, and thread counts on CPU); the fastest setup that fits in
   memory wins.
3. **Measures what the AI does to this footage.** Small crops are upscaled with and
   without AI, and the difference is measured: how much sharpness comes back, whether
   noise goes down or texture gets invented, and whether the structure is preserved.
   That decides the AI strength, separately for each model.
4. **Builds two presets**: ``fast`` (the light video model) and ``best`` (a heavier,
   slower model with more careful encoding), each with its own time estimate.
5. **Renders a comparison sheet** so the final call can be made by eye.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
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
    Device,
    configure_torch,
    gpu_memory_gb,
    release_memory,
    resolve_device,
    synchronize,
)
from .encoder import nvenc_available, require_binary
from .errors import OutOfMemoryError
from .filters import Frame, get_profile
from .frames import blend_with_classic_upscale, mean_brightness, resize_exact
from .models import ensure_weights, get_model
from .options import DEFAULT_MODEL, EnhanceOptions
from .planning import RunPlan, plan_run
from .upscaler import AIUpscaler
from .video import VideoInfo, probe_video, read_frame_pairs

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BEST_MODELS = {"live": "realesrgan-x4plus", "anime": "realesrgan-x4plus-anime-6B"}
BEST_TIME_BUDGET_SECONDS = 10 * 60
"""``best`` is recommended over ``fast`` when it's estimated to finish within this."""

GPU_TILES = (0, 1024, 768, 512, 256)
CPU_TILES = (256, 128)
SHEET_STRENGTHS = (0.0, 0.3, 0.7, 1.0)
SHEET_TILE_SIZE = (400, 300)  # width, height of each crop, in output pixels
EFFECT_CROP_SIZE = (160, 120)  # source pixels used to measure what the AI does
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
class Preset:
    """A ready-to-use set of options, with its expected cost."""

    name: str
    options: EnhanceOptions
    seconds_per_frame: float | None
    estimated_seconds: float | None
    description: str


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
        """Rough total time for the recommended preset (encoding not included)."""
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
        if not device.is_cuda and "torch_threads" not in keep:
            runs += _thread_test(fast_spec, fast_weights, device, cleaned, plan.scale, runs)

    speed = _speed_settings(options, keep, device, runs, nvenc, default_threads)
    if "torch_threads" in speed:
        configure_torch(speed["torch_threads"])

    # 4. What the AI does to this footage, per model --------------------------------------
    crops = [
        most_detailed_region(frame, *EFFECT_CROP_SIZE).crop(frame) for _, frame in test_frames[:2]
    ]
    upscalers: dict[str, AIUpscaler] = {}
    model_tests: list[ModelTest] = []
    fast_crop_seconds = None
    for model in filter(None, (fast_model, best_model)):
        spec = get_model(model)
        weights = ensure_weights(spec, Path(options.weights_dir).resolve())
        log.info("Measuring what %s does to this footage…", spec.name)
        upscalers[model] = AIUpscaler(spec, weights, device, tile=0, half=speed.get("half", False))
        test, crop_seconds = _test_model(model, upscalers[model], crops, profile, plan.scale,
                                         keep, looked.ai_strength, fast_crop_seconds,
                                         soft)  # fmt: skip
        fast_crop_seconds = fast_crop_seconds or crop_seconds
        model_tests.append(test)
        release_memory(device)

    # 5. Presets ---------------------------------------------------------------------------
    base = dataclasses.replace(looked, **speed)
    fast_spf = _per_frame(runs, preprocess_seconds)
    presets = {"fast": _fast_preset(base, model_tests[0], keep, fast_spf, plan)}
    if len(model_tests) > 1:
        presets["best"] = _best_preset(base, model_tests[1], keep, fast_spf, preprocess_seconds,
                                       plan, device, cleaned, speed)  # fmt: skip

    best = presets.get("best")
    recommended = (
        "best"
        if best and best.estimated_seconds and best.estimated_seconds <= BEST_TIME_BUDGET_SECONDS
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
    )


# -- speed test ----------------------------------------------------------------------------


def _candidates(device: Device, frame: Frame, options: EnhanceOptions, keep: set[str]):
    """Which tile sizes (and FP16 or not) are worth trying on this device."""
    if "tile" in keep:
        tiles = [options.tile]
    else:
        # A tile as big as the frame is the same as no tiling, so skip duplicates.
        longest_side = max(frame.shape[:2])
        pool = GPU_TILES if device.is_cuda else CPU_TILES
        tiles = [tile for tile in pool if 0 < tile < longest_side]
        if device.is_cuda or not tiles:
            tiles.insert(0, 0)

    half = options.half if "half" in keep else device.is_cuda
    return tiles, bool(half and device.is_cuda)


def _benchmark(spec, weights, device, frame, scale, options, keep) -> list[BenchmarkRun]:
    tiles, half = _candidates(device, frame, options, keep)
    repeats = 2 if device.is_cuda else 1

    runs = [_measure(spec, weights, device, frame, scale, tile, half, repeats) for tile in tiles]

    # FP16 is a big win on modern NVIDIA cards but *slower* on some older ones
    # (e.g. GTX 10xx), so double-check the winner in full precision too.
    successful = [run for run in runs if run.ok]
    if half and "half" not in keep and successful:
        best = min(successful, key=lambda run: run.seconds_per_frame)
        runs.append(_measure(spec, weights, device, frame, scale, best.tile, False, repeats))

    return runs


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


def _measure(spec, weights, device, frame, scale, tile, half, repeats, threads=None):
    label = f"tile {tile or 'off'}{', FP16' if half else ''}"
    if threads is not None:
        label += f", {threads} threads"
    last_output = []

    try:
        upscaler = AIUpscaler(spec, weights, device, tile=tile, half=half)

        def upscale_once() -> None:
            last_output[:] = [upscaler.upscale(frame, scale)]

        # The first call pays one-time setup costs. On GPU, cuDNN also tunes itself
        # for this exact frame size, so warm up with the full frame; on CPU a small
        # crop is enough and much cheaper.
        warm_up = frame if device.is_cuda else frame[:64, :64]
        _timed(device, lambda: upscaler.upscale(warm_up, scale))
        seconds = min(_timed(device, upscale_once) for _ in range(repeats))
    except OutOfMemoryError:
        log.info("  %-28s out of memory", label)
        return BenchmarkRun(tile, half, error="out of memory", threads=threads)
    finally:
        release_memory(device)

    if mean_brightness(last_output[0]) < 3:
        log.info("  %-28s produced a black frame", label)
        return BenchmarkRun(tile, half, error="black output", threads=threads)

    log.info("  %-28s %.2f s/frame", label, seconds)
    return BenchmarkRun(tile, half, seconds_per_frame=seconds, threads=threads)


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
        if "tile" not in keep:
            settings["tile"] = best.tile
        if "half" not in keep:
            settings["half"] = best.half
        if best.threads is not None and best.threads != default_threads:
            settings["torch_threads"] = best.threads

    if "encoder" not in keep:
        settings["encoder"] = "nvenc" if nvenc else "cpu"

    return settings


def _per_frame(runs, preprocess_seconds) -> float | None:
    best = min((run for run in runs if run.ok), key=lambda run: run.seconds_per_frame, default=None)
    if best is None:
        return None
    # The reader pre-cleans frames in parallel with the AI: the slower one sets the pace.
    return max(best.seconds_per_frame, preprocess_seconds)


# -- what the AI does ------------------------------------------------------------------------


def _test_model(model, upscaler, crops, profile, scale, keep, kept_strength, fast_seconds,
                source_softness):  # fmt: skip
    """Measure the AI's effect on the test crops, and how long a crop takes."""
    effects, seconds = [], None
    for index, crop in enumerate(crops):
        cleaned = profile.apply(crop)
        height, width = cleaned.shape[:2]
        size = (round(width * scale), round(height * scale))
        started = time.perf_counter()
        enhanced = resize_exact(upscaler.upscale(cleaned, scale), *size)
        if index == len(crops) - 1:  # the first call includes one-time setup
            seconds = time.perf_counter() - started
        classic = resize_exact(cleaned, *size)
        effects.append(AIEffect.measure(cleaned, classic, enhanced))

    effect = AIEffect.average(effects)
    if "ai_strength" in keep:
        strength, reasons = kept_strength, ["your choice"]
    else:
        strength, reasons = suggest_ai_strength(effect, source_softness)

    relative = 1.0 if not fast_seconds else seconds / fast_seconds
    return ModelTest(model, effect, strength, reasons, relative), seconds


# -- presets ------------------------------------------------------------------------------------


def _fast_preset(base, test, keep, per_frame, plan) -> Preset:
    changes = {"model": test.model, "ai_strength": test.ai_strength}
    # When the AI is slow, libx264 has time to spare: "medium" makes smaller files
    # at the same quality for practically free.
    if per_frame is not None and per_frame >= 0.25 and "x264_preset" not in keep:
        changes["x264_preset"] = "medium"
    options = dataclasses.replace(base, **_unkept(changes, keep))
    return Preset(
        "fast",
        options,
        per_frame,
        None if per_frame is None else per_frame * plan.frames_to_process,
        "light video model",
    )


def _best_preset(base, test, keep, fast_per_frame, preprocess, plan, device, frame, speed):
    changes = {
        "model": test.model,
        "ai_strength": test.ai_strength,
        "pre_pad": 10,  # less risk of artifacts at the frame edges
        "crf": 18,  # the encoder smooths away less of the recovered detail
    }
    if speed.get("encoder", base.encoder) == "cpu":
        changes["x264_preset"] = "slow"
    options = dataclasses.replace(base, **_unkept(changes, keep))

    per_frame = None
    if fast_per_frame is not None:
        if device.is_cuda:
            # GPUs are fast enough to time the real thing on a full frame.
            spec = get_model(test.model)
            weights = ensure_weights(spec, Path(base.weights_dir).resolve())
            run = _measure(spec, weights, device, frame, plan.scale, options.tile, options.half, 1)
            per_frame = max(run.seconds_per_frame or 0.0, preprocess) if run.ok else None
        else:
            # On CPU a full frame can take minutes; scale from the crop test instead.
            per_frame = max(fast_per_frame * test.relative_cost, preprocess)

    return Preset(
        "best",
        options,
        per_frame,
        None if per_frame is None else per_frame * plan.frames_to_process,
        "heavier model, more careful encoding",
    )


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
    left = max(0, region.x - _CROP_MARGIN)
    top = max(0, region.y - _CROP_MARGIN)
    right = min(video.width, region.x + region.width + _CROP_MARGIN)
    bottom = min(video.height, region.y + region.height + _CROP_MARGIN)
    offset_x = round((region.x - left) * scale)
    offset_y = round((region.y - top) * scale)
    out_width = round((right - left) * scale)
    out_height = round((bottom - top) * scale)

    rendered = []
    for _, profile_name, upscaler in rows:
        cleaned = get_profile(profile_name).apply(frame[top:bottom, left:right])
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
