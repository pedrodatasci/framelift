"""``framelift tune``: find good settings for *this* video on *this* machine.

Two kinds of settings, handled differently:

* **Speed settings** (device, FP16, tile, encoder) have an objectively best
  answer, so we measure it: a few real frames of the video are upscaled with
  each candidate configuration and the fastest one that fits in memory wins.
* **Look settings** (profile, scale, AI strength) are a matter of taste, so we
  only *suggest*: noise and resolution heuristics give a starting point, and a
  comparison sheet shows the same crop with different settings side by side.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2

from .analysis import (
    SizeSuggestion,
    build_comparison_sheet,
    describe_noise,
    detail_score,
    estimate_noise,
    most_detailed_region,
    sample_positions,
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
from .frames import blend_with_classic_upscale, mean_brightness
from .models import ensure_weights, get_model
from .options import EnhanceOptions
from .planning import RunPlan, plan_run
from .upscaler import AIUpscaler
from .video import VideoInfo, probe_video, read_frames

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

GPU_TILES = (0, 1024, 768, 512, 256)
CPU_TILES = (256, 128)
SHEET_STRENGTHS = (0.0, 0.3, 0.45, 0.7, 1.0)
SHEET_TILE_SIZE = (400, 300)  # width, height of each crop, in output pixels
_CROP_MARGIN = 12  # source pixels of context around each crop, trimmed afterwards


@dataclass(frozen=True)
class BenchmarkRun:
    """One configuration tried during the benchmark."""

    tile: int
    half: bool
    seconds_per_frame: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.seconds_per_frame is not None


@dataclass(frozen=True)
class TuneReport:
    """Everything ``tune_video`` found out, plus the settings it recommends."""

    video: VideoInfo
    device: Device
    gpu_memory_gb: float | None
    nvenc: bool
    noise: float
    noise_label: str
    suggested_profile: str
    size: SizeSuggestion
    plan: RunPlan
    runs: list[BenchmarkRun]
    preprocess_seconds: float
    recommended: EnhanceOptions
    sample_frame: int
    sheet_path: Path | None = None

    @property
    def best_run(self) -> BenchmarkRun | None:
        successful = [run for run in self.runs if run.ok]
        return min(successful, key=lambda run: run.seconds_per_frame) if successful else None

    @property
    def seconds_per_frame(self) -> float | None:
        """Expected time per input frame with the recommended settings.

        The reader thread pre-cleans frames in parallel with the AI, so the
        slower of the two sets the pace.
        """
        best = self.best_run
        if best is None:
            return None
        return max(best.seconds_per_frame, self.preprocess_seconds)

    @property
    def estimated_seconds(self) -> float | None:
        """Rough total processing time (encoding not included)."""
        per_frame = self.seconds_per_frame
        return None if per_frame is None else per_frame * self.plan.frames_to_process


def tune_video(
    input_path: PathLike,
    options: EnhanceOptions | None = None,
    *,
    keep: Iterable[str] = (),
    samples: int = 3,
    benchmark: bool = True,
    sheet_path: PathLike | None = None,
) -> TuneReport:
    """Analyse a video and recommend settings for enhancing it.

    ``options`` is the starting point. Option names listed in ``keep`` are
    treated as your decision and never overridden (e.g. ``keep={"profile"}``).
    Pass ``sheet_path`` to also render a comparison sheet PNG there.
    """
    options = options or EnhanceOptions()
    options.validate()
    keep = set(keep)
    unknown = keep - set(EnhanceOptions.field_names())
    if unknown:
        raise ValueError(f"Unknown option(s) in keep: {', '.join(sorted(unknown))}")

    require_binary("ffmpeg")
    video = probe_video(input_path)

    # 1. Look at the footage -----------------------------------------------------
    positions = sample_positions(video.frame_count, samples) or [1]
    frames = read_frames(video.path, positions)
    noise = statistics.median(estimate_noise(image) for _, image in frames)
    noise_label, suggested_profile = describe_noise(noise)
    size = suggest_size(video.width, video.height)
    sample_number, sample = max(frames, key=lambda item: detail_score(item[1]))

    log.info(
        "Checked %d frame%s: %s (noise %.1f), %dp.",
        len(frames), "" if len(frames) == 1 else "s", noise_label, noise, video.height,
    )  # fmt: skip

    looks = {}
    if "profile" not in keep:
        looks["profile"] = suggested_profile
    if not keep & {"scale", "same_resolution"}:
        looks.update(same_resolution=size.same_resolution, scale=size.scale)
    looked = dataclasses.replace(options, **looks)
    plan = plan_run(video, looked)

    # 2. Look at the machine ----------------------------------------------------------
    configure_torch(options.torch_threads)
    device = resolve_device(options.device, options.gpu_id)
    memory = gpu_memory_gb(device)
    nvenc = device.is_cuda and nvenc_available()

    spec = get_model(options.model)
    weights = ensure_weights(spec, Path(options.weights_dir).resolve())
    profile = get_profile(looked.profile)
    cleaned = profile.apply(sample)
    preprocess_seconds = _time_it(lambda: profile.apply(sample))

    # 3. Measure speed ---------------------------------------------------------------
    runs: list[BenchmarkRun] = []
    if benchmark:
        log.info("Benchmarking %s on %s (a real %dx%d frame)…", spec.name, device.name,
                 video.width, video.height)  # fmt: skip
        runs = _benchmark(spec, weights, device, cleaned, plan.scale, options, keep)

    speed = _speed_settings(options, keep, device, runs, nvenc)
    recommended = dataclasses.replace(looked, **speed)

    # 4. Show the looks ---------------------------------------------------------------
    written_sheet = None
    if sheet_path is not None:
        upscaler = AIUpscaler(spec, weights, device, tile=0, half=recommended.half)
        rows = _sheet_profiles(looked.profile, keep_profile="profile" in keep)
        image = _render_sheet(video, sample_number, sample, rows, upscaler, plan)
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
        size=size,
        plan=plan,
        runs=runs,
        preprocess_seconds=preprocess_seconds,
        recommended=recommended,
        sample_frame=sample_number,
        sheet_path=written_sheet,
    )


# -- benchmark ---------------------------------------------------------------------------


def _candidates(device: Device, frame: Frame, options: EnhanceOptions, keep: set[str]):
    """Which (tile, half) pairs are worth trying on this device."""
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


def _measure(spec, weights, device, frame, scale, tile, half, repeats) -> BenchmarkRun:
    label = f"tile {tile or 'off'}{', FP16' if half else ''}"
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
        log.info("  %-18s out of memory", label)
        return BenchmarkRun(tile, half, error="out of memory")
    finally:
        release_memory(device)

    if mean_brightness(last_output[0]) < 3:
        log.info("  %-18s produced a black frame", label)
        return BenchmarkRun(tile, half, error="black output")

    log.info("  %-18s %.2f s/frame", label, seconds)
    return BenchmarkRun(tile, half, seconds_per_frame=seconds)


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


def _speed_settings(options, keep, device, runs, nvenc) -> dict:
    settings = {}
    if "device" not in keep:
        settings["device"] = device.kind

    best = min((run for run in runs if run.ok), key=lambda run: run.seconds_per_frame, default=None)
    if best is not None:
        if "tile" not in keep:
            settings["tile"] = best.tile
        if "half" not in keep:
            settings["half"] = best.half

    if "encoder" not in keep:
        settings["encoder"] = "nvenc" if nvenc else "cpu"

    # When the AI is slow, libx264 has time to spare: "medium" makes smaller
    # files at the same quality for practically free.
    if best is not None and "x264_preset" not in keep and best.seconds_per_frame >= 0.25:
        settings["x264_preset"] = "medium"

    return settings


# -- comparison sheet ------------------------------------------------------------------


def _sheet_profiles(profile: str, keep_profile: bool) -> list[str]:
    if keep_profile:
        return [profile]
    # Always compare against the untouched frame ("none") first.
    return ["none", "minimal" if profile == "none" else profile]


def _render_sheet(
    video: VideoInfo,
    frame_number: int,
    frame: Frame,
    profiles: Sequence[str],
    upscaler: AIUpscaler,
    plan: RunPlan,
) -> Frame:
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

    crops = {}
    for name in profiles:
        cleaned = get_profile(name).apply(frame[top:bottom, left:right])
        crops[name] = (cleaned, upscaler.upscale(cleaned, scale))

    def render(row: int, col: int) -> Frame:
        cleaned, enhanced = crops[profiles[row]]
        mixed = blend_with_classic_upscale(
            cleaned, enhanced, out_width, out_height, SHEET_STRENGTHS[col]
        )
        tile = mixed[offset_y : offset_y + tile_height, offset_x : offset_x + tile_width]
        # Rounding can leave a crop a pixel short; pad so every tile matches.
        return cv2.copyMakeBorder(
            tile, 0, tile_height - tile.shape[0], 0, tile_width - tile.shape[1],
            cv2.BORDER_REPLICATE,
        )  # fmt: skip

    columns = [_strength_label(value) for value in SHEET_STRENGTHS]
    name = video.path.name.encode("ascii", "replace").decode()
    title = f"{name} - frame {frame_number} - 100% crop of the {plan.width}x{plan.height} output"
    return build_comparison_sheet(profiles, columns, render, title)


def _strength_label(value: float) -> str:
    if value == 0:
        return "no AI (plain resize)"
    label = f"--ai-strength {value:g}"
    return label + "  (default)" if value == 0.45 else label


def _write_png(path: Path, image: Frame) -> Path:
    # cv2.imwrite can't handle non-ASCII paths on Windows; encoding in memory can.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Couldn't encode the comparison sheet as PNG.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path.resolve()
