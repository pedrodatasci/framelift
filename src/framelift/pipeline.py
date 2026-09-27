"""The enhancement pipeline: read → pre-clean → AI upscale → blend → encode.

::

    ┌──────────────┐   queue    ┌─────────────┐        ┌───────┐  pipe   ┌────────┐
    │ reader thread│ ─────────► │ Real-ESRGAN │ ─────► │ blend │ ──────► │ FFmpeg │
    │ decode+clean │ (prefetch) │  (GPU/CPU)  │        │       │  (raw)  │  x264  │
    └──────────────┘            └─────────────┘        └───────┘         └────────┘

The public entry points are :func:`enhance_video` for one-off calls and
:class:`VideoEnhancer` when processing several videos with the same settings
(the model is loaded once and reused).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Union

from tqdm import tqdm

from .device import (
    Device,
    configure_torch,
    current_cpu_threads,
    release_memory,
    resolve_device,
    synchronize,
)
from .encoder import ENCODER_LABELS, FFmpegWriter, require_binary, select_encoder
from .errors import BlackFrameError, InvalidOptionError, NothingToDoError, VideoReadError
from .filters import Frame, get_profile
from .frames import blend_with_classic_upscale, mean_brightness, repeat_count
from .models import ensure_weights, get_model
from .options import EnhanceOptions
from .planning import EnhanceResult, RunPlan, plan_run
from .upscaler import AIUpscaler
from .video import FrameReader, SourceFrame, probe_video

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

BLACK_FRAME_THRESHOLD = 3.0
"""Mean brightness (0–255) below which the first frame is considered broken."""


# -- the enhancer ------------------------------------------------------------------


class VideoEnhancer:
    """Enhance one or more videos with the same settings.

    >>> enhancer = VideoEnhancer(EnhanceOptions(scale=2, ai_strength=0.6))
    >>> enhancer.enhance("a.mp4", "a_hd.mp4")    # doctest: +SKIP
    >>> enhancer.enhance("b.mp4", "b_hd.mp4")    # model is reused  # doctest: +SKIP
    """

    def __init__(self, options: EnhanceOptions | None = None) -> None:
        self.options = options or EnhanceOptions()
        self._upscaler: AIUpscaler | None = None
        self._upscaler_key: tuple | None = None

    def enhance(
        self,
        input_path: PathLike,
        output_path: PathLike,
        *,
        stop_event: threading.Event | None = None,
        show_progress: bool = True,
    ) -> EnhanceResult:
        """Enhance ``input_path`` and write a silent MP4 to ``output_path``.

        Set ``stop_event`` from another thread (or a signal handler) to stop
        early: the current frame is finished and everything processed so far
        is saved as a valid, playable MP4.
        """
        options = self.options
        options.validate()

        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        stop_event = stop_event or threading.Event()

        require_binary("ffmpeg")
        if not input_path.exists():
            raise VideoReadError(f"Input video not found: {input_path}")

        configure_torch(options.torch_threads)
        device = resolve_device(options.device, options.gpu_id)
        encoder = select_encoder(options.encoder, device.kind)

        if options.half and not device.is_cuda:
            log.info("FP16 (--half) only applies to CUDA GPUs, so it's ignored on CPU.")

        video = probe_video(input_path)
        plan = plan_run(video, options)
        _log_plan(plan, options, output_path, device, encoder)

        if not device.is_cuda:
            log.warning(
                "Running the AI on the CPU. It works (Intel Iris laptops included), "
                "but expect it to be much slower than an NVIDIA GPU."
            )

        upscaler = self._get_upscaler(device)
        return self._run(plan, upscaler, device, encoder, output_path, stop_event, show_progress)

    # -- internals -------------------------------------------------------------

    def _get_upscaler(self, device: Device) -> AIUpscaler:
        opts = self.options
        key = (opts.model, str(Path(opts.weights_dir).resolve()), device, opts.tile,
               opts.tile_pad, opts.pre_pad, opts.half)  # fmt: skip

        if self._upscaler is None or self._upscaler_key != key:
            spec = get_model(opts.model)
            weights = ensure_weights(spec, Path(opts.weights_dir).resolve())
            log.info("Loading %s…", spec.name)
            log.debug("  weights: %s", weights)
            self._upscaler = AIUpscaler(
                spec,
                weights,
                device,
                tile=opts.tile,
                tile_pad=opts.tile_pad,
                pre_pad=opts.pre_pad,
                half=opts.half,
            )
            self._upscaler_key = key

        return self._upscaler

    def _run(
        self,
        plan: RunPlan,
        upscaler: AIUpscaler,
        device: Device,
        encoder: str,
        output_path: Path,
        stop_event: threading.Event,
        show_progress: bool,
    ) -> EnhanceResult:
        opts = self.options
        started_at = time.perf_counter()

        if plan.duplicates_frames:
            log.info(
                "Converting %.3f → %.3f fps by repeating frames (duration stays the same).",
                plan.video.fps,
                plan.output_fps,
            )

        writer = FFmpegWriter(
            output_path,
            fps=plan.output_fps,
            encoder=encoder,
            crf=opts.crf,
            x264_preset=opts.x264_preset,
        )
        reader = FrameReader(
            plan.video.path,
            start_frame=plan.start_frame,
            max_frames=opts.max_frames,
            preprocess=get_profile(opts.profile).apply,
            prefetch=opts.prefetch,
            stop_event=stop_event,
        )
        progress = tqdm(
            total=plan.frames_to_process,
            unit="frame",
            desc="Enhancing",
            disable=not show_progress,
        )
        laps = _Laps(sync=lambda: synchronize(device) if opts.debug_timings else None)

        processed = 0
        last_frame = plan.start_frame - 1
        finished = False

        try:
            with reader, progress:
                for frame in reader:
                    laps.lap("queue")

                    if stop_event.is_set() and processed > 0:
                        break

                    output = self._enhance_frame(
                        frame, plan, upscaler, laps, is_first=processed == 0
                    )
                    copies = repeat_count(processed + 1, plan.video.fps, plan.output_fps)
                    writer.write(output, copies=copies)
                    laps.lap("write")

                    processed += 1
                    last_frame = frame.number
                    progress.update(1)

                    if opts.debug_timings:
                        self._diagnostics(processed, frame.number, reader, writer, laps, device)

                    if stop_event.is_set():
                        break

                    laps.restart()

            if processed == 0:
                raise NothingToDoError("No frames were processed, so there's no video to write.")

            interrupted = stop_event.is_set()
            if interrupted:
                log.info("Stopping early — finishing the MP4 with the %d frames done…", processed)

            writer.finish()
            finished = True

        finally:
            if not finished:
                writer.abort()
                if writer.started:
                    log.warning("A partial encode may be left at: %s", writer.temp_path)

        return EnhanceResult(
            output_path=output_path,
            first_frame=plan.start_frame,
            last_frame=last_frame,
            frames_processed=processed,
            frames_written=writer.frames_written,
            total_frames=plan.video.frame_count,
            interrupted=interrupted,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _enhance_frame(
        self,
        frame: SourceFrame,
        plan: RunPlan,
        upscaler: AIUpscaler,
        laps: _Laps,
        *,
        is_first: bool,
    ) -> Frame:
        """AI-upscale one frame and blend it with a classic resize."""
        if is_first:
            _ensure_not_black(frame.image, _BLACK_BEFORE_AI)

        laps.sync()
        enhanced = upscaler.upscale(frame.image, plan.scale)
        laps.sync()
        laps.lap("ai")

        output = blend_with_classic_upscale(
            frame.image, enhanced, plan.width, plan.height, self.options.ai_strength
        )
        laps.lap("blend")

        if is_first:
            _ensure_not_black(output, _BLACK_AFTER_AI)

        return output

    def _diagnostics(
        self,
        processed: int,
        frame_number: int,
        reader: FrameReader,
        writer: FFmpegWriter,
        laps: _Laps,
        device: Device,
    ) -> None:
        """``--debug-timings``: periodic timing report and optional memory cleanup."""
        opts = self.options

        if opts.timing_every > 0 and processed % opts.timing_every == 0:
            log.info(
                "frame %d (#%d this run) | %s | queued=%d | written=%d",
                frame_number,
                processed,
                laps.summary(),
                reader.pending,
                writer.frames_written,
            )

        if opts.gc_every > 0 and processed % opts.gc_every == 0:
            release_memory(device)


def enhance_video(
    input_path: PathLike,
    output_path: PathLike,
    options: EnhanceOptions | None = None,
    *,
    stop_event: threading.Event | None = None,
    show_progress: bool = True,
    **overrides,
) -> EnhanceResult:
    """Enhance a video in one call.

    Options can be passed as an :class:`EnhanceOptions`, as keyword
    arguments, or both (keywords win)::

        enhance_video("in.mp4", "out.mp4", scale=2, profile="old_tv")
    """
    if overrides:
        base = options or EnhanceOptions()
        unknown = set(overrides) - set(EnhanceOptions.field_names())
        if unknown:
            raise InvalidOptionError(f"Unknown option(s): {', '.join(sorted(unknown))}")
        options = EnhanceOptions(**{**vars(base), **overrides})

    return VideoEnhancer(options).enhance(
        input_path, output_path, stop_event=stop_event, show_progress=show_progress
    )


# -- helpers -------------------------------------------------------------------------


_BLACK_BEFORE_AI = (
    "The first frame is almost completely black *before* the AI touches it, "
    "so the problem is in decoding or pre-cleaning, not the model."
)
_BLACK_AFTER_AI = (
    "The first enhanced frame came out black. This usually points to a corrupt "
    "weights file or a GPU/driver problem — try --device cpu to compare."
)


class _Laps:
    """A tiny stopwatch that records how long each named stage took."""

    def __init__(self, sync: Callable[[], None] = lambda: None) -> None:
        self.sync = sync
        self.restart()

    def restart(self) -> None:
        self._started = self._last = time.perf_counter()
        self.durations: dict[str, float] = {}

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.durations[stage] = now - self._last
        self._last = now

    def summary(self) -> str:
        stages = {**self.durations, "total": self._last - self._started}
        return " | ".join(f"{stage}={seconds:.3f}s" for stage, seconds in stages.items())


def _ensure_not_black(frame: Frame, explanation: str) -> None:
    brightness = mean_brightness(frame)
    log.debug("Sanity check: mean brightness %.2f", brightness)
    if brightness < BLACK_FRAME_THRESHOLD:
        raise BlackFrameError(explanation)


def _log_plan(
    plan: RunPlan,
    options: EnhanceOptions,
    output_path: Path,
    device: Device,
    encoder: str,
) -> None:
    video = plan.video
    size_note = "same size" if options.same_resolution else f"{plan.scale:g}x"
    fp16 = "on" if options.half and device.is_cuda else "off"
    encoder_label = ENCODER_LABELS[encoder]
    if encoder == "cpu":
        encoder_label += f", {options.x264_preset}"

    threads = current_cpu_threads()
    rows = [
        ("Input", f"{video.path.name}  ({video.width}x{video.height} @ {video.fps:.3f} fps, "
                  f"{video.frame_count:,} frames)"),
        ("Output", f"{output_path.name}  ({plan.width}x{plan.height} @ {plan.output_fps:.3f} fps, "
                   f"{size_note})"),
        ("Frames", f"{plan.start_frame:,} → {plan.last_frame:,}  ({plan.frames_to_process:,} to "
                   f"enhance, ~{plan.expected_output_frames:,} to write)"),
        ("Look", f"model {options.model} · profile {options.profile} · "
                 f"AI strength {options.clamped_ai_strength:.2f}"),
        ("Device", f"{device.name} · tile {options.tile} · FP16 {fp16} · "
                   f"{threads} CPU thread{'' if threads == 1 else 's'}"),
        ("Encoder", f"{encoder_label} · CRF {options.crf} · prefetch {options.prefetch}"),
    ]  # fmt: skip
    if options.debug_timings:
        rows.append(("Timings", f"every {options.timing_every} frames"))

    width = max(len(label) for label, _ in rows)
    lines = "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)
    log.info("\n%s\n", lines)
