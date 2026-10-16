"""The enhancement pipeline: read → pre-clean → AI upscale → blend → encode.

::

    ┌──────────────┐   queue    ┌─────────────┐        ┌───────┐  pipe   ┌────────┐
    │ reader thread│ ─────────► │ Real-ESRGAN │ ─────► │ blend │ ──────► │ FFmpeg │
    │ decode+clean │ (prefetch) │  (GPU/CPU)  │        │       │  (raw)  │  x264  │
    └──────────────┘            └─────────────┘        └───────┘         └────────┘

Without the AI (``--no-ai``, or an AI strength of 0) no model is loaded and each frame
just gets the classic resize, so the same pipeline also serves to resize, pre-clean or
change the frame rate quickly. With ``--interpolate``, the finished frames go through
RIFE (:mod:`framelift.interpolation`), which draws the extra frames of the higher frame
rate instead of repeating them.

The public entry points are :func:`enhance_video` for one-off calls and
:class:`VideoEnhancer` when processing several videos with the same settings
(the model is loaded once and reused).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Union

import numpy as np
from tqdm import tqdm

from .audio import add_audio
from .device import (
    DEFAULT_CPU_THREADS,
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
from .frames import blend_with_classic_upscale, mean_brightness, repeat_count, resize_exact
from .interpolation import FrameInterpolator, Retimer, ensure_rife_weights
from .models import ensure_weights, get_model, resolve_weights_dir
from .options import EnhanceOptions
from .planning import EnhanceResult, RunPlan, interpolation_advice, plan_run
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

    reuse_repeated_frames = True
    """Reuse the result of a frame identical to the one before it, instead of running the
    AI again (the result would be the same, pixel for pixel)."""

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

        # Without the AI or RIFE there's no device to pick (or fail on) and no model to load.
        needs_device = options.uses_ai or options.interpolate
        device = resolve_device(options.device, options.gpu_id) if needs_device else None
        if device is not None:
            threads = (
                _threads_per_worker(options, device) if options.uses_ai else options.torch_threads
            )
            configure_torch(threads)
        encoder = select_encoder(options.encoder, device.kind if device else None)

        if device is not None and options.half and not device.is_gpu:
            log.info("FP16 (--half) only applies to GPUs, so it's ignored on CPU.")

        video = probe_video(input_path)
        plan = plan_run(video, options)
        _log_plan(plan, options, output_path, device, encoder)

        if device is not None and not device.is_gpu:
            log.warning(
                "Running %s on the CPU. It works (Intel Iris laptops included), "
                "but expect it to be much slower than an NVIDIA GPU.",
                "the AI" if options.uses_ai else "RIFE",
            )

        upscaler = self._get_upscaler(device) if options.uses_ai else None
        return self._run(plan, upscaler, device, encoder, output_path, stop_event, show_progress)

    # -- internals -------------------------------------------------------------

    def _get_upscaler(self, device: Device) -> AIUpscaler:
        opts = self.options
        key = (opts.model, str(resolve_weights_dir(opts.weights_dir)), device, opts.tile,
               opts.tile_pad, opts.pre_pad, opts.half, opts.backend,
               current_cpu_threads())  # fmt: skip

        if self._upscaler is None or self._upscaler_key != key:
            spec = get_model(opts.model)
            weights = ensure_weights(spec, opts.weights_dir)
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
                backend=opts.backend,
            )
            self._upscaler_key = key

        return self._upscaler

    def _run(
        self,
        plan: RunPlan,
        upscaler: AIUpscaler | None,
        device: Device | None,
        encoder: str,
        output_path: Path,
        stop_event: threading.Event,
        show_progress: bool,
    ) -> EnhanceResult:
        opts = self.options
        started_at = time.perf_counter()

        retimer = None
        if opts.interpolate:
            weights = ensure_rife_weights(opts.weights_dir)
            interpolator = FrameInterpolator(weights, device, plan.width, plan.height,
                                             half=opts.half and device.is_gpu,
                                             backend=opts.backend)  # fmt: skip
            retimer = Retimer(plan.video.fps, plan.output_fps, interpolator)
            log.info(
                "Converting %.3f → %.3f fps with in-between frames drawn by RIFE "
                "(duration stays the same).",
                plan.video.fps,
                plan.output_fps,
            )
            advice = interpolation_advice(plan.video.fps, plan.output_fps, plan.frames_to_process)
            if advice:
                log.warning(advice)
        elif plan.duplicates_frames:
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
            desc="Enhancing" if upscaler is not None else "Processing",
            disable=not show_progress,
        )
        laps = _Laps(sync=lambda: synchronize(device) if opts.debug_timings else None)

        processed = 0
        reused = 0
        last_frame = plan.start_frame - 1
        # Identical consecutive frames (stills, duplicated frames) skip the AI: their
        # result is the previous one. Comparing costs a few ms, so only with the AI on.
        check_repeats = upscaler is not None and self.reuse_repeated_frames
        previous_source: Frame | None = None
        last_output: list[Frame | None] = [None]
        finished = False
        # Frames whose AI is running (or done) in the workers, oldest first.
        in_flight: deque[tuple[SourceFrame, Future]] = deque()
        workers = _AIWorkers(upscaler, opts.workers) if upscaler is not None else _NoAI()
        # Frames handed to the AI before the oldest is finished: one per worker. Without
        # the AI there's nothing to wait for, so each frame is finished right away.
        ahead = opts.workers if upscaler is not None else 0
        reader_wait = [0.0]  # time the last frame took to come out of the reader

        def finish_oldest() -> None:
            """Wait for the oldest frame's AI, then blend, encode and count it."""
            nonlocal processed, last_frame, reused
            frame, result = in_flight.popleft()
            laps.restart()
            laps.durations["queue"] = reader_wait[0]
            if result is None:  # identical to the frame before: same result
                output = last_output[0]
                reused += 1
                laps.lap("reused")
            else:
                native = result.result()
                if upscaler is not None:
                    laps.sync()
                    laps.lap("ai")
                output = self._finish_frame(
                    frame, native, plan, upscaler, laps, is_first=processed == 0
                )
            last_output[0] = output
            if retimer is None:
                copies = repeat_count(processed + 1, plan.video.fps, plan.output_fps)
                writer.write(output, copies=copies)
            else:
                ready = retimer.push(output, frame.image)
                laps.sync()
                laps.lap("interpolate")
                for image in ready:
                    writer.write(image)
            laps.lap("write")

            processed += 1
            last_frame = frame.number
            progress.update(1)
            if opts.debug_timings:
                self._diagnostics(processed, frame.number, reader, writer, laps, device)

        try:
            with reader, progress:
                waiting_since = time.perf_counter()
                for submitted, frame in enumerate(reader):
                    reader_wait[0] = time.perf_counter() - waiting_since
                    if stop_event.is_set() and submitted > 0:
                        break

                    if submitted == 0 and upscaler is not None:
                        _ensure_not_black(frame.image, _BLACK_BEFORE_AI)
                    repeat = (
                        check_repeats
                        and previous_source is not None
                        and (np.array_equal(frame.image, previous_source))
                    )
                    in_flight.append((frame, None if repeat else workers.submit(frame.image)))
                    previous_source = frame.image

                    # Keep every worker busy with a frame while this thread finishes the
                    # oldest one (resize, blend, encode): the two overlap.
                    if len(in_flight) > ahead:
                        finish_oldest()

                    if stop_event.is_set():
                        break
                    waiting_since = time.perf_counter()

                # Frames already in the AI are finished too, not thrown away.
                while in_flight:
                    finish_oldest()

            if processed == 0:
                raise NothingToDoError("No frames were processed, so there's no video to write.")

            interrupted = stop_event.is_set()
            if interrupted:
                log.info("Stopping early — finishing the MP4 with the %d frames done…", processed)

            if retimer is not None:
                last, copies = retimer.flush()
                writer.write(last, copies=copies)
                instead = [f"{retimer.scene_cuts} at scene cuts"] if retimer.scene_cuts else []
                if retimer.unchanged:
                    instead.append(f"{retimer.unchanged} between identical frames")
                log.info(
                    "RIFE drew %d in-between frames%s.",
                    retimer.interpolated,
                    f" (repeated instead: {', '.join(instead)})" if instead else "",
                )
            if reused:
                log.info(
                    "%d frames were identical to the one before; the AI's result was reused.",
                    reused,
                )
            writer.finish()
            audio = self._add_audio(plan, output_path, processed) if opts.audio else "disabled"
            finished = True

        finally:
            workers.close(wait=finished)
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
            audio=audio,
            elapsed_seconds=time.perf_counter() - started_at,
            frames_interpolated=retimer.interpolated if retimer is not None else 0,
            frames_reused=reused,
        )

    def _add_audio(self, plan: RunPlan, output_path: Path, processed: int) -> str:
        """Copy the stretch of the input's audio that matches the enhanced frames."""
        fps = plan.video.fps
        outcome = add_audio(
            output_path,
            plan.video.path,
            start_seconds=(plan.start_frame - 1) / fps,
            duration_seconds=processed / fps,
        )
        if outcome == "copied":
            log.info("Kept the original audio.")
        elif outcome == "converted":
            log.info("Kept the audio, converted to AAC (its codec doesn't fit in an MP4).")
        return outcome

    def _finish_frame(
        self,
        frame: SourceFrame,
        native: Frame | None,
        plan: RunPlan,
        upscaler: AIUpscaler | None,
        laps: _Laps,
        *,
        is_first: bool,
    ) -> Frame:
        """Bring the AI's output (at the model's own scale) to size and blend it with a
        classic resize. Without the AI, the classic resize alone (what a blend with an AI
        strength of 0 gives, pixel for pixel)."""
        if upscaler is None:
            output = resize_exact(frame.image, plan.width, plan.height)
            laps.lap("resize")
            return output

        enhanced = upscaler.finish(native, frame.image.shape, plan.scale)
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
        device: Device | None,
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


class _NoAI:
    """Stands in for :class:`_AIWorkers` when the AI is off: every frame is "done" at once."""

    def submit(self, image: Frame) -> Future:
        result: Future = Future()
        result.set_result(None)
        return result

    def close(self, wait: bool = True) -> None:
        pass


class _AIWorkers:
    """Background threads that run the AI, each on its own frame.

    With one worker, the AI of the next frame runs while the main thread finishes the
    previous one. With more, several frames go through the AI at once, each with its
    share of the CPU threads. The threads are daemons: a hard quit doesn't wait for them.
    """

    def __init__(self, upscaler: AIUpscaler, count: int) -> None:
        self._upscaler = upscaler
        self._jobs: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._work, name=f"framelift-ai-{n}", daemon=True)
            for n in range(count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, image: Frame) -> Future:
        result: Future = Future()
        self._jobs.put((result, image))
        return result

    def close(self, wait: bool = True) -> None:
        """Stop the threads. ``wait``: let them exit cleanly first (they're idle by then),
        so none is still inside PyTorch while Python shuts down."""
        for _ in self._threads:
            self._jobs.put(None)
        if wait:
            for thread in self._threads:
                thread.join(timeout=10)

    def _work(self) -> None:
        while (job := self._jobs.get()) is not None:
            result, image = job
            if not result.set_running_or_notify_cancel():
                continue
            try:
                result.set_result(self._upscaler.infer(image))
            except BaseException as exc:  # handed to the main thread, which re-raises it
                result.set_exception(exc)


def _threads_per_worker(options: EnhanceOptions, device: Device) -> int:
    """CPU threads for each AI worker: yours, or PyTorch's default split between them."""
    if options.torch_threads or options.workers == 1 or device.is_cuda:
        return options.torch_threads
    return max(1, DEFAULT_CPU_THREADS // options.workers)


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
    device: Device | None,
    encoder: str,
) -> None:
    video = plan.video
    size_note = "same size" if options.same_resolution else f"{plan.scale:g}x"
    encoder_label = ENCODER_LABELS[encoder]
    if encoder == "cpu":
        encoder_label += f", {options.x264_preset}"

    verb = "enhance" if options.uses_ai else "process"
    if options.uses_ai:
        look = (f"model {options.model} · profile {options.profile} · "
                f"AI strength {options.clamped_ai_strength:.2f}")  # fmt: skip
    else:
        resized = (plan.width, plan.height) != (video.width, video.height)
        look = f"no AI{' (Lanczos resize)' if resized else ''} · profile {options.profile}"

    rows = [
        ("Input", f"{video.path.name}  ({video.width}x{video.height} @ {video.fps:.3f} fps, "
                  f"{video.frame_count:,} frames)"),
        ("Output", f"{output_path.name}  ({plan.width}x{plan.height} @ {plan.output_fps:.3f} fps"
                   f"{' with RIFE' if options.interpolate else ''}, {size_note})"),
        ("Frames", f"{plan.start_frame:,} → {plan.last_frame:,}  ({plan.frames_to_process:,} to "
                   f"{verb}, ~{plan.expected_output_frames:,} to write)"),
        ("Look", look),
    ]  # fmt: skip
    if device is not None:
        fp16 = "on" if options.half and device.is_gpu else "off"
        if options.uses_ai:
            workers = _workers_label(options, device, current_cpu_threads())
            details = f"tile {options.tile} · FP16 {fp16} · {workers}"
        else:
            details = f"for RIFE · FP16 {fp16}"
        rows.append(("Device", f"{device.name} · {details}"))
    rows.append(("Encoder", f"{encoder_label} · CRF {options.crf} · prefetch {options.prefetch}"))
    if options.debug_timings:
        rows.append(("Timings", f"every {options.timing_every} frames"))

    width = max(len(label) for label, _ in rows)
    lines = "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)
    log.info("\n%s\n", lines)


def _workers_label(options: EnhanceOptions, device: Device, threads: int) -> str:
    cpu = f"{threads} CPU thread{'' if threads == 1 else 's'}"
    backend = " · OpenVINO" if options.backend == "openvino" and device.kind == "cpu" else ""
    if options.workers == 1:
        return cpu + backend
    return f"{options.workers} workers × {cpu}{backend}"
