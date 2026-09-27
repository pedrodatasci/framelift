"""Writing video: streams raw frames straight into FFmpeg through a pipe.

No temporary PNG/JPG frames on disk — each enhanced frame goes from memory
into the encoder. The MP4 is written to a temporary name and only renamed to
the final path once FFmpeg has finished cleanly, so a crash never leaves a
half-written file where you expect a good one.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .errors import EncodingError, MissingDependencyError
from .filters import Frame

log = logging.getLogger(__name__)

MIN_VALID_OUTPUT_BYTES = 1024
TEMP_SUFFIX = ".encoding_tmp.mp4"
ENCODER_LABELS = {"cpu": "libx264 (CPU)", "nvenc": "h264_nvenc (NVIDIA)"}


# -- environment checks --------------------------------------------------------


def require_binary(name: str) -> None:
    """Fail early, with a clear message, if ``name`` isn't on the PATH."""
    if shutil.which(name) is None:
        raise MissingDependencyError(
            f"Couldn't find '{name}' on your PATH. "
            f"Install it (https://ffmpeg.org/download.html) and try again."
        )


def format_command(command: Sequence[object]) -> str:
    """Render a command for display, quoting arguments that contain spaces."""
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def nvenc_available() -> bool:
    """Try encoding one tiny frame with NVENC to see if it really works here."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=1",
        "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
    ]  # fmt: skip
    log.debug("$ %s", format_command(command))

    probe = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if probe.returncode == 0:
        return True

    log.warning("NVENC isn't usable with this FFmpeg build/driver:\n%s", probe.stdout.strip())
    return False


def select_encoder(requested: str, device_kind: str) -> str:
    """Pick the encoder that will actually be used, falling back to CPU when needed."""
    if requested not in ENCODER_LABELS:
        raise ValueError(f"encoder must be one of {', '.join(ENCODER_LABELS)}")

    if requested == "cpu":
        return "cpu"

    if device_kind == "cpu":
        log.info("NVENC needs an NVIDIA GPU and we're running on CPU, so encoding with libx264.")
        return "cpu"

    if nvenc_available():
        return "nvenc"

    log.info("Falling back to libx264 (CPU) for encoding.")
    return "cpu"


def codec_arguments(encoder: str, crf: int, x264_preset: str) -> list[str]:
    """FFmpeg arguments for the chosen encoder and quality level."""
    if encoder == "cpu":
        return ["-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf)]

    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf)]

    raise ValueError(f"encoder must be one of {', '.join(ENCODER_LABELS)}")


def temp_path_for(destination: str | Path) -> Path:
    """Where the MP4 is written while encoding is still in progress."""
    destination = Path(destination)
    return destination.with_name(destination.stem + TEMP_SUFFIX)


# -- the writer ------------------------------------------------------------------


class FFmpegWriter:
    """Feeds BGR frames into an FFmpeg process and produces a silent MP4.

    FFmpeg is started lazily on the first :meth:`write`, because that's when
    we know the exact frame size.
    """

    def __init__(
        self,
        destination: str | Path,
        *,
        fps: float,
        encoder: str = "cpu",
        crf: int = 20,
        x264_preset: str = "veryfast",
    ) -> None:
        self.destination = Path(destination)
        self.temp_path = temp_path_for(self.destination)
        self.fps = fps
        self.codec_args = codec_arguments(encoder, crf, x264_preset)
        self.frames_written = 0
        self._process: subprocess.Popen | None = None

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.unlink(missing_ok=True)

    @property
    def started(self) -> bool:
        return self._process is not None

    def write(self, frame: Frame, copies: int = 1) -> None:
        """Send ``frame`` to the encoder ``copies`` times (0 is allowed)."""
        if self._process is None:
            height, width = frame.shape[:2]
            self._process = self._start(width, height)

        payload = frame.tobytes()  # always C-ordered raw bytes, computed once

        try:
            for _ in range(copies):
                self._process.stdin.write(payload)
                self.frames_written += 1
        except BrokenPipeError as exc:
            raise EncodingError(
                "FFmpeg stopped accepting frames (broken pipe). "
                "Check the messages above for FFmpeg's own error."
            ) from exc

    def finish(self) -> Path:
        """Close the stream, wait for FFmpeg, and move the MP4 into place."""
        if self._process is None:
            raise EncodingError("No frames were written, so there's no video to finish.")

        self._close_stdin()
        return_code = self._process.wait()

        if return_code != 0:
            raise EncodingError(f"FFmpeg failed while finishing the MP4 (exit code {return_code}).")

        if not _looks_like_a_real_file(self.temp_path):
            raise EncodingError(
                "FFmpeg finished, but the output is missing or suspiciously small: "
                f"{self.temp_path}"
            )

        self.destination.unlink(missing_ok=True)
        shutil.move(str(self.temp_path), str(self.destination))
        return self.destination

    def abort(self) -> None:
        """Stop feeding FFmpeg after an error.

        We only close its input: FFmpeg then wraps up on its own and leaves a
        playable partial file at :attr:`temp_path`, which can still be useful.
        """
        if self._process is not None:
            self._close_stdin()

    # -- internals -------------------------------------------------------------

    def _start(self, width: int, height: int) -> subprocess.Popen:
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            # Input: raw BGR frames on stdin, exactly as OpenCV holds them.
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "pipe:0",
            # Output: no audio; even dimensions + yuv420p for maximum player support.
            "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            *self.codec_args,
            "-movflags", "+faststart",
            str(self.temp_path),
        ]  # fmt: skip
        log.debug("Starting FFmpeg:\n$ %s", format_command(command))

        # FFmpeg runs in its own process group/session so that Ctrl+C in the
        # terminal reaches *us* but not FFmpeg. That's what lets a first Ctrl+C
        # finish the current frame and still close the MP4 properly.
        isolation = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )

        return subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=10**8, **isolation)

    def _close_stdin(self) -> None:
        if self._process is not None and self._process.stdin:
            with contextlib.suppress(BrokenPipeError):
                self._process.stdin.close()


def _looks_like_a_real_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_VALID_OUTPUT_BYTES
