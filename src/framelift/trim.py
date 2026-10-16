"""Cutting out the part of a video to enhance.

The cut is saved as a **new file** (the original is never touched), and that file
becomes the input of the run. Everything that counts frames — ``--start-frame`` to
resume, ``--max-frames``, the audio — then refers to the cut copy, so stopping halfway
and resuming works exactly as with any other input.

Frames are numbered as everywhere else in framelift: from 1, as OpenCV decodes them.
The copy is cut on the exact frame and losslessly, so decoding it gives the very same
pixels as the original:

* video: H.264 in lossless mode (``-qp 0``). A plain stream copy would be cheaper, but
  it can only start on a keyframe, which can be seconds away from the chosen frame;
* audio: ALAC (lossless), so it can be cut on the exact sample. It's converted to AAC
  when the enhanced video is saved, since ALAC isn't widely playable in an MP4.

Lossless video is big (roughly 10–30x the original), which is fine for a working copy.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .encoder import find_binary, format_command, require_binary, temp_path_for
from .errors import EncodingError, InvalidOptionError, VideoReadError
from .video import VideoInfo, probe_video

log = logging.getLogger(__name__)

SEEK_MARGIN_SECONDS = 0.001
"""Seek this much before the first frame, so rounding (some containers store times in
whole milliseconds) can never skip it. The extra audio is trimmed off again."""


# -- ranges -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRange:
    """A stretch of a video, in frames counted from 1. Both ends are included."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise InvalidOptionError("A range starts at frame 1 or later (frames count from 1).")
        if self.end < self.start:
            raise InvalidOptionError(
                f"The range ends (frame {self.end}) before it starts (frame {self.start})."
            )

    @property
    def frames(self) -> int:
        return self.end - self.start + 1

    def start_seconds(self, fps: float) -> float:
        """When the first frame appears."""
        return (self.start - 1) / fps

    def end_seconds(self, fps: float) -> float:
        """When the last frame stops being shown."""
        return self.end / fps

    def describe(self, fps: float) -> str:
        """e.g. ``0:10.000 → 1:25.500 (75.5 s, frames 301–2,565)``."""
        start, end = self.start_seconds(fps), self.end_seconds(fps)
        return (
            f"{format_time(start)} → {format_time(end)} "
            f"({end - start:.1f} s, frames {self.start:,}–{self.end:,})"
        )


_TIME = re.compile(r"^(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d*)?)$")


def parse_time(text: str) -> float:
    """``"90"``, ``"90.5"``, ``"1:30"`` or ``"1:02:03.25"`` → seconds."""
    match = _TIME.match(text.strip())
    if not match:
        raise InvalidOptionError(
            f"Couldn't read the time '{text}'. "
            "Use seconds (90.5), M:SS (1:30) or H:MM:SS (1:02:03)."
        )
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)


def format_time(seconds: float) -> str:
    """``83.5`` → ``"1:23.500"``; ``3723`` → ``"1:02:03.000"``."""
    milliseconds = int(round(max(0.0, seconds) * 1000))
    minutes, milliseconds = divmod(milliseconds, 60_000)
    hours, minutes = divmod(minutes, 60)
    clock = f"{minutes}:{milliseconds / 1000:06.3f}"
    return f"{hours}:{minutes:02d}:{milliseconds / 1000:06.3f}" if hours else clock


def range_from_times(video: VideoInfo, start: float | None, end: float | None) -> FrameRange:
    """The frames shown between ``start`` and ``end`` seconds (``None`` = that end of the video)."""
    first = 1 if start is None else math.floor(start * video.fps + 1e-6) + 1
    last = video.frame_count if end is None else math.ceil(end * video.fps - 1e-6)
    last = min(last, video.frame_count)

    if first > video.frame_count:
        raise InvalidOptionError(
            f"The start ({format_time(start or 0)}) is past the end of the video "
            f"({format_time(video.duration_seconds)})."
        )
    if last < first:
        raise InvalidOptionError("The end has to come after the start.")
    return FrameRange(first, last)


def default_trim_path(input_path: str | Path, frame_range: FrameRange) -> Path:
    """``clip.mkv`` → ``clip_trim_301-2565.mp4``, next to the original."""
    path = Path(input_path)
    return path.with_name(f"{path.stem}_trim_{frame_range.start}-{frame_range.end}.mp4")


# -- cutting ------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimResult:
    """What :func:`trim_video` produced."""

    path: Path
    frame_range: FrameRange
    """The range, in frames of the *original* video."""

    frames: int
    """Frames in the cut copy (normally ``frame_range.frames``)."""

    audio: bool
    """Whether the copy has audio."""


def trim_video(
    input_path: str | Path,
    output_path: str | Path,
    frame_range: FrameRange,
    *,
    show_progress: bool = True,
) -> TrimResult:
    """Save ``frame_range`` of ``input_path`` as a new, losslessly cut MP4.

    The file is written under a temporary name and only renamed once FFmpeg finishes,
    so an interrupted cut never leaves a half-written file behind.
    """
    require_binary("ffmpeg")
    require_binary("ffprobe")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if not input_path.exists():
        raise VideoReadError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise InvalidOptionError("The cut copy needs a different name than the original.")

    video = probe_video(input_path)
    if frame_range.start > video.frame_count:
        raise InvalidOptionError(
            f"The range starts at frame {frame_range.start}, but the video has "
            f"{video.frame_count} frames."
        )
    frame_range = FrameRange(frame_range.start, min(frame_range.end, video.frame_count))

    streams = _probe_streams(input_path)
    # Frame N is shown (N - 1) / fps after the first video frame. FFmpeg counts time from
    # the start of the file, which can be a little earlier (e.g. in MPEG-TS).
    offset = streams["video_start"] - streams["file_start"] + frame_range.start_seconds(video.fps)
    expected = frame_range.frames

    temp = temp_path_for(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Cutting %s: %s", input_path.name, frame_range.describe(video.fps))

    frames = 0
    try:
        # Seeking is fast but needs a seekable file with regular keyframes. When it
        # doesn't give the exact frames (e.g. some MPEG-TS captures), decode from the
        # start instead: slower, but always exact.
        for mode in ("seek", "decode"):
            frames = _cut(mode, input_path, temp, offset, frame_range, video.fps,
                          streams["audio"], show_progress)  # fmt: skip
            if frames == expected:
                break
            log.debug("Cutting by %s gave %d frames instead of %d.", mode, frames, expected)
        if frames == 0:
            raise EncodingError("FFmpeg couldn't cut the video. Run again with -v to see why.")

        output_path.unlink(missing_ok=True)
        shutil.move(str(temp), str(output_path))
    finally:
        temp.unlink(missing_ok=True)

    if frames != expected:
        log.warning(
            "The cut copy has %d frames instead of %d (the original's frame count may be "
            "approximate).",
            frames,
            expected,
        )
    return TrimResult(output_path, frame_range, frames, streams["audio"])


def _cut(mode, source, target, offset, frame_range, fps, audio, show_progress) -> int:
    """Run one cutting strategy; returns the frames in ``target`` (0 if it failed)."""
    for pixel_format in (None, "yuv444p"):  # a few sources (e.g. RGB) need the fallback
        command = _cut_command(mode, source, target, offset, frame_range, fps, pixel_format,
                               audio)  # fmt: skip
        if _run_with_progress(command, frame_range.frames, show_progress):
            try:
                return probe_video(target).frame_count
            except VideoReadError:
                return 0
        log.debug("Lossless H.264 may not keep this pixel format; retrying as yuv444p.")
    return 0


def _cut_command(mode, source, target, offset, frame_range, fps, pixel_format, audio):
    frames = frame_range.frames
    command = [find_binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostats",
               "-progress", "pipe:1"]  # fmt: skip

    if mode == "seek":
        # Land a hair before the first frame, so float rounding can never skip it.
        margin = max(0.0, min(SEEK_MARGIN_SECONDS, 0.25 / fps, offset))
        command += [
            "-ss", f"{offset - margin:.6f}", "-i", str(source),
            # trim (rather than -frames:v, which would also cut the audio short) stops the
            # video after exactly `frames` frames; -t ends the audio with the last frame.
            "-map", "0:v:0", "-filter:v", f"trim=end_frame={frames}",
            "-t", f"{frames / fps + margin:.6f}",
        ]  # fmt: skip
        # The video starts at its first frame; drop the margin from the audio to match.
        audio_filter = f"atrim=start={margin:.6f},asetpts=PTS-STARTPTS"
    else:
        first = frame_range.start - 1  # frames counted from 0, as decoded from the start
        command += [
            "-i", str(source), "-map", "0:v:0",
            "-filter:v", f"trim=start_frame={first}:end_frame={first + frames},"
                         "setpts=PTS-STARTPTS",
        ]  # fmt: skip
        audio_filter = f"atrim=start={offset:.6f}:duration={frames / fps:.6f},asetpts=PTS-STARTPTS"

    # Keep every decoded frame as it is: no duplicates or drops to force a constant rate.
    command += [*_passthrough(), "-c:v", "libx264", "-preset", "veryfast", "-qp", "0"]
    if pixel_format:
        command += ["-pix_fmt", pixel_format]
    if audio:
        command += ["-map", "0:a", "-c:a", "alac"]
        command += ["-filter:a", audio_filter]
    return [*command, "-movflags", "+faststart", str(target)]


@functools.cache
def _passthrough() -> tuple[str, ...]:
    """``-fps_mode passthrough``, spelled ``-vsync passthrough`` before FFmpeg 5.1."""
    help_text = subprocess.run(
        [find_binary("ffmpeg"), "-hide_banner", "-h", "full"],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    return ("-fps_mode", "passthrough") if "-fps_mode" in help_text else ("-vsync", "passthrough")


def _run_with_progress(command: list[str], frames: int, show_progress: bool) -> bool:
    log.debug("$ %s", format_command(command))
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    with tqdm(total=frames, unit="frame", desc="Cutting", disable=not show_progress) as bar:
        for line in process.stdout:
            if line.startswith("frame="):
                done = int(line.split("=", 1)[1].strip() or 0)
                bar.update(max(0, min(done, frames) - bar.n))
    error = process.stderr.read()
    if process.wait() != 0:
        log.debug("FFmpeg said:\n%s", error.strip())
        return False
    return True


def _probe_streams(path: Path) -> dict:
    """When the file and its first video stream start, and whether there's audio."""
    result = subprocess.run(
        [find_binary("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_entries", "format=start_time:stream=codec_type,start_time", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        raise VideoReadError(f"ffprobe couldn't read {path}: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    file_start = _seconds(data.get("format", {}).get("start_time"))
    videos = [_seconds(s.get("start_time")) for s in streams if s.get("codec_type") == "video"]
    return {
        "file_start": file_start,
        "video_start": videos[0] if videos else file_start,
        "audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _seconds(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
