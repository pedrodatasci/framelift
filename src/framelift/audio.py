"""Carry the input's audio over to the enhanced video.

The video is always encoded first, exactly as without audio. Then a quick second
pass copies the matching stretch of the original audio next to it, without touching
the video stream. Doing it afterwards means the exact length is known (even after a
Ctrl+C), and if anything goes wrong with the audio the silent video is still saved.

The stretch is cut in two steps: extract whole audio packets from the start point
on, then shift them by the few milliseconds between the start point and the first
whole packet. The simpler one-step cut keeps up to a second of "pre-roll" audio that
players are told to skip, and tools that ignore that instruction (video editors,
FFmpeg's own concat) end up with the audio out of sync.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

AUDIO_TEMP_SUFFIX = ".audio_tmp.mp4"
SEGMENT_SUFFIX = ".audio_segment.mka"  # Matroska holds any audio codec
FALLBACK_AUDIO_CODEC = ["-c:a", "aac", "-b:a", "192k"]

MP4_FRIENDLY_CODECS = {"aac", "mp3", "ac3", "eac3"}
"""Copied as-is. FFmpeg can put other codecs (like PCM) in an MP4 too, but many
players can't play them there, so those are converted to AAC instead."""


@dataclass(frozen=True)
class AudioInfo:
    """What the input's audio looks like."""

    codecs: list[str]
    video_start: float
    """Timestamp of the first video frame, in seconds (usually 0; not in MPEG-TS files)."""


def probe_audio(path: str | Path) -> AudioInfo | None:
    """Describe the audio tracks of ``path``; ``None`` if there are none.

    Raises ``FileNotFoundError`` if ffprobe (which ships with FFmpeg) is missing.
    """
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe")

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_entries", "format=start_time:stream=codec_type,codec_name,start_time", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        return None

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    audio = [s.get("codec_name", "?") for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return None

    file_start = _seconds(data.get("format", {}).get("start_time"))
    video_starts = [
        _seconds(s.get("start_time")) for s in streams if s.get("codec_type") == "video"
    ]
    video_start = video_starts[0] if video_starts else file_start
    return AudioInfo(codecs=audio, video_start=video_start)


def add_audio(video: Path, source: Path, start_seconds: float, duration_seconds: float) -> str:
    """Add ``source``'s audio, from ``start_seconds`` for ``duration_seconds``, to ``video``.

    ``video`` is replaced in place. Returns what happened:

    * ``"copied"``: the original audio was copied as-is (no quality loss);
    * ``"converted"``: the codec doesn't fit in MP4, so it was converted to AAC;
    * ``"none"``: the input has no audio;
    * ``"failed"``: something went wrong; ``video`` is left silent and untouched.
    """
    try:
        info = probe_audio(source)
    except FileNotFoundError:
        log.warning("ffprobe (part of FFmpeg) wasn't found, so the video is saved without audio.")
        return "failed"

    if info is None:
        log.info("The input has no audio track, so the video is silent.")
        return "none"

    temp = video.with_name(video.stem + AUDIO_TEMP_SUFFIX)
    segment = video.with_name(video.stem + SEGMENT_SUFFIX)
    try:
        delay = _extract_segment(source, segment, info.video_start + start_seconds,
                                 duration_seconds)  # fmt: skip
        if delay is None:
            log.info("The input has no audio for this part of the video, so it's silent.")
            return "none"

        attempts = [("converted", FALLBACK_AUDIO_CODEC)]
        if set(info.codecs) <= MP4_FRIENDLY_CODECS:
            attempts.insert(0, ("copied", ["-c:a", "copy"]))

        for outcome, audio_codec in attempts:
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-itsoffset", f"{delay:.6f}", "-i", str(segment),
                "-map", "0:v:0", "-map", "1:a", "-c:v", "copy", *audio_codec,
                "-movflags", "+faststart", str(temp),
            ]  # fmt: skip
            result = _run(command)
            if result.returncode == 0 and temp.exists() and temp.stat().st_size > 0:
                temp.replace(video)
                return outcome
            temp.unlink(missing_ok=True)
    finally:
        segment.unlink(missing_ok=True)
        temp.unlink(missing_ok=True)

    log.warning(
        "Couldn't add the audio (%s), so the video was saved without it. "
        "Run with -v for FFmpeg's error.",
        ", ".join(info.codecs),
    )
    return "failed"


def _extract_segment(source: Path, segment: Path, start: float, duration: float) -> float | None:
    """Copy the audio packets from ``start`` on into ``segment``.

    Returns how late the first whole packet begins after ``start`` (a few ms),
    which is the delay to apply when muxing; ``None`` if there's no audio there.
    """
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # -copyts keeps the source's clock, so -ss/-t below mean the same instants
        # as the video frames, even in files whose clock doesn't start at 0.
        "-copyts", "-i", str(source), "-vn", "-map", "0:a",
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-c:a", "copy", str(segment),
    ]  # fmt: skip
    if _run(command).returncode != 0 or not segment.exists():
        return None

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=pts_time",
         "-read_intervals", "%+#1", "-of", "csv=p=0", str(segment)],
        capture_output=True, text=True, errors="replace",
    )  # fmt: skip
    first = probe.stdout.strip().splitlines()
    if not first:
        return None
    return max(0.0, _seconds(first[0].strip(",")))


def _run(command: list[str]) -> subprocess.CompletedProcess:
    log.debug("$ %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        log.debug("FFmpeg said:\n%s", result.stderr.strip())
    return result


def _seconds(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
