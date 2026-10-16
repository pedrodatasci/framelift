"""Putting the parts of a stopped and resumed run back together: ``framelift join``.

A run stopped with Ctrl+C and continued with ``--start-frame`` leaves two or more MP4s,
each with its own stretch of the original audio, cut on the exact frame. :func:`join_videos`
puts them one after the other without re-encoding, so it takes seconds and loses nothing,
after checking that they were encoded the same way: joining files that weren't would give
a video that plays wrong, or not at all.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .encoder import find_binary, temp_path_for
from .errors import EncodingError, InvalidOptionError, VideoReadError

# What has to be the same in every part for them to be joined without re-encoding.
_VIDEO_FIELDS = {
    "codec_name": "video codec",
    "width": "width",
    "height": "height",
    "pix_fmt": "pixel format",
    "r_frame_rate": "frame rate",
}
_AUDIO_FIELDS = {"codec_name": "audio codec", "sample_rate": "sample rate", "channels": "channels"}


@dataclass(frozen=True)
class JoinResult:
    """What :func:`join_videos` made."""

    path: Path
    parts: int
    duration_seconds: float
    has_audio: bool


def join_videos(parts: Sequence[str | Path], output: str | Path) -> JoinResult:
    """Join ``parts``, in this order, into ``output``, without re-encoding.

    Raises :class:`InvalidOptionError` if there are fewer than two parts, one is missing,
    ``output`` is one of them, or they weren't encoded the same way (the message says
    what differs).
    """
    parts = [Path(part) for part in parts]
    output = Path(output)
    if len(parts) < 2:
        raise InvalidOptionError("Joining needs at least two parts.")
    for part in parts:
        if not part.is_file():
            raise InvalidOptionError(f"Can't find {part}.")
    if any(output.resolve() == part.resolve() for part in parts):
        raise InvalidOptionError(
            f"{output} is one of the parts. Write the joined video to a new file."
        )

    layouts = [_layout(part) for part in parts]
    for part, layout in zip(parts[1:], layouts[1:]):
        differences = _differences(layouts[0], layout)
        if differences:
            raise InvalidOptionError(
                f"{part.name} doesn't match {parts[0].name} ({'; '.join(differences)}), so "
                "they can't be joined without re-encoding. Parts must be made with the same "
                "options: size, frame rate, encoder and audio."
            )

    temp = temp_path_for(output)
    with tempfile.TemporaryDirectory(prefix="framelift-join-") as folder:
        listing = Path(folder) / "parts.txt"
        # FFmpeg reads this list as UTF-8, whatever the system's encoding.
        listing.write_text("".join(f"file '{_quoted(part)}'\n" for part in parts), encoding="utf-8")
        command = [
            find_binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", "-movflags", "+faststart", str(temp),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace")  # fmt: skip
    if result.returncode != 0:
        temp.unlink(missing_ok=True)
        raise EncodingError(f"FFmpeg couldn't join the parts: {result.stderr.strip()}")

    output.unlink(missing_ok=True)
    shutil.move(str(temp), str(output))
    return JoinResult(output, len(parts), _duration(output), layouts[0]["audio"] is not None)


def next_part_path(output: str | Path) -> Path:
    """The output name for the next part of a run: ``concert.mp4`` → ``concert_part2.mp4``,
    ``concert_part2.mp4`` → ``concert_part3.mp4``."""
    output = Path(output)
    match = re.fullmatch(r"(.*)_part(\d+)", output.stem)
    stem, number = (match.group(1), int(match.group(2)) + 1) if match else (output.stem, 2)
    return output.with_name(f"{stem}_part{number}{output.suffix}")


def all_parts(last: str | Path) -> list[Path]:
    """Every part up to ``last``, in order: ``concert_part3.mp4`` → ``concert.mp4``,
    ``concert_part2.mp4``, ``concert_part3.mp4``."""
    last = Path(last)
    match = re.fullmatch(r"(.*)_part(\d+)", last.stem)
    if not match:
        return [last]
    stem, number = match.group(1), int(match.group(2))
    first = last.with_name(f"{stem}{last.suffix}")
    return [first] + [last.with_name(f"{stem}_part{n}{last.suffix}") for n in range(2, number + 1)]


def joined_path(last: str | Path) -> Path:
    """A name for the joined video: ``concert_part3.mp4`` → ``concert_full.mp4``."""
    first = all_parts(last)[0]
    return first.with_name(f"{first.stem}_full{first.suffix}")


def _quoted(path: Path) -> str:
    """A path for FFmpeg's concat list: absolute, with single quotes escaped."""
    return path.resolve().as_posix().replace("'", "'\\''")


def _layout(path: Path) -> dict:
    """The properties of the first video and audio streams that have to match."""
    fields = sorted({"codec_type", *_VIDEO_FIELDS, *_AUDIO_FIELDS})
    result = subprocess.run(
        [find_binary("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_entries", "stream=" + ",".join(fields), str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        raise VideoReadError(f"ffprobe couldn't read {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout or "{}").get("streams", [])

    def first(kind: str, wanted: dict) -> dict | None:
        found = next((s for s in streams if s.get("codec_type") == kind), None)
        return None if found is None else {key: found.get(key) for key in wanted}

    video = first("video", _VIDEO_FIELDS)
    if video is None:
        raise VideoReadError(f"{path} has no video.")
    return {"video": video, "audio": first("audio", _AUDIO_FIELDS)}


def _differences(a: dict, b: dict) -> list[str]:
    found = [
        f"{label} {a['video'][key]} vs {b['video'][key]}"
        for key, label in _VIDEO_FIELDS.items()
        if a["video"][key] != b["video"][key]
    ]
    if (a["audio"] is None) != (b["audio"] is None):
        found.append("one has audio and the other doesn't")
    elif a["audio"] is not None:
        found += [
            f"{label} {a['audio'][key]} vs {b['audio'][key]}"
            for key, label in _AUDIO_FIELDS.items()
            if a["audio"][key] != b["audio"][key]
        ]
    return found


def _duration(path: Path) -> float:
    result = subprocess.run(
        [find_binary("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0
