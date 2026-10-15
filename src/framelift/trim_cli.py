"""``framelift trim``: choose the part of a video to enhance and save it as a new file."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .console import setup_console
from .errors import FrameliftError
from .trim import (
    TrimResult,
    _probe_streams,
    default_trim_path,
    parse_time,
    range_from_times,
    trim_video,
)
from .video import probe_video

log = logging.getLogger("framelift.cli")

EXAMPLES = """\
examples:
  # Pick the start and end on a timeline (opens a player window)
  framelift trim -i concert.mp4

  # Same, by time, without a window (servers, scripts)
  framelift trim -i concert.mp4 --start 1:30 --end 2:45.5

  # Pick and enhance in one go
  framelift -i concert.mp4 -o concert_hd.mp4 --pick-range

The cut is a new file (the original isn't touched), cut on the exact frame without
losing quality. Use it as the input from then on: resuming with --start-frame refers
to it.
"""


def build_trim_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framelift trim",
        description=(
            "Choose the part of a video to enhance, on a timeline in a player window "
            "(VLC) or by time, and save it as a new file that becomes your input."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", metavar="VIDEO", required=True, help="Video to cut.")
    parser.add_argument(
        "-o", "--output", metavar="MP4",
        help="Where to save the cut. (default: <input>_trim_<first>-<last frame>.mp4, "
             "next to the input)",
    )  # fmt: skip
    parser.add_argument(
        "--start", metavar="TIME",
        help="Start time instead of the window: seconds (90.5), M:SS or H:MM:SS. "
             "(default: the beginning)",
    )  # fmt: skip
    parser.add_argument(
        "--end", metavar="TIME",
        help="End time instead of the window, same formats. (default: the end)",
    )  # fmt: skip
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Show more detail.")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Only show warnings and errors."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_trim_parser().parse_args(argv)
    setup_console(args.verbose - args.quiet)

    try:
        if args.start is None and args.end is None:
            result = pick_and_trim(args.input, args.output)
        else:
            video = probe_video(args.input)
            start = None if args.start is None else parse_time(args.start)
            end = None if args.end is None else parse_time(args.end)
            frames = range_from_times(video, start, end)
            result = trim_video(args.input, args.output or default_trim_path(args.input, frames),
                                frames)  # fmt: skip
    except KeyboardInterrupt:
        log.warning("Cancelled.")
        return 130
    except FrameliftError as exc:
        log.error("%s", exc)
        return 1

    if result is None:
        log.warning("No range chosen, so nothing was cut.")
        return 1

    print(render_summary(result))
    return 0


def pick_and_trim(input_path, output_path=None) -> TrimResult | None:
    """Open the range picker and cut what's chosen; ``None`` if the window was cancelled.

    If a cut of exactly that range already exists (say, from a run that was stopped),
    it's reused instead of being cut again.
    """
    from .picker import pick_range  # imports Tk and VLC only now

    log.info("Choose the part to enhance in the player window (Enter to confirm, Esc to cancel).")
    frames = pick_range(input_path)
    if frames is None:
        return None

    output = Path(output_path) if output_path else default_trim_path(input_path, frames)
    if output_path is None and output.exists():
        log.info("Reusing the cut made earlier for this range: %s", output)
        has_audio = _probe_streams(output)["audio"]
        return TrimResult(output.resolve(), frames, probe_video(output).frame_count, has_audio)
    return trim_video(input_path, output, frames)


def render_summary(result: TrimResult) -> str:
    source_fps = probe_video(result.path).fps
    path = str(result.path)
    shown = f'"{path}"' if " " in path else path
    stem = Path(path).stem
    return "\n".join(
        [
            "",
            f"Saved {result.path}",
            f"  {result.frame_range.describe(source_fps)} of the original, "
            f"{result.frames:,} frames{'' if result.audio else ', no audio'}",
            "",
            "Next (it's the input from now on, also for --start-frame):",
            f"  framelift tune -i {shown}",
            f"  framelift -i {shown} -o {stem}_enhanced.mp4",
        ]
    )
