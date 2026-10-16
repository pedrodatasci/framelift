"""``framelift join``: put the parts of a stopped and resumed run back together."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .console import setup_console
from .errors import FrameliftError
from .join import join_videos

log = logging.getLogger("framelift.cli")

EXAMPLES = """\
examples:
  # A run stopped after frame 10480 and was resumed with -o concert_hd_part2.mp4
  # --start-frame 10481: put the two parts together
  framelift join concert_hd.mp4 concert_hd_part2.mp4 -o concert_hd_full.mp4

The parts are joined in the order given, without re-encoding (it takes seconds and
loses nothing), each with its own stretch of the original audio, so the result stays in
sync. They must have been made with the same options: framelift checks that first.
"""


def build_join_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framelift join",
        description="Join the parts of a run that was stopped and resumed into one video.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("parts", nargs="+", metavar="PART", help="The parts, in order.")
    parser.add_argument(
        "-o", "--output", metavar="MP4", required=True, help="Where to save the joined video."
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Show more detail.")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Only show warnings and errors."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_join_parser().parse_args(argv)
    setup_console(args.verbose - args.quiet)
    try:
        result = join_videos(args.parts, args.output)
    except FrameliftError as exc:
        log.error("%s", exc)
        return 1

    minutes, seconds = divmod(round(result.duration_seconds), 60)
    log.info(
        "Joined %d parts into %s (%dm%02ds%s).",
        result.parts,
        result.path,
        minutes,
        seconds,
        ", with audio" if result.has_audio else "",
    )
    return 0
