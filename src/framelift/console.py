"""Pretty terminal output that plays nicely with tqdm progress bars.

framelift itself only *emits* log records (as a library should). This module
is what turns them into friendly terminal output; the CLI calls
:func:`setup_console`, and you can too from a script or notebook.
"""

from __future__ import annotations

import logging
import os
import sys

from tqdm import tqdm

_COLORS = {
    logging.DEBUG: "\033[2m",  # dim
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[31;1m",
}
_RESET = "\033[0m"
_PREFIXES = {logging.WARNING: "warning: ", logging.ERROR: "error: ", logging.CRITICAL: "error: "}


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR") or not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":  # classic cmd.exe doesn't speak ANSI; modern terminals do
        return any(key in os.environ for key in ("WT_SESSION", "TERM_PROGRAM", "ANSICON"))
    return True


class _ConsoleHandler(logging.Handler):
    """Writes through ``tqdm.write`` so messages never break a progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
            message = _PREFIXES.get(record.levelno, "") + self.format(record)

            color = _COLORS.get(record.levelno)
            if color and _supports_color(stream):
                message = f"{color}{message}{_RESET}"

            tqdm.write(message, file=stream)
        except Exception:
            self.handleError(record)


def setup_console(verbosity: int = 0) -> None:
    """Show framelift's messages in the terminal.

    ``verbosity``: ``-1`` warnings only, ``0`` normal, ``1`` or more adds
    debug details (FFmpeg commands, file paths, sanity checks).
    """
    level = {-1: logging.WARNING, 0: logging.INFO}.get(verbosity, logging.DEBUG)
    if verbosity < -1:
        level = logging.ERROR

    logger = logging.getLogger("framelift")
    logger.handlers[:] = [h for h in logger.handlers if not isinstance(h, _ConsoleHandler)]

    handler = _ConsoleHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
