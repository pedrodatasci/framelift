"""Exceptions raised by framelift.

Everything inherits from :class:`FrameliftError`, so callers who just want
"did it work?" can catch a single type, while callers who care can react to
specific failures (for example, retrying with a smaller tile on
:class:`OutOfMemoryError`).
"""

from __future__ import annotations


class FrameliftError(Exception):
    """Base class for every error framelift raises on purpose."""


class MissingDependencyError(FrameliftError):
    """A required Python package or external program is not installed."""


class InvalidOptionError(FrameliftError, ValueError):
    """An option has a value that can't work (e.g. a negative scale)."""


class VideoReadError(FrameliftError):
    """The input video could not be opened or its metadata is unusable."""


class EncodingError(FrameliftError):
    """FFmpeg failed while writing the output video."""


class OutOfMemoryError(FrameliftError):
    """The GPU/CPU ran out of memory while upscaling a frame."""


class BlackFrameError(FrameliftError):
    """A sanity check found an (almost) completely black frame.

    This almost always means something upstream is broken — a bad decode,
    a corrupt model file, or a driver issue — so we stop early instead of
    spending hours rendering a black video.
    """


class NothingToDoError(FrameliftError):
    """The run finished without processing a single frame."""
