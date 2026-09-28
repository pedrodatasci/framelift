"""Reading video: metadata probing and a background frame reader.

The reader runs in its own thread so that decoding and pre-cleaning the next
frames happens *while* the AI is busy with the current one. On CPU-bound
filters (like ``heavy_noise``) this overlap is a free speed-up.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from .errors import VideoReadError
from .filters import Frame

_END_OF_STREAM = object()
_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class VideoInfo:
    """Basic facts about a video file."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


def probe_video(path: str | Path) -> VideoInfo:
    """Read a video's frame rate, frame count and resolution."""
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise VideoReadError(f"Couldn't open the video: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0:
        raise VideoReadError(
            f"Couldn't detect the frame rate of {path}. "
            "The file may be damaged or use an unusual container."
        )

    return VideoInfo(Path(path), fps, frame_count, width, height)


def read_frames(path: str | Path, numbers: Iterable[int]) -> list[tuple[int, Frame]]:
    """Read specific frames (1-based). Frames that can't be decoded are skipped."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(f"Couldn't open the video: {path}")

    frames = []
    try:
        for number in numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, number - 1))
            ok, image = capture.read()
            if ok:
                frames.append((number, image))
    finally:
        capture.release()

    if not frames:
        raise VideoReadError(f"Couldn't decode any frames from {path}")
    return frames


@dataclass(frozen=True)
class SourceFrame:
    """A decoded (and pre-cleaned) frame on its way to the AI."""

    number: int
    """Position in the original video, 1-based. Use this to resume later."""

    position: int
    """Position within the current run, 1-based (1 = first frame processed now)."""

    image: Frame


class FrameReader:
    """Decode and pre-process frames on a background thread.

    Use it as a context manager and iterate over it::

        with FrameReader("in.mp4", start_frame=1, preprocess=clean) as frames:
            for frame in frames:
                ...

    Iteration ends at the end of the video, after ``max_frames`` frames, or
    shortly after ``stop_event`` is set. If the background thread crashes, the
    exception is re-raised in the consuming thread instead of being lost.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        start_frame: int = 1,
        max_frames: int | None = None,
        preprocess: Callable[[Frame], Frame] = lambda frame: frame,
        prefetch: int = 2,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._path = Path(path)
        self._start_frame = max(1, int(start_frame))
        self._max_frames = max_frames
        self._preprocess = preprocess
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._external_stop = stop_event or threading.Event()
        self._closed = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="framelift-reader", daemon=True)

    # -- public API ----------------------------------------------------------

    def __enter__(self) -> FrameReader:
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __iter__(self) -> Iterator[SourceFrame]:
        while True:
            item = self._queue.get()

            if item is _END_OF_STREAM:
                if self._error is not None:
                    raise self._error
                return

            yield item

    @property
    def pending(self) -> int:
        """Frames already prepared and waiting in the queue."""
        return self._queue.qsize()

    def close(self, timeout: float = 3.0) -> None:
        """Stop the reader thread and wait (briefly) for it to exit."""
        self._closed.set()
        deadline = time.monotonic() + timeout

        # Keep draining so a reader blocked on a full queue can notice and exit.
        while self._thread.is_alive() and time.monotonic() < deadline:
            self._drain()
            self._thread.join(timeout=_POLL_SECONDS)

    # -- background thread ---------------------------------------------------

    def _should_stop(self) -> bool:
        return self._closed.is_set() or self._external_stop.is_set()

    def _run(self) -> None:
        capture = cv2.VideoCapture(str(self._path))

        try:
            if not capture.isOpened():
                raise VideoReadError(f"Couldn't open the video: {self._path}")

            first_index = self._start_frame - 1  # OpenCV counts from 0
            if first_index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)

            frame_number = first_index
            position = 0

            while not self._should_stop():
                if self._max_frames is not None and position >= self._max_frames:
                    break

                ok, image = capture.read()
                if not ok:
                    break

                frame_number += 1
                position += 1

                frame = SourceFrame(frame_number, position, self._preprocess(image))
                if not self._offer(frame):
                    break

        except BaseException as exc:  # surfaced to the consumer in __iter__
            self._error = exc
        finally:
            capture.release()
            self._put_end_of_stream()

    def _offer(self, item: SourceFrame) -> bool:
        """Queue ``item``, giving up (returning False) if we're asked to stop."""
        while not self._should_stop():
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _put_end_of_stream(self) -> None:
        # Always deliver the end marker so the consumer never blocks forever —
        # unless the consumer has already closed us and isn't listening anymore.
        while True:
            try:
                self._queue.put(_END_OF_STREAM, timeout=_POLL_SECONDS)
                return
            except queue.Full:
                if self._closed.is_set():
                    return

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
