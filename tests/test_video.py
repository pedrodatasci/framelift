import threading

import pytest

from framelift import VideoReadError, probe_video
from framelift.video import FrameReader


def test_probe_video(make_video):
    info = probe_video(make_video(frames=12, fps=24))
    assert (info.width, info.height, info.frame_count) == (64, 48, 12)
    assert info.fps == pytest.approx(24)
    assert info.duration_seconds == pytest.approx(0.5)


def test_probe_missing_file(tmp_path):
    with pytest.raises(VideoReadError):
        probe_video(tmp_path / "missing.mp4")


def test_reader_yields_numbered_frames(make_video):
    with FrameReader(make_video(frames=5)) as reader:
        frames = list(reader)
    assert [f.number for f in frames] == [1, 2, 3, 4, 5]
    assert [f.position for f in frames] == [1, 2, 3, 4, 5]


def test_reader_start_frame_and_max_frames(make_video):
    with FrameReader(make_video(frames=10), start_frame=4, max_frames=3) as reader:
        frames = list(reader)
    assert [f.number for f in frames] == [4, 5, 6]
    assert [f.position for f in frames] == [1, 2, 3]
    # Frame n was written with brightness n*10 (±JPEG noise).
    assert frames[0].image.mean() == pytest.approx(40, abs=3)


def test_reader_applies_preprocess(make_video):
    with FrameReader(make_video(frames=2), preprocess=lambda img: img[:10, :10]) as reader:
        assert all(f.image.shape == (10, 10, 3) for f in reader)


def test_reader_surfaces_errors_from_its_thread(make_video):
    def broken(_):
        raise RuntimeError("filter exploded")

    with FrameReader(make_video(frames=3), preprocess=broken) as reader:
        with pytest.raises(RuntimeError, match="filter exploded"):
            list(reader)


def test_reader_stops_when_asked(make_video):
    stop = threading.Event()
    with FrameReader(make_video(frames=50), prefetch=1, stop_event=stop) as reader:
        seen = []
        for frame in reader:
            seen.append(frame.number)
            stop.set()
    assert len(seen) < 50


def test_closing_early_does_not_hang(make_video):
    reader = FrameReader(make_video(frames=50), prefetch=1)
    with reader:
        next(iter(reader))
    assert not reader._thread.is_alive()
