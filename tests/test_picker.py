"""The range picker's logic. The window itself needs a display and VLC, so it isn't
opened here; :class:`Selection` holds everything it decides."""

import pytest

from framelift import FrameRange, MissingDependencyError
from framelift.picker import Selection, _load_backends


@pytest.fixture
def selection():
    return Selection(frame_count=100, fps=25.0)


def test_starts_with_the_whole_video(selection):
    assert selection.frame_range == FrameRange(1, 100)
    assert selection.is_everything and selection.playhead == 1


def test_initial_range_is_clamped():
    selection = Selection(100, 25.0, FrameRange(90, 500))
    assert selection.frame_range == FrameRange(90, 100)
    assert selection.playhead == 90


def test_keys_push_the_other_end_along(selection):
    selection.set_end(40)
    selection.set_start(60)  # after the end: the end follows
    assert selection.frame_range == FrameRange(60, 60)
    selection.set_end(10)  # before the start: the start follows
    assert selection.frame_range == FrameRange(10, 10)


def test_dragging_stops_at_the_other_handle(selection):
    selection.set_start(30)
    selection.set_end(50)
    selection.drag_start(80)
    assert selection.start == 50
    selection.drag_end(5)
    assert selection.end == 50
    selection.drag_start(-10)
    selection.drag_end(1000)
    assert selection.frame_range == FrameRange(1, 100)


def test_pixels_and_frames(selection):
    assert selection.frame_at(0, 500) == 1
    assert selection.frame_at(250, 500) == 51
    assert selection.frame_at(499.9, 500) == 100
    assert selection.frame_at(9999, 500) == 100
    assert selection.x_of(51, 500) == 250
    assert selection.frame_at(selection.x_of(37, 500), 500) == 37


def test_times_and_frames(selection):
    assert selection.time_of(26) == 1.0
    assert selection.frame_at_time(1.0) == 26
    assert selection.frame_at_time(1.039) == 26
    assert selection.frame_at_time(99) == 100


def test_summary(selection):
    selection.set_start(26)
    selection.set_end(75)
    selection.move_playhead(30)
    assert selection.summary().endswith("Selected 0:02.000 · 50 frames")
    assert "Start 0:01.000 (frame 26)" in selection.summary()
    assert "End 0:03.000 (frame 75)" in selection.summary()
    assert selection.playhead_label() == "0:01.160 / 0:04.000  ·  frame 30"
    selection.reset()
    assert selection.is_everything


def test_missing_vlc_gives_install_instructions(monkeypatch):
    import builtins

    pytest.importorskip("tkinter")  # checked first, and some Pythons ship without it

    real_import = builtins.__import__

    def no_vlc(name, *args, **kwargs):
        if name == "vlc":
            raise ImportError("No module named 'vlc'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_vlc)
    with pytest.raises(MissingDependencyError, match="python-vlc"):
        _load_backends()
