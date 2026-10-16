"""Choosing the part of a video to enhance, by eye: a player with a timeline.

::

    ┌──────────────────────────────────────────────┐
    │                 video (VLC)                  │
    ├──────────────────────────────────────────────┤
    │ ░░░░░░▕██████████████│█████████▏░░░░░░░░░░░░ │  ← start ▕, end ▏, playhead │
    ├──────────────────────────────────────────────┤
    │ Start 0:10.000 · End 1:25.500 · 75.5 s       │
    │ [Set start (I)] [Set end (O)] [Play]  [Use]  │
    └──────────────────────────────────────────────┘

Playback is done by VLC (through ``python-vlc``), which decodes any format smoothly
even on modest machines; the window and timeline are plain Tkinter. Neither is needed
by the rest of framelift, so both are imported only when the picker opens.

:class:`Selection` holds the picker's state and logic and needs no GUI, so it's tested
on its own.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import sys
import time
from pathlib import Path

from .errors import MissingDependencyError, VideoReadError
from .trim import FrameRange, format_time
from .video import VideoInfo, probe_video

INSTALL_VLC = (
    "The range picker plays the video with VLC. Install VLC media player "
    "(https://www.videolan.org/vlc/, same 32/64-bit as Python) and its Python binding: "
    'pip install python-vlc   (or: pip install "framelift-video[picker]")'
)
INSTALL_TK = (
    "The range picker needs Tkinter, which some Linux distributions ship separately: "
    "sudo apt install python3-tk  (Fedora: sudo dnf install python3-tkinter)"
)
NO_DISPLAY = (
    "The range picker needs a desktop session, and none was found. "
    "On a server, choose the range by time instead: framelift trim -i VIDEO --start 1:30 --end 2:45"
)

HANDLE_GRAB_PX = 10
"""How close (in pixels) a click must land to a handle to grab it."""


# -- state (no GUI) -----------------------------------------------------------------------


class Selection:
    """The chosen range and the playhead, in frames counted from 1.

    The ``set_*`` methods follow a key press: setting the start after the end moves
    the end along (and vice versa). The ``drag_*`` methods follow the mouse: a handle
    stops at the other one.
    """

    def __init__(self, frame_count: int, fps: float, initial: FrameRange | None = None) -> None:
        if frame_count < 1:
            raise VideoReadError("The video has no frames to choose from.")
        self.frame_count = frame_count
        self.fps = fps
        self.start = 1
        self.end = frame_count
        self.playhead = 1
        if initial is not None:
            self.start = self._clamp(initial.start)
            self.end = max(self.start, self._clamp(initial.end))
            self.playhead = self.start

    def _clamp(self, frame: int) -> int:
        return max(1, min(self.frame_count, int(frame)))

    # keys
    def set_start(self, frame: int) -> None:
        self.start = self._clamp(frame)
        self.end = max(self.end, self.start)

    def set_end(self, frame: int) -> None:
        self.end = self._clamp(frame)
        self.start = min(self.start, self.end)

    # mouse
    def drag_start(self, frame: int) -> None:
        self.start = min(self._clamp(frame), self.end)

    def drag_end(self, frame: int) -> None:
        self.end = max(self._clamp(frame), self.start)

    def move_playhead(self, frame: int) -> None:
        self.playhead = self._clamp(frame)

    def reset(self) -> None:
        self.start, self.end = 1, self.frame_count

    # frames ↔ time ↔ pixels
    def time_of(self, frame: int) -> float:
        """Seconds at which ``frame`` appears."""
        return (frame - 1) / self.fps

    def frame_at_time(self, seconds: float) -> int:
        """The frame on screen ``seconds`` into the video."""
        return self._clamp(int(seconds * self.fps + 1e-6) + 1)

    def x_of(self, frame: int, width: float) -> float:
        """Where ``frame`` begins on a timeline ``width`` pixels long."""
        return (frame - 1) / self.frame_count * width

    def frame_at(self, x: float, width: float) -> int:
        """The frame under pixel ``x`` of a timeline ``width`` pixels long."""
        if width <= 0:
            return self.playhead
        return self._clamp(int(x / width * self.frame_count) + 1)

    @property
    def frame_range(self) -> FrameRange:
        return FrameRange(self.start, self.end)

    @property
    def is_everything(self) -> bool:
        return self.start == 1 and self.end == self.frame_count

    def summary(self) -> str:
        length = (self.end - self.start + 1) / self.fps
        return (
            f"Start {format_time(self.time_of(self.start))} (frame {self.start:,})    "
            f"End {format_time(self.end / self.fps)} (frame {self.end:,})    "
            f"Selected {format_time(length)} · {self.end - self.start + 1:,} frames"
        )

    def playhead_label(self) -> str:
        return (
            f"{format_time(self.time_of(self.playhead))} / "
            f"{format_time(self.frame_count / self.fps)}  ·  frame {self.playhead:,}"
        )


# -- public entry point -------------------------------------------------------------------


def pick_range(path: str | Path, initial: FrameRange | None = None) -> FrameRange | None:
    """Open the picker on ``path`` and return the chosen range (``None`` if cancelled)."""
    path = Path(path).resolve()
    if not path.exists():
        raise VideoReadError(f"Input video not found: {path}")
    video = probe_video(path)
    tk, vlc = _load_backends()
    return _PickerWindow(tk, vlc, video, initial).run()


def _load_backends():
    try:
        import tkinter as tk
    except ImportError as exc:
        raise MissingDependencyError(INSTALL_TK) from exc

    try:
        import vlc
    except (ImportError, OSError, NameError, AttributeError) as exc:
        # python-vlc raises OSError/NameError when it can't find the VLC library itself.
        raise MissingDependencyError(f"{INSTALL_VLC}\n({exc})") from exc

    return tk, vlc


# -- the window ---------------------------------------------------------------------------

_BG = "#1e1e1e"
_PANEL = "#2b2b2b"
_TRACK = "#4a4a4a"
_SELECTED = "#3d7bd9"
_HANDLE = "#f2f2f2"
_PLAYHEAD = "#ffb020"
_TEXT = "#e8e8e8"
_MUTED = "#9a9a9a"
_PAD = 14  # timeline margin, so the handles at the very ends stay grabbable

_KEYS_HELP = (
    "Space play/pause · I / O set start / end at the playhead · ← → one frame · "
    "Shift+← → one second · Home / End jump to start / end · M mute · "
    "Enter use this range · Esc cancel"
)


class _PickerWindow:
    def __init__(self, tk, vlc, video: VideoInfo, initial: FrameRange | None) -> None:
        self.tk, self.vlc, self.video = tk, vlc, video
        self.selection = Selection(video.frame_count, video.fps, initial)
        self.result: FrameRange | None = None
        self.dragging: str | None = None
        self.play_until: int | None = None
        self._pending_seek: int | None = None
        self._last_seek = 0.0
        self._ready = False

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise MissingDependencyError(f"{NO_DISPLAY}\n({exc})") from exc

        args = ["--quiet", "--no-video-title-show", "--play-and-pause", "--no-osd"]
        if sys.platform.startswith("linux"):
            args.append("--no-xlib")  # Tk owns the X connection
        self.instance = vlc.Instance(args)
        if self.instance is None:
            self.root.destroy()
            raise MissingDependencyError(INSTALL_VLC)
        self.player = self.instance.media_player_new()
        self.player.set_media(self.instance.media_new(str(video.path)))

        self._build()

    # -- layout ---------------------------------------------------------------------

    def _build(self) -> None:
        tk, root = self.tk, self.root
        root.title(f"framelift — choose the part to enhance: {self.video.path.name}")
        root.configure(bg=_BG)
        root.minsize(640, 420)

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        ratio = self.video.width / max(1, self.video.height)
        width = min(1100, int(screen_w * 0.8), self.video.width * 2)
        height = int(width / ratio)
        if height > screen_h * 0.62:
            height = int(screen_h * 0.62)
            width = int(height * ratio)
        width = max(width, 640)

        self.screen = tk.Frame(root, bg="black", width=width, height=height)
        self.screen.pack(fill="both", expand=True)

        self.timeline = tk.Canvas(root, height=56, bg=_PANEL, highlightthickness=0,
                                  cursor="hand2")  # fmt: skip
        self.timeline.pack(fill="x")
        self.timeline.bind("<ButtonPress-1>", self._press)
        self.timeline.bind("<B1-Motion>", self._drag)
        self.timeline.bind("<ButtonRelease-1>", self._release)
        self.timeline.bind("<Configure>", lambda _event: self._draw())

        self.clock = tk.Label(root, bg=_BG, fg=_PLAYHEAD, anchor="w", font=("TkFixedFont", 11))
        self.clock.pack(fill="x", padx=10, pady=(8, 0))
        self.summary = tk.Label(root, bg=_BG, fg=_TEXT, anchor="w", font=("TkDefaultFont", 11))
        self.summary.pack(fill="x", padx=10, pady=(2, 0))

        buttons = tk.Frame(root, bg=_BG)
        buttons.pack(fill="x", padx=10, pady=8)

        def button(text, command, side="left", accent=False):
            widget = tk.Button(
                buttons, text=text, command=command, relief="flat", padx=10, pady=4,
                bg=_SELECTED if accent else _PANEL, fg=_TEXT, activebackground=_TRACK,
                activeforeground=_TEXT, highlightthickness=0, borderwidth=0,
            )  # fmt: skip
            widget.pack(side=side, padx=(0, 6) if side == "left" else (6, 0))
            return widget

        self.play_button = button("▶ Play", self.toggle_play)
        button("Set start  [I]", self.mark_start)
        button("Set end  [O]", self.mark_end)
        button("Play selection", self.play_selection)
        button("Reset", self.reset)
        button("Use this range  [Enter]", self.confirm, side="right", accent=True)
        button("Cancel  [Esc]", self.cancel, side="right")

        hint = tk.Label(root, text=_KEYS_HELP, bg=_BG, fg=_MUTED, font=("TkDefaultFont", 9),
                        wraplength=width - 20, justify="left", anchor="w")  # fmt: skip
        hint.pack(fill="x", padx=10, pady=(0, 8))

        keys = {
            "<space>": self.toggle_play,
            "<i>": self.mark_start, "<I>": self.mark_start,
            "<o>": self.mark_end, "<O>": self.mark_end,
            "<m>": self.toggle_mute, "<M>": self.toggle_mute,
            "<Left>": lambda: self.step(-1), "<Right>": lambda: self.step(1),
            "<Shift-Left>": lambda: self.step(-round(self.video.fps)),
            "<Shift-Right>": lambda: self.step(round(self.video.fps)),
            "<Home>": lambda: self.jump(self.selection.start),
            "<End>": lambda: self.jump(self.selection.end),
            "<Return>": self.confirm, "<KP_Enter>": self.confirm,
            "<Escape>": self.cancel,
        }  # fmt: skip
        for key, action in keys.items():
            root.bind(key, lambda _event, act=action: (act(), "break")[1])
        root.protocol("WM_DELETE_WINDOW", self.cancel)

    # -- lifecycle ----------------------------------------------------------------------

    def run(self) -> FrameRange | None:
        self.root.update_idletasks()
        self.root.update()
        self._attach_video()
        self.player.audio_set_mute(True)
        self.player.play()  # paused again on the first frame, see _tick
        self._update_labels()
        self.root.after(40, self._tick)
        self.root.focus_force()
        try:
            self.root.mainloop()
        finally:
            self._close()
        return self.result

    def _attach_video(self) -> None:
        handle = self.screen.winfo_id()
        if sys.platform.startswith("win"):
            self.player.set_hwnd(handle)
        elif sys.platform == "darwin":
            view = _macos_view(handle, self.tk.TkVersion)
            if view:
                self.player.set_nsobject(view)
        else:
            self.player.set_xwindow(handle)

    def _close(self) -> None:
        for release in (self.player.stop, self.player.release, self.instance.release):
            with contextlib.suppress(Exception):  # best effort while shutting down
                release()
        with contextlib.suppress(self.tk.TclError):
            self.root.destroy()

    def _tick(self) -> None:
        """Follow playback: move the playhead, stop at the end of 'Play selection'."""
        player = self.player
        if not self._ready and player.is_playing():
            # VLC only shows a picture once playing; pause right on the first frame.
            self._ready = True
            player.set_pause(1)
            player.audio_set_mute(False)
            self.seek(self.selection.playhead)

        if self._pending_seek is not None and time.monotonic() - self._last_seek > 0.06:
            self.seek(self._pending_seek)

        if self.dragging is None and player.is_playing():
            milliseconds = player.get_time()
            if milliseconds >= 0:
                self.selection.move_playhead(self.selection.frame_at_time(milliseconds / 1000))
            if self.play_until is not None and self.selection.playhead >= self.play_until:
                end, self.play_until = self.play_until, None
                player.set_pause(1)
                self.jump(end)  # land exactly on the last frame of the selection

        self.play_button.configure(text="❚❚ Pause" if player.is_playing() else "▶ Play")
        self._update_labels()
        self.root.after(40, self._tick)

    # -- actions ------------------------------------------------------------------------

    def seek(self, frame: int) -> None:
        """Show ``frame`` (a third of a frame in, so rounding can't land on the one before)."""
        self._pending_seek = None
        self._last_seek = time.monotonic()
        milliseconds = (self.selection.time_of(frame) + 0.33 / self.video.fps) * 1000
        self.player.set_time(int(milliseconds))

    def _seek_soon(self, frame: int) -> None:
        """While dragging: seek at most every 60 ms, so VLC keeps up."""
        if time.monotonic() - self._last_seek > 0.06:
            self.seek(frame)
        else:
            self._pending_seek = frame

    def toggle_play(self) -> None:
        self.play_until = None
        if self.player.is_playing():
            self.player.set_pause(1)
            self.seek(self.selection.playhead)
        else:
            if self.selection.playhead >= self.selection.frame_count:
                self.jump(1)
            self.player.set_pause(0)

    def play_selection(self) -> None:
        self.jump(self.selection.start)
        self.play_until = self.selection.end
        self.player.set_pause(0)

    def pause(self) -> None:
        self.play_until = None
        if self.player.is_playing():
            self.player.set_pause(1)

    def step(self, frames: int) -> None:
        self.pause()
        self.jump(self.selection.playhead + frames)

    def jump(self, frame: int) -> None:
        self.selection.move_playhead(frame)
        self.seek(self.selection.playhead)
        self._draw()

    def mark_start(self) -> None:
        self.selection.set_start(self.selection.playhead)
        self._draw()

    def mark_end(self) -> None:
        self.selection.set_end(self.selection.playhead)
        self._draw()

    def reset(self) -> None:
        self.selection.reset()
        self._draw()

    def toggle_mute(self) -> None:
        self.player.audio_set_mute(not self.player.audio_get_mute())

    def confirm(self) -> None:
        self.result = self.selection.frame_range
        self.root.quit()

    def cancel(self) -> None:
        self.result = None
        self.root.quit()

    # -- timeline ---------------------------------------------------------------------

    def _track_width(self) -> float:
        return max(1, self.timeline.winfo_width() - 2 * _PAD)

    def _x(self, frame: int) -> float:
        return _PAD + self.selection.x_of(frame, self._track_width())

    def _frame_at(self, x: float) -> int:
        return self.selection.frame_at(x - _PAD, self._track_width())

    def _press(self, event) -> None:
        to_start = abs(event.x - self._x(self.selection.start))
        to_end = abs(event.x - self._x(self.selection.end + 1))
        if min(to_start, to_end) > HANDLE_GRAB_PX:
            self.dragging = "playhead"
        else:
            self.dragging = "start" if to_start <= to_end else "end"
        self.pause()
        self._drag(event)

    def _drag(self, event) -> None:
        frame = self._frame_at(event.x)
        if self.dragging == "start":
            self.selection.drag_start(frame)
            frame = self.selection.start
        elif self.dragging == "end":
            self.selection.drag_end(frame)
            frame = self.selection.end
        self.selection.move_playhead(frame)  # the video follows whatever is dragged
        self._seek_soon(frame)
        self._draw()

    def _release(self, _event) -> None:
        if self._pending_seek is not None:
            self.seek(self._pending_seek)
        self.dragging = None

    def _draw(self) -> None:
        canvas, selection = self.timeline, self.selection
        canvas.delete("all")
        height = canvas.winfo_height()
        top, bottom = 14, height - 14
        left, right = _PAD, _PAD + self._track_width()

        canvas.create_rectangle(left, top, right, bottom, fill=_TRACK, width=0)
        start_x, end_x = self._x(selection.start), self._x(selection.end + 1)
        canvas.create_rectangle(start_x, top, end_x, bottom, fill=_SELECTED, width=0)

        for x, direction in ((start_x, 1), (end_x, -1)):
            canvas.create_rectangle(x - 2, top - 6, x + 2, bottom + 6, fill=_HANDLE, width=0)
            canvas.create_polygon(x, top - 6, x + 8 * direction, top - 6, x, top + 2,
                                  fill=_HANDLE, width=0)  # fmt: skip

        playhead_x = self._x(selection.playhead)
        canvas.create_line(playhead_x, 4, playhead_x, height - 4, fill=_PLAYHEAD, width=2)
        canvas.create_polygon(playhead_x - 5, 2, playhead_x + 5, 2, playhead_x, 9,
                              fill=_PLAYHEAD, width=0)  # fmt: skip

    def _update_labels(self) -> None:
        self.summary.configure(text=self.selection.summary())
        self.clock.configure(text=self.selection.playhead_label())
        if self.dragging is None:
            self._draw()


def _macos_view(window_id: int, tk_version: float) -> int | None:
    """The NSView behind a Tk window, which VLC needs to draw into on macOS.

    The same approach as python-vlc's own Tk example: Tk's library exposes it through
    ``TkMacOSXGetRootControl``. Returns ``None`` if it can't be found.
    """
    candidates = [
        os.path.join(getattr(sys, "base_prefix", sys.prefix), "lib", f"libtk{tk_version}.dylib"),
        ctypes.util.find_library(f"tk{tk_version}"),
    ]
    for candidate in filter(None, candidates):
        try:
            library = ctypes.cdll.LoadLibrary(candidate)
            get_view = library.TkMacOSXGetRootControl
        except (OSError, AttributeError):
            continue
        get_view.restype = ctypes.c_void_p
        get_view.argtypes = [ctypes.c_void_p]
        return get_view(window_id)
    return None
