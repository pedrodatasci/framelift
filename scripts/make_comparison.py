"""Make before/after comparison GIFs for the README gallery.

    python scripts/make_comparison.py original.mp4 enhanced.mp4 --name concert

Writes to ``docs/gallery/``:

* ``<name>_zoom.gif``: a 100% crop of the most detailed area, original vs. framelift,
  taken from the steadiest scene (fast motion and cuts make a zoom jumpy).
* ``<name>_full.gif``: the whole frame, side by side.
* The same two as ``.mp4`` (full quality, for sharing; ``.gitignore`` keeps them out of git).

Then it prints the Markdown to paste into GALLERY.md and GALLERY.pt-BR.md.
Needs FFmpeg on the PATH and framelift installed (``pip install -e .``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from framelift.analysis import detail_score, softness

GALLERY = Path("docs/gallery")
ZOOM_SIZE = (720, 540)  # enhanced pixels shown at 100% in each half of the zoom
FULL_HEIGHT = 540
GIF_ATTEMPTS = ((12, 960), (10, 840), (8, 720))  # (fps, width), tried until the GIF fits
MAX_GIF_MB = 6.0
MAX_ZOOM_SECONDS = 3.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("original", type=Path, help="The video before framelift.")
    parser.add_argument("enhanced", type=Path, help="framelift's output for that video.")
    parser.add_argument("--name", required=True, help="Short file name, e.g. 'vhs_wedding'.")
    parser.add_argument(
        "--zoom", metavar="X,Y,W,H",
        help="Zoom region in ORIGINAL pixels, if you'd rather pick it yourself.",
    )  # fmt: skip
    parser.add_argument("--start", type=float, help="Zoom scene start, in seconds (with --end).")
    parser.add_argument("--end", type=float, help="Zoom scene end, in seconds.")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        sys.exit("FFmpeg isn't on the PATH.")

    original, fps = _read_all(args.original)
    enhanced_size = _video_size(args.enhanced)
    height, width = original[0].shape[:2]
    ratio = enhanced_size[1] / height
    print(f"{args.original.name}: {width}x{height}, {len(original)} frames, "
          f"enhanced {enhanced_size[0]}x{enhanced_size[1]} ({ratio:g}x)")  # fmt: skip

    if args.start is not None and args.end is not None:
        scene = (round(args.start * fps), round(args.end * fps))
    else:
        scene = steadiest_scene(original, max_frames=round(MAX_ZOOM_SECONDS * fps))
    print(f"Zoom scene: frames {scene[0]}–{scene[1]} ({(scene[1] - scene[0]) / fps:.1f} s)")

    crop_width = min(ZOOM_SIZE[0], enhanced_size[0])
    crop_height = min(ZOOM_SIZE[1], enhanced_size[1])
    if args.zoom:
        x, y, w, h = (int(v) for v in args.zoom.split(","))
    else:
        w, h = round(crop_width / ratio), round(crop_height / ratio)
        x, y = detailed_region(original[scene[0] : scene[1]], w, h)
    print(f"Zoom region (original pixels): {x},{y},{w},{h}")

    GALLERY.mkdir(parents=True, exist_ok=True)
    zoom_mp4 = GALLERY / f"{args.name}_zoom.mp4"
    full_mp4 = GALLERY / f"{args.name}_full.mp4"
    label = _label_filter()

    _ffmpeg(
        args.original, args.enhanced, zoom_mp4,
        f"[0:v]trim=start_frame={scene[0]}:end_frame={scene[1]},setpts=PTS-STARTPTS,"
        f"crop={w}:{h}:{x}:{y},scale={crop_width}:{crop_height}:flags=bicubic,"
        f"{label}:text='Original (zoomed)'[a];"
        f"[1:v]trim=start_frame={scene[0]}:end_frame={scene[1]},setpts=PTS-STARTPTS,"
        f"crop={crop_width}:{crop_height}:{round(x * ratio)}:{round(y * ratio)},"
        f"{label}:text='framelift (zoomed)'[b];[a][b]hstack",
    )  # fmt: skip
    full_width = round(FULL_HEIGHT * enhanced_size[0] / enhanced_size[1] / 2) * 2
    _ffmpeg(
        args.original, args.enhanced, full_mp4,
        f"[0:v]scale={full_width}:{FULL_HEIGHT}:flags=bicubic,"
        f"{label}:text='Original ({height}p)'[a];"
        f"[1:v]scale={full_width}:{FULL_HEIGHT}:flags=lanczos,{label}:text='framelift'[b];"
        f"[a][b]hstack",
    )  # fmt: skip

    grayscale = _is_grayscale(original)
    for mp4 in (zoom_mp4, full_mp4):
        gif = mp4.with_suffix(".gif")
        # Fast motion and flashing lights compress badly in GIF: step down the frame
        # rate and width until the file is light enough for a README.
        for fps, gif_width in GIF_ATTEMPTS:
            _to_gif(mp4, gif, grayscale, fps, gif_width)
            size = gif.stat().st_size / 1024**2
            if size <= MAX_GIF_MB:
                break
        note = "  (still heavy: try a shorter clip)" if size > MAX_GIF_MB else ""
        print(f"  {gif}  {size:.1f} MB at {fps} fps, {gif_width} px wide{note}")

    _print_markdown(args.name)
    return 0


def steadiest_scene(frames: list[np.ndarray], max_frames: int) -> tuple[int, int]:
    """A stretch without cuts or fast motion, as long and as sharp as possible.

    Long but motion-blurred scenes make poor examples (the AI can't undo motion
    blur), so each candidate's length is weighed by how sharp it is.
    """
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120)).astype(float)
             for f in frames]  # fmt: skip
    changes = np.array([np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))])
    threshold = max(12.0, 1.5 * float(np.median(changes)))

    start, runs = 0, []
    for i, change in enumerate(changes, start=1):
        if change > threshold:
            runs.append((start, i))
            start = i
    runs.append((start, len(frames)))

    def score(run: tuple[int, int]) -> float:
        length = min(run[1] - run[0], max_frames)
        sharpness = 1 - softness(frames[(run[0] + run[1]) // 2])
        return length * sharpness**2

    long_enough = [run for run in runs if run[1] - run[0] >= min(15, len(frames))] or runs
    best = max(long_enough, key=score)

    if best[1] - best[0] > max_frames:
        middle = (best[0] + best[1]) // 2
        best = (middle - max_frames // 2, middle - max_frames // 2 + max_frames)
    return best


def detailed_region(frames: list[np.ndarray], width: int, height: int) -> tuple[int, int]:
    """Top-left corner of the window with the most detail across the scene."""
    frame_height, frame_width = frames[0].shape[:2]
    width, height = min(width, frame_width), min(height, frame_height)
    samples = frames[:: max(1, len(frames) // 5)]

    best, best_score = (0, 0), -1.0
    for row in range(5):
        for col in range(5):
            x = round((frame_width - width) * col / 4)
            y = round((frame_height - height) * row / 4)
            score = sum(detail_score(f[y : y + height, x : x + width]) for f in samples)
            if score > best_score:
                best, best_score = (x, y), score
    return best


def _read_all(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        sys.exit(f"Couldn't read {path}")
    return frames, fps


def _video_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return size


def _is_grayscale(frames: list[np.ndarray]) -> bool:
    middle = frames[len(frames) // 2]
    return float(cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 12


def _label_filter() -> str:
    font = ""
    for candidate in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        found = next((p for d in _font_dirs() for p in d.rglob(candidate)), None)
        if found:
            # FFmpeg's filter syntax needs the drive colon escaped on Windows.
            font = "fontfile='" + found.as_posix().replace(":", r"\:") + "':"
            break
    return (f"drawtext={font}x=16:y=16:fontsize=26:fontcolor=white:"
            "box=1:boxcolor=black@0.6:boxborderw=8")  # fmt: skip


def _font_dirs() -> list[Path]:
    candidates = [Path("/usr/share/fonts"), Path("C:/Windows/Fonts"), Path("/Library/Fonts"),
                  Path("/System/Library/Fonts")]  # fmt: skip
    return [d for d in candidates if d.exists()]


def _ffmpeg(original: Path, enhanced: Path, output: Path, graph: str) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(original), "-i", str(enhanced),
         "-filter_complex", graph, "-an", "-c:v", "libx264", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        check=True,
    )  # fmt: skip


def _to_gif(mp4: Path, gif: Path, grayscale: bool, fps: int, width: int) -> None:
    if grayscale:  # fewer gray levels are plenty, and much smaller
        palette = ("format=gray,format=rgb24,split[a][b];[a]palettegen=max_colors=48:"
                   "stats_mode=full[p];[b][p]paletteuse=dither=none")  # fmt: skip
    else:
        palette = ("split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
                   "[b][p]paletteuse=dither=bayer:bayer_scale=4")  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
         f"fps={fps},scale={width}:-1:flags=lanczos,{palette}", str(gif)],
        check=True,
    )  # fmt: skip


def _print_markdown(name: str) -> None:
    print(f"""
Paste into GALLERY.md (and translate the text for GALLERY.pt-BR.md):

## Title of this example

One sentence about the footage and what to look for. 480p → 1080p.

![Before and after, zoomed in](docs/gallery/{name}_zoom.gif)

<details>
<summary>Full frame</summary>

![Before and after, full frame](docs/gallery/{name}_full.gif)

</details>
""")


if __name__ == "__main__":
    raise SystemExit(main())
