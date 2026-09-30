"""Command-line interface: ``framelift --input in.mp4 --output out.mp4``.

Every flag from the original ``main.py`` is still here with the same name and
default, so old commands keep working unchanged.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence

from . import __version__
from .console import setup_console
from .errors import FrameliftError
from .filters import PROFILES
from .models import MODELS
from .options import DEFAULT_MODEL, DEVICES, ENCODERS, X264_PRESETS, EnhanceOptions

log = logging.getLogger("framelift.cli")

EXAMPLES = """\
examples:
  # Quick 10-second preview before committing to a long render
  framelift -i clip.mp4 -o preview.mp4 --max-frames 300

  # Old TV capture: clean it up, double the size, lean a bit more on the AI
  framelift -i vhs.mp4 -o vhs_hd.mp4 --profile old_tv --scale 2 --ai-strength 0.6

  # NVIDIA GPU at full speed
  framelift -i in.mp4 -o out.mp4 --device cuda --half --tile 768 --encoder nvenc

  # Resume a run that stopped after frame 1200
  framelift -i in.mp4 -o part2.mp4 --start-frame 1201

Not sure which settings to use? Let framelift measure and suggest them:
  framelift tune -i clip.mp4

Press Ctrl+C once to stop early and still get a playable MP4 of what's done.
Full documentation: https://github.com/pedrodatasci/framelift#readme
"""


def build_parser() -> argparse.ArgumentParser:
    defaults = EnhanceOptions()
    parser = argparse.ArgumentParser(
        prog="framelift",
        description=(
            "Upscale and restore videos with Real-ESRGAN. Uses an NVIDIA GPU when "
            "available and falls back to the CPU everywhere else (Intel Iris included)."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    files = parser.add_argument_group("input / output")
    files.add_argument("-i", "--input", metavar="VIDEO", help="Video to enhance.")
    files.add_argument("-o", "--output", metavar="MP4", help="Where to save the result (MP4).")
    files.add_argument(
        "--no-audio", dest="audio", action="store_false",
        help="Leave the audio out. By default the input's audio is copied over, "
             "trimmed to match the enhanced part.",
    )  # fmt: skip

    look = parser.add_argument_group("how the result looks")
    look.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS),
        help="Which Real-ESRGAN model to use. realesr-animevideov3 is the fastest and "
        "smoothest; realesrgan-x4plus gives the most detail on real footage. "
        "See --list-models. (default: %(default)s)",
    )
    look.add_argument(
        "--scale",
        type=float,
        default=defaults.scale,
        help="Output size relative to the input: 2 doubles width and height. "
        "Ignored with --same-resolution. (default: %(default)s)",
    )
    look.add_argument(
        "--same-resolution",
        action="store_true",
        help="Keep the exact input resolution — restore quality without resizing.",
    )
    look.add_argument(
        "--ai-strength",
        type=float,
        default=defaults.ai_strength,
        help="How much of the AI result to keep, from 0 to 1. 1.0 = full AI, "
        "0.45 = balanced, 0.30 = natural. (default: %(default)s)",
    )
    look.add_argument(
        "--profile",
        default=defaults.profile,
        choices=list(PROFILES),
        help="Clean-up filter applied before the AI. none/minimal are the fastest. "
        "See --list-profiles. (default: %(default)s)",
    )
    look.add_argument(
        "--output-fps",
        type=float,
        default=defaults.output_fps,
        metavar="FPS",
        help="Output frame rate, e.g. 60. Frames are repeated to reach it, so the "
        "duration doesn't change. Can't be lower than the input's. (default: same as input)",
    )

    frames = parser.add_argument_group("which frames to process")
    frames.add_argument(
        "--start-frame",
        type=int,
        default=defaults.start_frame,
        metavar="N",
        help="Frame to start from, counting from 1. If a run stopped after frame 1200, "
        "use 1201 to continue. (default: %(default)s)",
    )
    frames.add_argument(
        "--max-frames",
        type=int,
        default=defaults.max_frames,
        metavar="N",
        help="Only process N frames from --start-frame. Perfect for quick previews.",
    )

    hardware = parser.add_argument_group("hardware & performance")
    hardware.add_argument(
        "--device",
        default=defaults.device,
        choices=DEVICES,
        help="auto uses an NVIDIA GPU (CUDA) if there is one, otherwise the CPU. "
        "Use cpu on Intel Iris / AMD / Apple machines. (default: %(default)s)",
    )
    hardware.add_argument(
        "--gpu-id",
        type=int,
        default=defaults.gpu_id,
        metavar="ID",
        help="Which CUDA GPU to use if you have several. (default: %(default)s)",
    )
    hardware.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 on the GPU: faster and uses less VRAM. Ignored on CPU.",
    )
    hardware.add_argument(
        "--tile",
        type=int,
        default=defaults.tile,
        metavar="PX",
        help="Process each frame in tiles of this size to save memory. CPU: 128–256. "
        "CUDA: try 768–1024. 0 = no tiling. Lower it if you run out of memory. "
        "(default: %(default)s)",
    )
    hardware.add_argument(
        "--tile-pad",
        type=int,
        default=defaults.tile_pad,
        metavar="PX",
        help="Overlap between tiles, hides visible seams. (default: %(default)s)",
    )
    hardware.add_argument(
        "--pre-pad",
        type=int,
        default=defaults.pre_pad,
        metavar="PX",
        help="Padding around the whole frame before inference; can reduce edge "
        "artifacts. (default: %(default)s)",
    )
    hardware.add_argument(
        "--torch-threads",
        type=int,
        default=defaults.torch_threads,
        metavar="N",
        help="CPU threads for PyTorch. 0 lets PyTorch decide. (default: %(default)s)",
    )
    hardware.add_argument(
        "--prefetch",
        type=int,
        default=defaults.prefetch,
        metavar="N",
        help="Frames the reader thread prepares ahead of the AI. (default: %(default)s)",
    )
    hardware.add_argument(
        "--weights-dir",
        default=str(defaults.weights_dir),
        metavar="DIR",
        help="Folder for model weights; missing ones are downloaded automatically. "
        "(default: %(default)s)",
    )

    encoding = parser.add_argument_group("encoding")
    encoding.add_argument(
        "--encoder",
        default=defaults.encoder,
        choices=ENCODERS,
        help="cpu = libx264 (works everywhere). nvenc = NVIDIA hardware encoder; "
        "falls back to cpu automatically if it's not available. (default: %(default)s)",
    )
    encoding.add_argument(
        "--crf",
        type=int,
        default=defaults.crf,
        help="Quality: lower is better and bigger. 18–22 is the sweet spot. (default: %(default)s)",
    )
    encoding.add_argument(
        "--x264-preset",
        default=defaults.x264_preset,
        choices=X264_PRESETS,
        help="libx264 speed vs. file size. Slower presets make smaller files at the "
        "same quality. (default: %(default)s)",
    )

    diagnostics = parser.add_argument_group("diagnostics")
    diagnostics.add_argument(
        "--debug-timings",
        action="store_true",
        help="Print how long each stage (queue, AI, blend, write) takes.",
    )
    diagnostics.add_argument(
        "--timing-every",
        type=int,
        default=defaults.timing_every,
        metavar="N",
        help="With --debug-timings: report every N frames. (default: %(default)s)",
    )
    diagnostics.add_argument(
        "--gc-every",
        type=int,
        default=defaults.gc_every,
        metavar="N",
        help="With --debug-timings: free memory every N frames. 0 = never. (default: %(default)s)",
    )
    diagnostics.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show more detail (FFmpeg commands, paths, sanity checks).",
    )
    diagnostics.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Only show warnings and errors (the progress bar stays).",
    )

    info = parser.add_argument_group("information")
    info.add_argument(
        "--list-models", action="store_true", help="Describe the available models and exit."
    )
    info.add_argument(
        "--list-profiles", action="store_true", help="Describe the clean-up profiles and exit."
    )
    info.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    legacy = parser.add_argument_group("legacy (accepted but ignored)")
    legacy.add_argument(
        "--frame-format",
        default="jpg",
        help="Unused: frames go straight to FFmpeg now. Kept so old commands still run.",
    )
    legacy.add_argument(
        "--keep-temp",
        action="store_true",
        help="Unused: there are no temporary frame files anymore. Kept so old commands still run.",
    )

    return parser


def options_from_args(args: argparse.Namespace) -> EnhanceOptions:
    """Map parsed CLI arguments onto :class:`EnhanceOptions` (same names)."""
    return EnhanceOptions(**{name: getattr(args, name) for name in EnhanceOptions.field_names()})


def options_to_args(options: EnhanceOptions) -> list[str]:
    """The CLI flags that reproduce ``options`` (only those that differ from the defaults)."""
    defaults = EnhanceOptions()
    args: list[str] = []

    for name in EnhanceOptions.field_names():
        value, default = getattr(options, name), getattr(defaults, name)
        if str(value) == str(default) or (name == "scale" and options.same_resolution):
            continue

        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            # Options that are on by default are turned off with --no-<name>.
            args.append(flag if value else "--no-" + name.replace("_", "-"))
        elif isinstance(value, float):
            args += [flag, f"{value:g}"]
        else:
            args += [flag, str(value)]

    return args


def format_command(input_path, output_path, options: EnhanceOptions) -> str:
    """A ready-to-paste ``framelift`` command line (works in bash and PowerShell)."""
    parts = ["framelift", "-i", str(input_path), "-o", str(output_path), *options_to_args(options)]
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


class GracefulInterrupt:
    """First Ctrl+C: finish the current frame and save. Second Ctrl+C: quit now."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._previous_handler = None

    def __enter__(self) -> threading.Event:
        self._previous_handler = signal.signal(signal.SIGINT, self._handle)
        return self.stop_event

    def __exit__(self, *exc_info) -> None:
        signal.signal(signal.SIGINT, self._previous_handler)

    def _handle(self, signum, frame) -> None:
        if self.stop_event.is_set():
            log.warning("Second Ctrl+C — quitting right away.")
            raise SystemExit(130)

        self.stop_event.set()
        log.warning(
            "Got it! Finishing the current frame and saving what's done so far as a playable MP4. "
            "(Press Ctrl+C again to quit immediately.)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["tune"]:
        from .tune_cli import main as tune_main

        return tune_main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        _print_catalog(
            "Models", {name: spec.summary for name, spec in MODELS.items()}, DEFAULT_MODEL
        )
        return 0
    if args.list_profiles:
        _print_catalog("Profiles", {name: p.description for name, p in PROFILES.items()}, "none")
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required (try --help for examples).")

    setup_console(args.verbose - args.quiet)

    try:
        from .pipeline import enhance_video  # heavy (PyTorch): import only when needed

        with GracefulInterrupt() as stop_event:
            log.info(
                "framelift %s — press Ctrl+C once to stop early and keep what's done.", __version__
            )
            result = enhance_video(
                args.input,
                args.output,
                options_from_args(args),
                stop_event=stop_event,
            )
    except FrameliftError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # unexpected: show the traceback only when asked
        log.error("%s", exc, exc_info=args.verbose > 0)
        if not args.verbose:
            log.error("Run again with -v to see the full traceback.")
        return 1

    _report(result)
    return 0


def _report(result) -> None:
    minutes, seconds = divmod(int(result.elapsed_seconds), 60)
    status = "Saved a partial video" if result.interrupted else "Done!"

    log.info(
        "\n%s %s\n  frames %d → %d (%d enhanced, %d written) in %dm%02ds",
        status,
        result.output_path,
        result.first_frame,
        result.last_frame,
        result.frames_processed,
        result.frames_written,
        minutes,
        seconds,
    )

    if result.audio in {"copied", "converted"}:
        log.info("  with the original audio%s", " (as AAC)" if result.audio == "converted" else "")

    if result.resume_from is not None:
        log.info("To continue from here, run again with:  --start-frame %d", result.resume_from)


def _print_catalog(title: str, entries: dict, default: str) -> None:
    width = max(len(name) for name in entries)
    print(f"{title}:")
    for name, description in entries.items():
        marker = " (default)" if name == default else ""
        print(f"  {name:<{width}}  {description}{marker}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
