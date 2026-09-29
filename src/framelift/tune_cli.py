"""``framelift tune``: the command-line face of :func:`framelift.tune_video`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .cli import format_command
from .console import setup_console
from .errors import FrameliftError
from .filters import PROFILES
from .models import MODELS
from .options import DEVICES, ENCODERS, EnhanceOptions

log = logging.getLogger("framelift.cli")

EXAMPLES = """\
examples:
  # Measure, suggest two presets (fast and best), save a comparison sheet
  framelift tune -i clip.mp4

  # Animation: the "best" preset uses the anime model instead
  framelift tune -i episode.mp4 --anime

  # You already know you want 2x and the old_tv look: tune everything else
  framelift tune -i vhs.mp4 --scale 2 --profile old_tv

  # Just the heuristics and the sheet, no speed test (fast on slow machines)
  framelift tune -i clip.mp4 --no-benchmark

Any setting you pass is kept as-is; everything else is measured or suggested.
The commands printed at the end are ready to copy and run.
"""

# Settings you can pin. Anything given on the command line is left untouched.
_PINNABLE = (
    "model", "device", "gpu_id", "weights_dir", "torch_threads", "profile", "scale",
    "same_resolution", "ai_strength", "tile", "half", "encoder", "start_frame", "max_frames",
)  # fmt: skip


def build_tune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framelift tune",
        description=(
            "Find good settings for a video on this machine. Speed settings (device, "
            "FP16, tile, CPU threads, encoder) are measured on real frames. The AI "
            "strength comes from measuring what the AI actually does to this footage. "
            "Ends with two ready-to-run presets, fast and best, with time estimates."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    files = parser.add_argument_group("video")
    files.add_argument("-i", "--input", metavar="VIDEO", required=True, help="Video to analyse.")
    files.add_argument(
        "-o", "--output", metavar="MP4",
        help="Output name to put in the suggested command. (default: <input>_enhanced.mp4)",
    )  # fmt: skip

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--samples", type=int, default=6, metavar="N",
        help="Frames spread across the video to inspect for noise, softness and motion. "
             "(default: 6)",
    )  # fmt: skip
    analysis.add_argument(
        "--sheet", metavar="PNG",
        help="Where to save the comparison sheet. (default: <input>_tune.png in this folder)",
    )  # fmt: skip
    analysis.add_argument("--no-sheet", action="store_true", help="Don't make a comparison sheet.")
    analysis.add_argument(
        "--no-benchmark", action="store_true",
        help="Skip the speed test. Much faster on CPU, but no tile/FP16/thread tuning and "
             "no time estimates.",
    )  # fmt: skip
    analysis.add_argument(
        "--no-best", action="store_true",
        help="Only tune the fast preset: skips the heavier model (a 67 MB download the "
             "first time) and its tests.",
    )  # fmt: skip
    analysis.add_argument(
        "--anime", action="store_true",
        help="The video is animation: the best preset uses realesrgan-x4plus-anime-6B.",
    )  # fmt: skip

    pinned = parser.add_argument_group(
        "settings to keep (anything you pass here is used as-is, not tuned)"
    )
    pinned.add_argument(
        "--model", choices=list(MODELS), help="Tune only this model (no fast/best choice)."
    )
    pinned.add_argument("--device", choices=DEVICES)
    pinned.add_argument("--gpu-id", type=int, metavar="ID")
    pinned.add_argument("--weights-dir", metavar="DIR")
    pinned.add_argument("--torch-threads", type=int, metavar="N")
    pinned.add_argument("--profile", choices=list(PROFILES))
    pinned.add_argument("--scale", type=float, metavar="FACTOR")
    pinned.add_argument("--same-resolution", action="store_true", default=None)
    pinned.add_argument("--ai-strength", type=float, metavar="N")
    pinned.add_argument("--tile", type=int, metavar="PX")
    pinned.add_argument("--half", action="store_true", default=None)
    pinned.add_argument("--encoder", choices=ENCODERS)
    pinned.add_argument("--start-frame", type=int, metavar="N")
    pinned.add_argument("--max-frames", type=int, metavar="N")

    output = parser.add_argument_group("output")
    output.add_argument("-v", "--verbose", action="count", default=0, help="Show more detail.")
    output.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="Only print the final report (plus warnings).",
    )  # fmt: skip

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_tune_parser()
    args = parser.parse_args(argv)
    setup_console(args.verbose - args.quiet)

    pinned = {name: getattr(args, name) for name in _PINNABLE if getattr(args, name) is not None}
    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_name(f"{input_path.stem}_enhanced.mp4"))
    sheet_path = None if args.no_sheet else Path(args.sheet or f"{input_path.stem}_tune.png")

    try:
        from .tuning import tune_video  # heavy (PyTorch): import only when needed

        log.info("Tuning — this takes a minute or two (longer on CPU). Ctrl+C cancels.")
        report = tune_video(
            input_path,
            EnhanceOptions(**pinned),
            keep=pinned,
            samples=args.samples,
            benchmark=not args.no_benchmark,
            try_best=not args.no_best,
            content="anime" if args.anime else "live",
            sheet_path=sheet_path,
        )
    except KeyboardInterrupt:
        log.warning("Cancelled.")
        return 130
    except FrameliftError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # unexpected: show the traceback only when asked
        log.error("%s", exc, exc_info=args.verbose > 0)
        if not args.verbose:
            log.error("Run again with -v to see the full traceback.")
        return 1

    print(render_report(report, input_path, output_path, pinned))
    return 0


def render_report(report, input_path, output_path, pinned=()) -> str:
    """The human-readable summary printed at the end of ``framelift tune``.

    ``pinned`` holds the names of the settings the user chose themselves.
    """
    pinned = set(pinned)
    video, plan = report.video, report.plan
    options = report.presets["fast"].options
    yours = "  (yours)"
    lines = [""]

    # Machine
    if report.device.is_cuda:
        memory = f" ({report.gpu_memory_gb:.0f} GB)" if report.gpu_memory_gb else ""
        encoder = "NVENC available" if report.nvenc else "NVENC not available"
        machine = f"{report.device.name}{memory} · {encoder}"
    else:
        machine = "CPU (no NVIDIA GPU in use, so expect it to be slow)"
        threads = options.torch_threads or report.default_threads
        if report.default_threads and threads != report.default_threads:
            machine += f"\n  {threads} CPU threads were faster than all {report.default_threads}"
    lines += ["Your machine", f"  {machine}", ""]

    # Video
    minutes, seconds = divmod(round(video.duration_seconds), 60)
    if "profile" in pinned:
        noise_line = f"profile {options.profile}{yours}"
    else:
        noise_line = f"{report.noise_label} ({report.noise:.1f}) → profile {options.profile}"
    if pinned & {"scale", "same_resolution"}:
        size_line = f"{plan.width}x{plan.height}{yours}"
    elif report.size.same_resolution:
        size_line = f"{report.size.reason} → --same-resolution"
    else:
        size_line = f"{report.size.reason} → --scale {report.size.scale:g}"
    lines += [
        "Your video",
        f"  {video.path.name}  {video.width}x{video.height} @ {video.fps:.3f} fps, "
        f"{video.frame_count:,} frames ({minutes}m{seconds:02d}s)",
        f"  Softness  {report.softness_label} ({report.softness:.2f})",
        f"  Noise     {noise_line}",
        f"  Size      {size_line}",
        f"  Tested on frame {report.sample_frame} (the sharpest steady one)",
        "",
    ]

    # What the AI does
    lines.append("What the AI does to it")
    width = max(len(_short(test.model)) for test in report.model_tests)
    for test in report.model_tests:
        effect = test.effect
        lines.append(
            f"  {_short(test.model):<{width}}  sharpness {effect.sharpening:+.2f} · "
            f"noise {_signed(effect.noise_change)} · structure {effect.structure:.2f}"
            f"  → AI strength {test.ai_strength:g}"
        )
        if test.reasons:
            lines.append(f"  {'':<{width}}  {', '.join(test.reasons)}")
    lines.append("")

    # Speed test
    lines.append(f"Speed test ({_short(options.model)}, {plan.width}x{plan.height} output)")
    if not report.runs:
        lines.append("  skipped (--no-benchmark)")
    best_run = report.best_run
    for run in report.runs:
        config = f"tile {run.tile or 'off'}, {'FP16' if run.half else 'FP32'}"
        if run.threads:
            config += f", {run.threads} threads"
        result = f"{run.seconds_per_frame:.2f} s/frame" if run.ok else run.error
        marker = "  ← fastest" if run is best_run else ""
        lines.append(f"  {config:<28} {result}{marker}")
    lines.append("")

    # Presets
    lines.append("Presets")
    for name, preset in report.presets.items():
        details = [_short(preset.options.model), f"AI {preset.options.ai_strength:g}"]
        if preset.estimated_seconds is not None:
            details.append(f"~{_human_duration(preset.estimated_seconds)}")
        marker = "  ← recommended" if name == report.recommended_preset else ""
        lines.append(f"  {name:<5} {' · '.join(details)}  ({preset.description}){marker}")
    if report.estimated_seconds is not None:
        lines.append("  Times are rough: they don't include loading the model or encoding.")
    lines.append("")

    # Sheet
    if report.sheet_path:
        lines += [
            f"Comparison sheet: {report.sheet_path}",
            f"  The same 100% crop of frame {report.sample_frame}: columns are AI strengths",
            "  (the presets' picks are marked), rows are the presets. Adjust to taste.",
            "",
        ]

    lines.append("Commands")
    for name, preset in report.presets.items():
        marker = " (recommended)" if name == report.recommended_preset else ""
        lines += [
            f"  {name}{marker}:",
            f"    {format_command(input_path, _output_for(output_path, name), preset.options)}",
        ]
    return "\n".join(lines)


def _output_for(output_path, preset: str) -> str:
    """``best`` writes to ``<name>_best.mp4`` so both presets can be compared side by side."""
    if preset == "fast":
        return str(output_path)
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{preset}{path.suffix}"))


def _short(model: str) -> str:
    return model.replace("realesrgan-", "").replace("realesr-", "")


def _signed(value: float) -> str:
    return "±0.0" if abs(value) < 0.05 else f"{value:+.1f}"


def _human_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}"
