"""The ``framelift tune`` report, rendered from a hand-made report (no PyTorch needed)."""

from pathlib import Path
from types import SimpleNamespace

from framelift import EnhanceOptions, VideoInfo, plan_run
from framelift.analysis import AIEffect, suggest_size
from framelift.tune_cli import _human_duration, _signed, build_tune_parser, render_report

VIDEO = VideoInfo(Path("clip.mp4"), fps=30.0, frame_count=900, width=640, height=360)


def _preset(name, model, strength, seconds, **extra):
    options = EnhanceOptions(
        model=model, scale=3, ai_strength=strength, device="cpu", tile=128, **extra
    )
    return SimpleNamespace(
        name=name,
        options=options,
        seconds_per_frame=None if seconds is None else seconds / 900,
        estimated_seconds=seconds,
        description="light video model" if name == "fast" else "heavier model",
    )


def _report(**changes):
    fast = _preset("fast", "realesr-animevideov3", 0.55, 450.0, torch_threads=4)
    best = _preset("best", "realesrgan-x4plus", 0.5, 12150.0, crf=18)
    run = SimpleNamespace(tile=128, half=False, threads=None, seconds_per_frame=0.6, ok=True,
                          error=None)  # fmt: skip
    threads = SimpleNamespace(tile=128, half=False, threads=4, seconds_per_frame=0.5, ok=True,
                              error=None)  # fmt: skip
    oom = SimpleNamespace(tile=256, half=False, threads=None, seconds_per_frame=None, ok=False,
                          error="out of memory")  # fmt: skip
    fields = dict(
        video=VIDEO,
        plan=plan_run(VIDEO, fast.options),
        device=SimpleNamespace(is_cuda=False, name="CPU"),
        gpu_memory_gb=None,
        nvenc=False,
        default_threads=10,
        noise=0.4,
        noise_label="very clean",
        softness=0.47,
        softness_label="soft",
        size=suggest_size(640, 360),
        runs=[oom, run, threads],
        best_run=threads,
        model_tests=[
            SimpleNamespace(
                model="realesr-animevideov3",
                effect=AIEffect(0.1, -0.27, 0.95),
                ai_strength=0.55,
                reasons=["recovers some sharpness"],
            ),
            SimpleNamespace(
                model="realesrgan-x4plus",
                effect=AIEffect(0.17, 0.38, 0.96),
                ai_strength=0.5,
                reasons=["invents grain or texture"],
            ),
        ],  # fmt: skip
        presets={"fast": fast, "best": best},
        recommended_preset="fast",
        estimated_seconds=450.0,
        sample_frame=450,
        sheet_path=Path("clip_tune.png"),
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_report_explains_the_measurements():
    text = render_report(_report(), "clip.mp4", "clip_hd.mp4")
    assert "Softness  soft (0.47)" in text
    assert "very clean (0.4) → profile none" in text
    assert "360p → 1080p → --scale 3" in text
    assert "4 CPU threads were faster than all 10" in text
    assert "animevideov3  sharpness +0.10 · noise -0.3 · structure 0.95  → AI strength 0.55" in text
    assert "invents grain or texture" in text
    assert "out of memory" in text
    assert "0.50 s/frame  ← fastest" in text


def test_report_lists_both_presets_with_commands():
    text = render_report(_report(), "clip.mp4", "clip_hd.mp4")
    assert "fast  animevideov3 · AI 0.55 · ~8 min" in text
    assert "← recommended" in text
    assert "best  x4plus · AI 0.5 · ~3h22" in text
    assert "fast (recommended):" in text
    assert "--model realesrgan-x4plus" in text.split("best:")[1]
    assert "-o clip_hd_best.mp4" in text.split("best:")[1]
    assert "-o clip_hd.mp4" in text.split("fast (recommended):")[1].split("best:")[0]
    assert "clip_tune.png" in text


def test_pinned_settings_are_labelled():
    text = render_report(_report(), "clip.mp4", "out.mp4", pinned={"profile": "none"})
    assert "profile none  (yours)" in text


def test_report_without_benchmark():
    fast = _preset("fast", "realesr-animevideov3", 0.55, None)
    text = render_report(
        _report(runs=[], best_run=None, estimated_seconds=None, presets={"fast": fast}),
        "clip.mp4",
        "out.mp4",
    )
    assert "skipped (--no-benchmark)" in text
    assert "Times are rough" not in text
    assert "best" not in text.split("Presets")[1].split("Commands")[0]


def test_formatting_helpers():
    assert [_human_duration(s) for s in (0.2, 45, 600, 7300)] == ["1 s", "45 s", "10 min", "2h02"]
    assert [_signed(v) for v in (0.02, -0.3, 0.38)] == ["±0.0", "-0.3", "+0.4"]


def test_parser_defaults():
    args = build_tune_parser().parse_args(["-i", "clip.mp4", "--profile", "old_tv", "--anime"])
    assert args.profile == "old_tv" and args.anime and not args.no_best
    assert args.samples == 6
    assert args.scale is None and args.half is None and args.same_resolution is None
