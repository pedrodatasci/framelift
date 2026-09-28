"""The ``framelift tune`` report, rendered from a hand-made report (no PyTorch needed)."""

from pathlib import Path
from types import SimpleNamespace

from framelift import EnhanceOptions, VideoInfo, plan_run
from framelift.analysis import suggest_size
from framelift.tune_cli import _human_duration, build_tune_parser, render_report

VIDEO = VideoInfo(Path("clip.mp4"), fps=30.0, frame_count=900, width=640, height=360)


def _report(**changes):
    recommended = EnhanceOptions(scale=3, profile="soft_camera", device="cpu", tile=128)
    run = SimpleNamespace(tile=128, half=False, seconds_per_frame=0.5, ok=True, error=None)
    oom = SimpleNamespace(tile=256, half=False, seconds_per_frame=None, ok=False,
                          error="out of memory")  # fmt: skip
    fields = dict(
        video=VIDEO,
        plan=plan_run(VIDEO, recommended),
        recommended=recommended,
        device=SimpleNamespace(is_cuda=False, name="CPU"),
        gpu_memory_gb=None,
        nvenc=False,
        noise=3.2,
        noise_label="light grain",
        size=suggest_size(640, 360),
        runs=[oom, run],
        best_run=run,
        preprocess_seconds=0.01,
        estimated_seconds=450.0,
        sample_frame=450,
        sheet_path=Path("clip_tune.png"),
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_report_contains_the_essentials():
    text = render_report(_report(), "clip.mp4", "clip_hd.mp4")
    assert "light grain (noise 3.2) → profile soft_camera" in text
    assert "360p → 1080p → --scale 3" in text
    assert "out of memory" in text
    assert "0.50 s/frame  ← fastest" in text
    assert "~8 min for 900 frames" in text
    assert "clip_tune.png" in text
    assert text.rstrip().endswith(
        "framelift -i clip.mp4 -o clip_hd.mp4 "
        "--scale 3 --profile soft_camera --device cpu --tile 128"
    )


def test_pinned_settings_are_labelled():
    text = render_report(_report(), "clip.mp4", "out.mp4", pinned={"profile": "soft_camera"})
    assert "soft_camera  (yours)" in text


def test_report_without_benchmark():
    text = render_report(
        _report(runs=[], best_run=None, estimated_seconds=None), "clip.mp4", "out.mp4"
    )
    assert "skipped (--no-benchmark)" in text
    assert "Estimate" not in text


def test_human_duration():
    assert [_human_duration(s) for s in (0.2, 45, 600, 7300)] == ["1 s", "45 s", "10 min", "2h02"]


def test_parser_leaves_unpinned_settings_as_none():
    args = build_tune_parser().parse_args(["-i", "clip.mp4", "--profile", "old_tv"])
    assert args.profile == "old_tv"
    assert args.scale is None and args.half is None and args.same_resolution is None
