from pathlib import Path

import pytest

from framelift import EnhanceOptions, EnhanceResult, InvalidOptionError, VideoInfo, plan_run

VIDEO = VideoInfo(Path("in.mp4"), fps=24.0, frame_count=240, width=640, height=360)


def test_default_plan():
    plan = plan_run(VIDEO, EnhanceOptions())
    assert (plan.width, plan.height) == (960, 540)
    assert (plan.start_frame, plan.frames_to_process, plan.last_frame) == (1, 240, 240)
    assert plan.output_fps == 24.0
    assert not plan.duplicates_frames


def test_same_resolution_keeps_size():
    plan = plan_run(VIDEO, EnhanceOptions(same_resolution=True, scale=4))
    assert (plan.width, plan.height, plan.scale) == (640, 360, 1.0)


def test_start_and_max_frames():
    plan = plan_run(VIDEO, EnhanceOptions(start_frame=201, max_frames=100))
    assert (plan.frames_to_process, plan.last_frame) == (40, 240)


def test_higher_output_fps():
    plan = plan_run(VIDEO, EnhanceOptions(output_fps=60))
    assert plan.duplicates_frames
    assert plan.expected_output_frames == 600


def test_lower_output_fps_is_rejected_with_advice():
    with pytest.raises(InvalidOptionError, match="Leave --output-fps out"):
        plan_run(VIDEO, EnhanceOptions(output_fps=12))


def test_start_past_the_end_is_rejected():
    with pytest.raises(InvalidOptionError, match="240 frames"):
        plan_run(VIDEO, EnhanceOptions(start_frame=241))


def _result(last_frame, total=240):
    return EnhanceResult(Path("o.mp4"), 1, last_frame, last_frame, last_frame, total, False)


def test_result_resume_hint():
    assert _result(100).resume_from == 101
    assert not _result(100).is_complete
    assert _result(240).resume_from is None
    assert _result(240).is_complete
