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


def test_interpolate_alone_doubles_the_frame_rate():
    plan = plan_run(VIDEO, EnhanceOptions(interpolate=True))
    assert plan.output_fps == 48.0
    assert plan.expected_output_frames == 480


def test_interpolate_keeps_a_chosen_frame_rate():
    assert plan_run(VIDEO, EnhanceOptions(interpolate=True, output_fps=60)).output_fps == 60


def test_interpolate_needs_a_higher_frame_rate():
    with pytest.raises(InvalidOptionError, match="--interpolate needs"):
        plan_run(VIDEO, EnhanceOptions(interpolate=True, output_fps=24))


def test_simple_multiple_finds_the_exact_rate_close_by():
    from framelift.planning import simple_multiple

    assert simple_multiple(29.99, 60) == pytest.approx(59.98)
    assert simple_multiple(23.976, 60) == pytest.approx(59.94)  # 2.5x
    assert simple_multiple(30, 60) == pytest.approx(60)
    assert simple_multiple(25, 60) is None  # 2.4x: nothing simple close by
    assert simple_multiple(30, 30) is None


def test_advice_when_almost_a_multiple_means_much_more_work():
    from framelift.planning import interpolation_advice

    advice = interpolation_advice(29.99, 60, 1635)
    assert "2,835" in advice and "1,634" in advice and "59.980" in advice
    assert "leave --output-fps out" in advice
    assert interpolation_advice(29.99, 59.98, 1635) is None
    assert interpolation_advice(30, 60, 1635) is None
    assert interpolation_advice(29.99, 60, 20) is None  # too short to matter


def test_a_rounded_multiple_is_made_exact_for_interpolation():
    video = VideoInfo(Path("in.mp4"), fps=30000 / 1001, frame_count=240, width=640, height=360)
    typed = EnhanceOptions(interpolate=True, output_fps=59.94)
    assert plan_run(video, typed).output_fps == 60000 / 1001
    # Without --interpolate the rate is used as given (frame repeating doesn't drift).
    assert plan_run(video, EnhanceOptions(output_fps=59.94)).output_fps == 59.94


def test_progress_fraction():
    from framelift.planning import Progress

    assert Progress("frames", 5, 20, 5, 1.0).fraction == 0.25
    assert Progress("loading", 0, 0, 0, 0.0).fraction == 0.0
    assert Progress("frames", 21, 20, 21, 1.0).fraction == 1.0
