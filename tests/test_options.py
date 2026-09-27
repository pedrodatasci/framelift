import pytest

from framelift import EnhanceOptions, InvalidOptionError


def test_defaults_match_the_original_script():
    o = EnhanceOptions()
    assert (o.model, o.scale, o.ai_strength) == ("realesr-animevideov3", 1.5, 0.45)
    assert o.profile == "none"
    assert (o.device, o.tile, o.tile_pad, o.pre_pad, o.gpu_id) == ("auto", 256, 10, 0, 0)
    assert (o.encoder, o.crf, o.x264_preset, o.prefetch) == ("cpu", 20, "veryfast", 2)
    assert (o.start_frame, o.max_frames, o.output_fps) == (1, None, None)
    assert (o.timing_every, o.gc_every, o.weights_dir) == (20, 0, "weights")
    o.validate()


def test_effective_scale_respects_same_resolution():
    assert EnhanceOptions(scale=3).effective_scale == 3
    assert EnhanceOptions(scale=3, same_resolution=True).effective_scale == 1.0


@pytest.mark.parametrize(("value", "expected"), [(-1, 0.0), (0.3, 0.3), (1.5, 1.0)])
def test_ai_strength_is_clamped(value, expected):
    assert EnhanceOptions(ai_strength=value).clamped_ai_strength == expected


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "nope"},
        {"profile": "nope"},
        {"device": "tpu"},
        {"encoder": "av1"},
        {"x264_preset": "placebo"},
        {"start_frame": 0},
        {"max_frames": 0},
        {"scale": 0},
        {"output_fps": -30},
        {"tile": -1},
        {"gpu_id": -1},
    ],
)
def test_validate_rejects_impossible_values(bad):
    with pytest.raises(InvalidOptionError):
        EnhanceOptions(**bad).validate()


def test_scale_is_irrelevant_with_same_resolution():
    EnhanceOptions(scale=0, same_resolution=True).validate()
