from pathlib import Path

import pytest

from framelift import EnhanceOptions, InvalidOptionError


def test_defaults_match_the_original_script():
    o = EnhanceOptions()
    assert (o.model, o.scale, o.ai_strength) == ("realesr-animevideov3", 1.5, 0.45)
    assert o.profile == "none"
    assert (o.device, o.tile, o.tile_pad, o.pre_pad, o.gpu_id) == ("auto", 256, 10, 0, 0)
    assert (o.encoder, o.crf, o.x264_preset, o.prefetch) == ("cpu", 20, "veryfast", 2)
    assert (o.start_frame, o.max_frames, o.output_fps) == (1, None, None)
    assert (o.timing_every, o.gc_every, o.weights_dir) == (20, 0, None)
    assert o.audio is True and o.ai is True
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


def test_workers_and_backend_are_validated():
    EnhanceOptions(workers=2, backend="openvino", device="igpu").validate()
    for bad in ({"workers": 0}, {"workers": 99}, {"backend": "onnx"}):
        with pytest.raises(InvalidOptionError):
            EnhanceOptions(**bad).validate()


@pytest.mark.parametrize(
    ("options", "uses_ai"),
    [({}, True), ({"ai": False}, False), ({"ai_strength": 0}, False), ({"ai_strength": -1}, False)],
)
def test_no_ai_and_zero_strength_both_turn_the_ai_off(options, uses_ai):
    assert EnhanceOptions(**options).uses_ai is uses_ai


def test_weights_go_to_the_user_cache_by_default(tmp_path, monkeypatch):
    import sys

    from framelift.models import default_weights_dir, resolve_weights_dir

    monkeypatch.chdir(tmp_path)  # no ./weights here
    monkeypatch.delenv("FRAMELIFT_WEIGHTS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    expected = {"win32": tmp_path / "local", "darwin": Path.home() / "Library" / "Caches"}
    base = expected.get(sys.platform, tmp_path / "cache")
    assert default_weights_dir() == base / "framelift" / "weights"

    # An explicit folder wins, made absolute.
    assert resolve_weights_dir("models") == tmp_path / "models"


def test_weights_from_older_versions_stay_where_they_are(tmp_path, monkeypatch):
    from framelift.models import default_weights_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRAMELIFT_WEIGHTS", raising=False)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "notes.txt").write_text("not a model")
    assert default_weights_dir() != Path("weights")  # a folder without models doesn't count
    (tmp_path / "weights" / "realesr-animevideov3.pth").write_bytes(b"")
    assert default_weights_dir() == Path("weights")


def test_the_environment_variable_picks_the_folder(tmp_path, monkeypatch):
    from framelift.models import default_weights_dir

    monkeypatch.setenv("FRAMELIFT_WEIGHTS", str(tmp_path / "shared"))
    assert default_weights_dir() == tmp_path / "shared"
