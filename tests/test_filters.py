import numpy as np
import pytest

from framelift.filters import PROFILES, adjust_tone, apply_profile, get_profile, unsharp_mask


def test_none_profile_returns_the_same_object(frame):
    assert apply_profile(frame, "none") is frame
    assert get_profile("none").is_passthrough


@pytest.mark.parametrize("name", [n for n in PROFILES if n != "none"])
def test_every_profile_keeps_shape_and_changes_pixels(frame, name):
    out = apply_profile(frame, name)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)
    assert not get_profile(name).is_passthrough


def test_profiles_have_descriptions():
    assert all(profile.description for profile in PROFILES.values())


def test_unknown_profile_lists_the_options():
    with pytest.raises(ValueError, match="old_tv"):
        get_profile("vhs")


def test_unsharp_with_zero_amount_is_identity(frame):
    assert unsharp_mask(frame, amount=0) is frame


def test_adjust_tone_neutral_settings_keep_pixels(frame):
    np.testing.assert_array_equal(adjust_tone(frame), frame)


def test_brightness_goes_up():
    dark = np.full((8, 8, 3), 50, np.uint8)
    assert adjust_tone(dark, brightness=20).mean() == pytest.approx(70)
