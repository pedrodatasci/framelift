"""Pieces of the tuner that need PyTorch but no model weights."""

import pytest

pytest.importorskip("torch")

from framelift.tuning import _sheet_columns, thread_candidates  # noqa: E402


@pytest.mark.parametrize(
    ("default", "expected"),
    [(1, [1]), (2, [2, 1]), (4, [4, 2]), (10, [10, 5, 4]), (16, [16, 8, 4])],
)
def test_thread_candidates(default, expected):
    assert thread_candidates(default) == expected


def test_sheet_columns_mark_each_presets_strength():
    from types import SimpleNamespace

    presets = {
        "fast": SimpleNamespace(options=SimpleNamespace(ai_strength=0.55)),
        "best": SimpleNamespace(options=SimpleNamespace(ai_strength=0.7)),
    }
    values, labels = _sheet_columns(presets)
    assert values == [0.0, 0.3, 0.55, 0.7, 1.0]
    assert labels[2].endswith("(fast)") and labels[3].endswith("(best)")
    assert labels[0] == "no AI (plain resize)"
