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


def test_tiled_area_counts_the_overlaps():
    from framelift.tuning import tiled_area

    assert tiled_area(720, 480, 0, 10) == 720 * 480
    # 720 → tiles of 266, 276 and 218 px with their overlap; 480 → 266 and 234.
    assert tiled_area(720, 480, 256, 10) == (266 + 276 + 218) * (266 + 234)
    assert tiled_area(720, 480, 256, 10, pre_pad=10) == (266 + 276 + 228) * (266 + 244)
    assert tiled_area(64, 48, 256, 10) == 64 * 48  # one tile covers it all


def test_intel_gpu_needs_openvino(monkeypatch):
    import framelift.device as devices
    from framelift import InvalidOptionError

    monkeypatch.setattr(devices, "intel_gpu_name", lambda: None)
    with pytest.raises(InvalidOptionError, match="OpenVINO"):
        devices.resolve_device("igpu")
    monkeypatch.setattr(devices, "intel_gpu_name", lambda: "Intel(R) Iris(R) Xe Graphics")
    device = devices.resolve_device("igpu")
    assert device.kind == "igpu" and device.is_gpu and not device.is_cuda


def test_threads_are_split_between_workers():
    from framelift import EnhanceOptions
    from framelift.device import DEFAULT_CPU_THREADS, Device
    from framelift.pipeline import _threads_per_worker

    cpu = Device("cpu", "CPU")
    assert _threads_per_worker(EnhanceOptions(), cpu) == 0
    assert _threads_per_worker(EnhanceOptions(workers=2, torch_threads=3), cpu) == 3
    assert _threads_per_worker(EnhanceOptions(workers=2), cpu) == max(1, DEFAULT_CPU_THREADS // 2)
