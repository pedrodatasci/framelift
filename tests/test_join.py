"""framelift join: the parts of a resumed run back into one video, checked first."""

import logging
import subprocess
from pathlib import Path

import pytest

from framelift.cli import _report, main
from framelift.errors import InvalidOptionError
from framelift.join import all_parts, join_videos, joined_path, next_part_path
from framelift.planning import EnhanceResult

from .conftest import needs_ffmpeg


def test_part_names():
    assert next_part_path("concert.mp4") == Path("concert_part2.mp4")
    assert next_part_path("videos/concert_part2.mp4") == Path("videos/concert_part3.mp4")
    assert all_parts("concert_part3.mp4") == [
        Path("concert.mp4"), Path("concert_part2.mp4"), Path("concert_part3.mp4"),
    ]  # fmt: skip
    assert joined_path("concert_part3.mp4") == Path("concert_full.mp4")


def _clip(path, seconds, size="160x120", audio=True):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    command += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac"] if audio else []
    subprocess.run([*command, str(path)], check=True)
    return path


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True)  # fmt: skip
    return float(out.stdout)


@needs_ffmpeg
def test_parts_are_joined_in_order(tmp_path):
    # An accent and a quote in the names: FFmpeg's list needs both handled.
    first = _clip(tmp_path / "Pedro Sá's part.mp4", 1.0)
    second = _clip(tmp_path / "part2.mp4", 0.5)
    result = join_videos([first, second], tmp_path / "full.mp4")
    assert (result.parts, result.has_audio) == (2, True)
    assert result.duration_seconds == pytest.approx(1.5, abs=0.1)
    assert _duration(tmp_path / "full.mp4") == pytest.approx(1.5, abs=0.1)
    assert not list(tmp_path.glob("*encoding_tmp*"))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("second", "message"),
    [({"size": "320x240"}, "width 160 vs 320"), ({"audio": False}, "one has audio")],
)
def test_parts_made_differently_are_refused(tmp_path, second, message):
    first = _clip(tmp_path / "a.mp4", 0.5)
    other = _clip(tmp_path / "b.mp4", 0.5, **second)
    with pytest.raises(InvalidOptionError, match=message):
        join_videos([first, other], tmp_path / "full.mp4")
    assert not (tmp_path / "full.mp4").exists()


@needs_ffmpeg
def test_join_refuses_what_would_lose_work(tmp_path):
    first, second = _clip(tmp_path / "a.mp4", 0.5), _clip(tmp_path / "b.mp4", 0.5)
    with pytest.raises(InvalidOptionError, match="one of the parts"):
        join_videos([first, second], first)
    with pytest.raises(InvalidOptionError, match="at least two"):
        join_videos([first], tmp_path / "full.mp4")
    with pytest.raises(InvalidOptionError, match="Can't find"):
        join_videos([first, tmp_path / "missing.mp4"], tmp_path / "full.mp4")


@needs_ffmpeg
def test_join_command(tmp_path):
    first, second = _clip(tmp_path / "a.mp4", 0.5), _clip(tmp_path / "b.mp4", 0.5)
    assert main(["join", str(first), str(second), "-o", str(tmp_path / "full.mp4")]) == 0
    assert (tmp_path / "full.mp4").exists()
    assert main(["join", str(first), "-o", str(tmp_path / "x.mp4")]) == 1


def test_resuming_into_the_finished_part_is_refused(tmp_path):
    done = tmp_path / "concert_hd.mp4"
    done.write_bytes(b"hours of work")
    code = main(["-i", str(tmp_path / "concert.mp4"), "-o", str(done), "--start-frame", "10481"])
    assert code == 1
    assert done.read_bytes() == b"hours of work"


def test_the_resume_hint_names_the_next_part_and_the_join(caplog):
    caplog.set_level(logging.INFO, logger="framelift.cli")
    result = EnhanceResult(output_path=Path("concert_hd.mp4"), first_frame=1, last_frame=10480,
                           frames_processed=10480, frames_written=10480, total_frames=20000,
                           interrupted=True)  # fmt: skip
    _report(result)
    text = caplog.text
    assert "-o concert_hd_part2.mp4 --start-frame 10481" in text
    assert "framelift join concert_hd.mp4 concert_hd_part2.mp4 -o concert_hd_full.mp4" in text
