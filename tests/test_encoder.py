import subprocess

import numpy as np
import pytest

from framelift.encoder import FFmpegWriter, codec_arguments, format_command, temp_path_for
from framelift.errors import EncodingError

from .conftest import needs_ffmpeg


def test_codec_arguments():
    assert codec_arguments("cpu", 20, "veryfast") == [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    ]  # fmt: skip
    assert codec_arguments("nvenc", 19, "ignored") == [
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        codec_arguments("av1", 20, "fast")


def test_temp_path_sits_next_to_the_output(tmp_path):
    assert temp_path_for(tmp_path / "movie.mp4") == tmp_path / "movie.encoding_tmp.mp4"


def test_format_command_quotes_spaces():
    assert format_command(["ffmpeg", "-i", "my clip.mp4"]) == 'ffmpeg -i "my clip.mp4"'


def test_finish_without_frames_is_an_error(tmp_path):
    with pytest.raises(EncodingError):
        FFmpegWriter(tmp_path / "o.mp4", fps=24).finish()


def _count_frames(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return int(out.stdout.strip())


@needs_ffmpeg
def test_writer_round_trip_with_repeats(tmp_path):
    output = tmp_path / "sub dir" / "out.mp4"
    writer = FFmpegWriter(output, fps=30)
    frame = np.full((48, 64, 3), 128, np.uint8)

    writer.write(frame, copies=2)
    writer.write(frame, copies=0)
    writer.write(frame, copies=3)
    assert writer.finish() == output

    assert writer.frames_written == 5
    assert _count_frames(output) == 5
    assert not writer.temp_path.exists()


@needs_ffmpeg
def test_odd_dimensions_are_made_even(tmp_path):
    writer = FFmpegWriter(tmp_path / "odd.mp4", fps=24)
    for _ in range(3):
        writer.write(np.full((47, 63, 3), 90, np.uint8))
    assert _count_frames(writer.finish()) == 3
