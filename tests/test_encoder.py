import os
import subprocess

import numpy as np
import pytest

from framelift import encoder
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


def test_without_the_ai_nvenc_is_simply_tried(monkeypatch):
    """--no-ai has no device to go by: NVENC is probed, with the usual fallback."""
    monkeypatch.setattr(encoder, "nvenc_available", lambda: True)
    assert encoder.select_encoder("nvenc", None) == "nvenc"
    assert encoder.select_encoder("cpu", None) == "cpu"
    monkeypatch.setattr(encoder, "nvenc_available", lambda: False)
    assert encoder.select_encoder("nvenc", None) == "cpu"


def _fake_program(folder, name):
    """An empty executable that ``shutil.which`` will find."""
    folder.mkdir(exist_ok=True)
    path = folder / (name + ".exe" if os.name == "nt" else name)
    path.write_text("")
    path.chmod(0o755)
    return path


def test_ffmpeg_folder_from_the_environment(tmp_path, monkeypatch):
    folder = tmp_path / "ffmpeg-7" / "bin"
    folder.parent.mkdir()
    ffmpeg, ffprobe = _fake_program(folder, "ffmpeg"), _fake_program(folder, "ffprobe")

    def same(found, expected):  # Windows may answer "ffmpeg.EXE" for "ffmpeg.exe"
        return os.path.samefile(found, expected)

    monkeypatch.setenv("FRAMELIFT_FFMPEG", str(folder))
    assert same(encoder.find_binary("ffmpeg"), ffmpeg)
    assert same(encoder.find_binary("ffprobe"), ffprobe)
    # Pointing at ffmpeg itself works too: its folder is used.
    monkeypatch.setenv("FRAMELIFT_FFMPEG", str(ffmpeg))
    assert same(encoder.find_binary("ffprobe"), ffprobe)


def test_ffmpeg_folder_without_the_program_says_so(tmp_path, monkeypatch):
    from framelift.errors import MissingDependencyError

    monkeypatch.setenv("FRAMELIFT_FFMPEG", str(tmp_path))
    with pytest.raises(MissingDependencyError, match="FRAMELIFT_FFMPEG is set to"):
        encoder.find_binary("ffprobe")


@pytest.mark.parametrize(("platform", "hint"), [("win32", "winget install Gyan.FFmpeg"),
                                                ("darwin", "brew install ffmpeg"),
                                                ("linux", "sudo apt install ffmpeg")])  # fmt: skip
def test_missing_ffmpeg_says_how_to_install_it(monkeypatch, platform, hint):
    from framelift.errors import MissingDependencyError

    monkeypatch.delenv("FRAMELIFT_FFMPEG", raising=False)
    monkeypatch.setattr(encoder.shutil, "which", lambda name, path=None: None)
    monkeypatch.setattr(encoder.sys, "platform", platform)
    with pytest.raises(MissingDependencyError, match="Couldn't find 'ffmpeg'") as error:
        encoder.find_binary("ffmpeg")
    assert hint in str(error.value) and "FRAMELIFT_FFMPEG" in str(error.value)


def test_ffmpeg_folder_that_does_not_exist_says_so(tmp_path, monkeypatch):
    from framelift.errors import MissingDependencyError

    monkeypatch.setenv("FRAMELIFT_FFMPEG", str(tmp_path / "nowhere"))
    with pytest.raises(MissingDependencyError, match="which doesn't exist"):
        encoder.find_binary("ffmpeg")
