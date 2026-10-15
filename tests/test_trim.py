import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import pytest

from framelift import FrameRange, InvalidOptionError, VideoInfo, probe_video, trim_video
from framelift.trim import default_trim_path, format_time, parse_time, range_from_times

from .conftest import needs_ffmpeg

VIDEO = VideoInfo(Path("clip.mp4"), fps=25.0, frame_count=250, width=64, height=48)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90", 90), ("90.5", 90.5), ("1:30", 90), ("01:02:03.25", 3723.25), (" 0:05 ", 5)],
)
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", ["", "abc", "1:2:3:4", "-5", "1,5"])
def test_parse_time_rejects_nonsense(bad):
    with pytest.raises(InvalidOptionError):
        parse_time(bad)


def test_format_time():
    assert format_time(83.5) == "1:23.500"
    assert format_time(3723) == "1:02:03.000"
    assert format_time(-1) == "0:00.000"


def test_frame_range_basics():
    frames = FrameRange(26, 75)
    assert frames.frames == 50
    assert frames.start_seconds(25) == 1.0 and frames.end_seconds(25) == 3.0
    assert frames.describe(25) == "0:01.000 → 0:03.000 (2.0 s, frames 26–75)"


@pytest.mark.parametrize("bad", [(0, 5), (10, 9)])
def test_frame_range_rejects_impossible_ranges(bad):
    with pytest.raises(InvalidOptionError):
        FrameRange(*bad)


def test_range_from_times_picks_the_frames_on_screen():
    assert range_from_times(VIDEO, 1.0, 3.0) == FrameRange(26, 75)
    assert range_from_times(VIDEO, 1.01, 2.99) == FrameRange(26, 75)
    assert range_from_times(VIDEO, None, None) == FrameRange(1, 250)
    assert range_from_times(VIDEO, 9.0, 60) == FrameRange(226, 250)  # clamped to the end


@pytest.mark.parametrize(("start", "end"), [(20.0, None), (3.0, 2.0), (3.0, 3.0)])
def test_range_from_times_rejects_empty_ranges(start, end):
    with pytest.raises(InvalidOptionError):
        range_from_times(VIDEO, start, end)


def test_default_trim_path():
    assert default_trim_path("/v/clip.mkv", FrameRange(301, 2565)) == Path(
        "/v/clip_trim_301-2565.mp4"
    )


def _frame_hashes(path):
    capture = cv2.VideoCapture(str(path))
    hashes = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        hashes.append(hashlib.md5(image.tobytes()).hexdigest())
    capture.release()
    return hashes


@pytest.fixture
def h264_with_audio(tmp_path):
    """A 2 s, 30 fps H.264 clip with keyframes far apart and an AAC beep track."""
    path = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2",
         "-c:v", "libx264", "-g", "250", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_trim_is_frame_exact_and_lossless(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    result = trim_video(h264_with_audio, output, FrameRange(17, 41), show_progress=False)

    assert result.path == output.resolve() and output.exists()
    assert result.frames == 25 and result.audio
    assert probe_video(output).frame_count == 25
    # Same pixels as frames 17–41 of the original, even though they're between keyframes.
    assert _frame_hashes(output) == _frame_hashes(h264_with_audio)[16:41]


@needs_ffmpeg
def test_trim_keeps_the_audio_in_sync(h264_with_audio, tmp_path):
    output = tmp_path / "cut.mp4"
    trim_video(h264_with_audio, output, FrameRange(31, 45), show_progress=False)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,start_time,"
         "duration", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams = {s["codec_type"]: s for s in json.loads(probe)["streams"]}
    assert streams["audio"]["codec_name"] == "alac"
    # Video and audio start together (the video a hair later, by the seek margin) …
    assert float(streams["video"]["start_time"]) - float(streams["audio"]["start_time"]) < 0.002
    # … and the audio lasts as long as the 15 frames.
    assert float(streams["audio"]["duration"]) == pytest.approx(15 / 30, abs=0.03)


@needs_ffmpeg
def test_trim_without_audio_and_range_past_the_end(make_video, tmp_path):
    result = trim_video(make_video(frames=12), tmp_path / "c.mp4", FrameRange(10, 99),
                        show_progress=False)  # fmt: skip
    assert result.frame_range == FrameRange(10, 12)
    assert result.frames == 3 and not result.audio


@needs_ffmpeg
def test_trim_refuses_to_overwrite_its_input(make_video):
    clip = make_video(frames=4)
    with pytest.raises(InvalidOptionError):
        trim_video(clip, clip, FrameRange(1, 2), show_progress=False)


@needs_ffmpeg
def test_trim_falls_back_to_decoding_when_seeking_misses(h264_with_audio, tmp_path):
    """MPEG-TS with a single keyframe: seeking lands mid-GOP, so the cut decodes from the
    start instead, and is still exact."""
    stream = tmp_path / "capture.ts"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(h264_with_audio),
                    "-c", "copy", "-f", "mpegts", str(stream)], check=True)  # fmt: skip
    result = trim_video(stream, tmp_path / "cut.mp4", FrameRange(31, 45), show_progress=False)
    assert result.frames == 15
    assert _frame_hashes(result.path) == _frame_hashes(stream)[30:45]


@needs_ffmpeg
def test_trim_command_by_time(h264_with_audio, capsys):
    from framelift.cli import main

    assert main(["trim", "-i", str(h264_with_audio), "--start", "0:00.5", "--end", "1"]) == 0
    cut = h264_with_audio.with_name("source_trim_16-30.mp4")
    assert probe_video(cut).frame_count == 15
    out = capsys.readouterr().out
    assert "framelift tune -i" in out and str(cut) in out


@needs_ffmpeg
def test_trim_command_rejects_a_bad_time(h264_with_audio):
    from framelift.cli import main

    assert main(["trim", "-i", str(h264_with_audio), "--start", "soon"]) == 1
