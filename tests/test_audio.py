"""Carrying the audio over. Needs FFmpeg, not PyTorch."""

import shutil
import subprocess

import numpy as np
import pytest

from framelift import audio as audio_module
from framelift.audio import add_audio, probe_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")

BEEP_AT = 2.0  # seconds, in the source


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def source(tmp_path):
    """A 4 s clip at 30 fps with silence and a 50 ms beep at exactly 2.000 s."""
    path = tmp_path / "source.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30",
        "-f", "lavfi", "-i", f"aevalsrc='if(between(t,{BEEP_AT},{BEEP_AT + 0.05}),"
        "sin(2*PI*1000*t),0)':s=48000:c=stereo",
        "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )  # fmt: skip
    return path


def _silent_part(source, tmp_path, start, duration, name="video.mp4"):
    """What framelift writes before the audio step: the enhanced stretch, no audio."""
    path = tmp_path / name
    _ffmpeg("-ss", str(start), "-i", str(source), "-t", str(duration), "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))  # fmt: skip
    return path


def _beep_time(path):
    pcm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:a",
         # Pad with silence from time zero, so an audio track that starts a few ms
         # late is measured where a player would actually play it.
         "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-f", "s16le", "-ar", "48000", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.abs(np.frombuffer(pcm, np.int16).astype(float))
    return np.argmax(samples > 0.3 * samples.max()) / 48000, len(samples) / 48000


def _codecs(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return sorted(line.strip() for line in out.splitlines())


@pytest.mark.parametrize("start", [0.0, 1.0])
def test_audio_is_copied_in_sync(source, tmp_path, start):
    video = _silent_part(source, tmp_path, start, 2.5)

    assert add_audio(video, source, start_seconds=start, duration_seconds=2.5) == "copied"

    beep, length = _beep_time(video)
    assert beep == pytest.approx(BEEP_AT - start, abs=0.005)
    assert length == pytest.approx(2.5, abs=0.03)
    assert _codecs(video) == ["aac,audio", "h264,video"]


def test_late_starting_audio_keeps_its_offset(source, tmp_path):
    late = tmp_path / "late.mp4"
    _ffmpeg("-i", str(source), "-itsoffset", "0.5", "-i", str(source),
            "-map", "0:v", "-map", "1:a", "-c", "copy", str(late))  # fmt: skip
    video = _silent_part(late, tmp_path, 1.0, 2.0)

    add_audio(video, late, start_seconds=1.0, duration_seconds=2.0)

    assert _beep_time(video)[0] == pytest.approx(BEEP_AT + 0.5 - 1.0, abs=0.005)


def test_codecs_players_dislike_in_mp4_are_converted(source, tmp_path):
    pcm = tmp_path / "pcm.mov"
    _ffmpeg("-i", str(source), "-c:v", "copy", "-c:a", "pcm_s16le", str(pcm))
    video = _silent_part(pcm, tmp_path, 0, 2.0)

    assert add_audio(video, pcm, 0.0, 2.0) == "converted"
    assert "aac,audio" in _codecs(video)


def test_input_without_audio(source, tmp_path):
    silent = tmp_path / "silent.mp4"
    _ffmpeg("-i", str(source), "-an", "-c", "copy", str(silent))
    video = _silent_part(silent, tmp_path, 0, 2.0)

    assert probe_audio(silent) is None
    assert add_audio(video, silent, 0.0, 2.0) == "none"
    assert _codecs(video) == ["h264,video"]


def test_missing_ffprobe_keeps_the_silent_video(source, tmp_path, monkeypatch):
    video = _silent_part(source, tmp_path, 0, 2.0)
    before = video.read_bytes()
    monkeypatch.setattr(audio_module.shutil, "which", lambda name: None)

    assert add_audio(video, source, 0.0, 2.0) == "failed"
    assert video.read_bytes() == before


def test_broken_source_keeps_the_silent_video(source, tmp_path):
    video = _silent_part(source, tmp_path, 0, 2.0)
    before = video.read_bytes()
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(source.read_bytes()[:2000])  # truncated file

    assert add_audio(video, broken, 0.0, 2.0) in {"none", "failed"}
    assert video.read_bytes() == before
    assert not list(tmp_path.glob("*.audio_tmp.mp4"))


def test_files_whose_clock_does_not_start_at_zero(source, tmp_path):
    transport = tmp_path / "capture.ts"  # MPEG-TS clocks usually start around 1.4 s
    _ffmpeg("-i", str(source), "-c", "copy", str(transport))
    assert probe_audio(transport).video_start > 1
    video = _silent_part(source, tmp_path, 1.0, 2.0)

    add_audio(video, transport, start_seconds=1.0, duration_seconds=2.0)

    assert _beep_time(video)[0] == pytest.approx(BEEP_AT - 1.0, abs=0.005)


def test_parts_can_be_joined_without_losing_sync(source, tmp_path):
    """FFmpeg's concat ignores "skip this pre-roll" hints, so the cut must not need them."""
    first = _silent_part(source, tmp_path, 0.0, 1.5, "first.mp4")
    second = _silent_part(source, tmp_path, 1.5, 1.5, "second.mp4")
    add_audio(first, source, 0.0, 1.5)
    add_audio(second, source, 1.5, 1.5)

    parts = tmp_path / "parts.txt"
    parts.write_text(f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n")
    joined = tmp_path / "joined.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(parts), "-c", "copy", str(joined))

    assert _beep_time(joined)[0] == pytest.approx(BEEP_AT, abs=0.005)
