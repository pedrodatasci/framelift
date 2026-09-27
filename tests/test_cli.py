import pytest

from framelift import EnhanceOptions
from framelift.cli import build_parser, main, options_from_args

ORIGINAL_COMMAND = (
    "--input a.mp4 --output b.mp4 --device cpu --torch-threads 4 --start-frame 11 "
    "--same-resolution --scale 2 --output-fps 60 --model realesrgan-x4plus "
    "--weights-dir w --profile old_tv --ai-strength 0.3 --tile 128 --tile-pad 8 "
    "--pre-pad 2 --gpu-id 1 --half --encoder nvenc --crf 18 --x264-preset slow "
    "--prefetch 4 --max-frames 50 --debug-timings --timing-every 5 --gc-every 10 "
    "--frame-format png --keep-temp"
).split()


def test_every_original_flag_is_still_accepted():
    args = build_parser().parse_args(ORIGINAL_COMMAND)
    options = options_from_args(args)
    assert options.start_frame == 11
    assert options.model == "realesrgan-x4plus"
    assert options.half and options.same_resolution and options.debug_timings
    assert (options.output_fps, options.crf, options.gc_every) == (60, 18, 10)
    options.validate()


def test_cli_defaults_equal_library_defaults():
    args = build_parser().parse_args(["-i", "a.mp4", "-o", "b.mp4"])
    assert options_from_args(args) == EnhanceOptions()


def test_input_and_output_are_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--list-models", "--list-profiles"])
def test_catalogs_print_and_exit_cleanly(flag, capsys):
    assert main([flag]) == 0
    out = capsys.readouterr().out
    assert "(default)" in out
