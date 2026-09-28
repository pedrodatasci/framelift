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


def test_options_to_args_round_trips():
    from framelift.cli import options_to_args

    options = EnhanceOptions(
        scale=2.25, profile="old_tv", device="cpu", tile=128, half=True, x264_preset="medium"
    )
    args = build_parser().parse_args(["-i", "a", "-o", "b", *options_to_args(options)])
    assert options_from_args(args) == options


def test_default_options_need_no_flags():
    from framelift.cli import options_to_args

    assert options_to_args(EnhanceOptions()) == []


def test_scale_is_dropped_with_same_resolution():
    from framelift.cli import options_to_args

    assert options_to_args(EnhanceOptions(same_resolution=True, scale=3)) == ["--same-resolution"]


def test_format_command_quotes_paths_with_spaces():
    from framelift.cli import format_command

    command = format_command("my clip.mp4", "out.mp4", EnhanceOptions(tile=128))
    assert command == 'framelift -i "my clip.mp4" -o out.mp4 --tile 128'


def test_tune_is_dispatched_as_a_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["tune", "--help"])
    assert exc.value.code == 0
    assert "comparison sheet" in capsys.readouterr().out


def test_tune_requires_an_input():
    with pytest.raises(SystemExit) as exc:
        main(["tune"])
    assert exc.value.code == 2
