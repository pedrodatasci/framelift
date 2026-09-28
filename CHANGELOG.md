# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-09-28

### Added

- **`framelift tune`**: finds good settings for a video on the current machine and ends
  with a ready-to-run command.
  - Speed settings are *measured*: a real frame of the video is upscaled with each
    candidate tile size (plus FP16 on NVIDIA, double-checked against FP32 because FP16 is
    slower on some older cards). Configurations that run out of memory are skipped. NVENC
    is probed, and a rough total time estimate is shown.
  - Look settings are *suggested*: a noise estimate (Immerkær's method, ignoring edges so
    texture isn't mistaken for noise, calibrated on H.264 output) picks a profile, and the
    resolution picks a scale that aims for Full HD without stretching more than 3x.
  - A comparison sheet PNG shows the same 100% crop of the most detailed area with
    different AI strengths and profiles, so taste is decided by eye.
  - Any setting passed on the command line is kept as-is and carried into the command.
    `--no-benchmark` skips the speed test on slow machines.
- Python API: `tune_video()` and `TuneReport`, plus the torch-free helpers
  `estimate_noise()` and `suggest_size()`.
- `framelift.cli.format_command()` / `options_to_args()` turn `EnhanceOptions` back into
  a command line.
- The GitHub Actions workflow can also be started by hand ("Run workflow" button).

### Unchanged

- `framelift -i … -o …` and `python main.py …` behave exactly as before; enhanced output
  is still bit-identical to the original script.

## [1.0.0] — 2026-09-26

First release as a library. It is a restructuring of the original single-file
`main.py` script.

### Compatibility guarantee

- Every command-line flag kept its **name, default and behavior**. `python main.py …`
  still works, and so does every flag previously marked as ignored.
- The video output was verified to be **bit-identical** to the original script (MD5 of
  every decoded frame) across nine scenarios: defaults, every profile, `--same-resolution`,
  60 fps conversion, `--start-frame` resume, `--max-frames`, several tile/pad settings,
  AI strength 0, 1 and out-of-range values, and the NVENC → CPU fallback.
- Exit codes are unchanged (0 success or partial save, 1 error, 2 bad usage,
  130 on a double Ctrl+C).

### Added

- Installable package with a `framelift` command and `python -m framelift`.
- Python API: `enhance_video()`, `VideoEnhancer` (reuses the loaded model across videos),
  `EnhanceOptions`, `EnhanceResult` (including `resume_from`), `plan_run()`,
  `probe_video()` and `apply_profile()`.
- A `stop_event` parameter to stop a run from code, with the same "save what's done"
  behavior as Ctrl+C.
- A specific exception hierarchy under `FrameliftError`.
- New CLI flags: `-i`/`-o` short aliases, `--list-models`, `--list-profiles`,
  `--version`, `-v/--verbose` and `-q/--quiet`.
- A pytest suite (70 tests), ruff configuration and GitHub Actions CI.
- README in English and Portuguese, with a complete flag reference.

### Fixed

- **`--device cpu` was ignored on machines with an NVIDIA GPU.** `RealESRGANer` picks
  CUDA whenever it exists unless it is given an explicit `device`. The chosen device is
  now passed explicitly.
- **The resume hint could skip a frame.** If Ctrl+C landed while waiting for the next
  frame, that frame's number was recorded as processed, and `--start-frame` then pointed
  one frame too far. The last frame is now recorded only after it has been written.
- **Interrupted downloads left corrupt weights behind**, which were then treated as
  valid on the next run. Downloads now go to a `.part` file that is renamed on completion.
- **Errors in the reader thread were swallowed**, silently producing a truncated video.
  They are now re-raised in the main thread.
- The torchvision compatibility shim replaced the real
  `torchvision.transforms.functional_tensor` module even on old torchvision versions that
  still ship it. It is now only installed when the module is missing.
- On an error mid-run, FFmpeg's input is now closed explicitly, so it finishes a playable
  partial file instead of waiting for the interpreter to exit.

### Changed

- The code is split into focused modules (`pipeline`, `planning`, `video`, `encoder`,
  `upscaler`, `filters`, `models`, `device`, …), with type hints and docstrings. The
  global `STOP_REQUESTED` flag is replaced by a `threading.Event`.
- Pre-cleaning profiles are declarative `Profile` objects instead of an `if` chain. The
  parameters are unchanged.
- Messages were rewritten to be friendlier and more actionable. Log output goes through
  `logging` and `tqdm.write`, so it no longer breaks the progress bar.
- Real-ESRGAN's per-tile `Tile 1/N` prints are suppressed.
- The FFmpeg command lines and first-frame brightness values moved to `--verbose`.
- The Ctrl+C handler is installed only by the CLI while a run is active, not when the
  module is imported.
- `torch.set_grad_enabled(False)` is no longer set process-wide. Inference already runs
  under `torch.inference_mode()`, so importing the library doesn't change global
  PyTorch state.
- When duplicating frames for a higher fps, each frame is converted to bytes once instead
  of once per copy.
- Heavy imports (PyTorch, Real-ESRGAN) are lazy: `--help` and `import framelift` no
  longer load PyTorch.
- Invalid values that used to crash deep inside OpenCV or FFmpeg (`--scale 0`,
  `--max-frames 0`, negative tiles) are now rejected upfront with a clear message.
