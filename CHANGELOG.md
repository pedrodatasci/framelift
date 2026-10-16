# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.3.0] — 2026-10-15

### Added

- **`FRAMELIFT_FFMPEG`**: an environment variable with the folder of `ffmpeg` and
  `ffprobe` (or the path of `ffmpeg` itself), for an FFmpeg that isn't on the `PATH`.
  It's used for every FFmpeg call: encoding, audio, `framelift trim` and the NVENC check.
- `framelift.encoder.find_binary()` returns the full path framelift will use.

### Changed

- **When FFmpeg is missing, the message says how to install it on your system**:
  `winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS,
  `sudo apt install ffmpeg` on Debian/Ubuntu. The README's installation section lists the
  same commands.

### Not changed, on purpose

- FFmpeg is still installed separately rather than bundled through pip. The package that
  bundles it (`imageio-ffmpeg`) has no `ffprobe`, which framelift relies on for the exact
  start times that keep the audio in sync and the trims frame-accurate; replacing it would
  mean parsing FFmpeg's text output, more fragile than a one-line install.

### Compatibility

- Every flag, default, API and output from 1.2 is unchanged.

## [1.2.0] — 2026-10-15

### Added

- **`on_progress` callback** in `enhance_video` and `VideoEnhancer.enhance`, to follow a
  run from code: progress bars in apps (the base for a graphical version), logs,
  notifications. It receives a `Progress` record with the `stage` (`"loading"`,
  `"frames"`, `"finishing"`), `frames_done`, `frames_total`, `frames_written`,
  `elapsed_seconds`, `eta_seconds` and `fraction`.
  - Called before the first frame, after the first and the last frame, at most 10 times
    a second in between, and once while finishing, in the thread that called
    `enhance`.
  - Set `stop_event` from it to stop early and keep what's done; an exception raised in
    it stops the run like any other error.
- `Progress` is exported from `framelift`.

### Compatibility

- Every flag, default, API and output from 1.1 is unchanged. The terminal progress bar
  works as before (`show_progress`).

## [1.1.0] — 2026-10-15

The first step towards publishing framelift on PyPI: it now installs with plain pip,
without the packages that used to break.

### Changed

- **No more `basicsr`, `realesrgan` or `torchvision`.** framelift only used the two
  Real-ESRGAN network definitions from them, and runs the networks itself. They now live
  in `framelift.archs`, copied from BasicSR 1.4.2 under its Apache-2.0 license
  (`LICENSE-BasicSR.txt`). Same parameters and same computations: the published weights
  load as before and give identical results (checked against the original classes with
  the default, x4plus and anime models). `basicsr` compiled at install time and needed a
  patch to import with torchvision 0.17+; both problems are gone.
- **Model weights go to your user cache folder by default**, so running framelift from
  any folder no longer downloads them there again: `%LOCALAPPDATA%\framelift\weights`
  on Windows, `~/Library/Caches/framelift/weights` on macOS and
  `~/.cache/framelift/weights` on Linux.
  - A `weights` folder in the current directory that already has models keeps being
    used, so nothing is downloaded again after upgrading.
  - The `FRAMELIFT_WEIGHTS` environment variable picks the folder for every run, and
    `--weights-dir` still overrides everything.
  - `EnhanceOptions.weights_dir` now defaults to `None` (meaning this default);
    `framelift.models.default_weights_dir()` says where that is.
- `requirements.txt` lists PyTorch instead of the Real-ESRGAN packages (and no longer the
  face-restoration ones they pulled in).

### Compatibility

- Every flag and the output are unchanged. The test that compares framelift with the
  original `RealESRGANer` still exists; it runs when `realesrgan` and `basicsr` are
  installed by hand, and is skipped otherwise.

## [1.0.0] — 2026-10-15

The first stable release: the same features as 0.9.0, now considered complete for the
command line. From here on the version number follows
[Semantic Versioning](https://semver.org/) strictly: flags, defaults and the Python API
only change in a backwards-incompatible way with a new major version.

### Changed

- Version numbers. The development versions were first numbered 1.0.0 to 1.8.0; below
  they're 0.1.0 to 0.9.0 (in the same order). Commit messages before this release still
  use the old numbers: "(1.3.0)" there is 0.4.0 here, and so on.
- The package is marked as stable (Development Status 5 - Production/Stable).

## [0.9.0] — 2026-10-15

### Added

- **A warning when `--output-fps` is almost, but not exactly, a multiple of the input's
  rate with `--interpolate`.** Between 29.99 and 60 fps, for example, the in-between
  moments slowly drift off the original frames until nearly every output frame has to be
  drawn: 2,835 RIFE frames for a 1,635-frame video, against 1,634 at exactly 2x (59.98
  fps), which looks the same. The warning gives both counts and the rate to use.
- **Identical consecutive frames reuse the result.** A frame identical to the one before
  it skips the AI (the result is the same, pixel for pixel) and RIFE draws nothing between
  the two. This helps with stills, title cards, renders and screen recordings; on camera
  footage that went through a lossy encoder, repeated frames are rarely identical, so
  expect no change there. `EnhanceResult.frames_reused` counts them.

### Changed

- With `--interpolate`, a rate typed with a few decimals that's within 0.01 fps of an
  exact multiple (59.94 for 29.97 x 2) is used as that exact multiple.
- `framelift tune` still runs the AI on every frame, so its timings stay comparable.

### Compatibility

- Every flag, default and API from 0.8 still works the same. With the AI on, the output
  is exactly the same as before. With `--interpolate`, frames between two identical
  frames are now repeats instead of being drawn by RIFE.

## [0.8.0] — 2026-10-15

### Added

- **`--interpolate`: smoother motion with RIFE.** The extra frames of a higher frame rate
  are drawn by [RIFE](https://github.com/hzwer/Practical-RIFE) 4.26, as real in-between
  frames, instead of being repeated. Without `--output-fps` it doubles the frame rate.
  - Works with the AI or with `--no-ai`. RIFE runs on the finished frames, so the AI still
    runs once per original frame.
  - Uses the same device as the AI: an NVIDIA GPU or the CPU through PyTorch, or an Intel
    GPU through OpenVINO (or the CPU with `--backend openvino`). `--half` applies to it on
    GPUs.
  - The output has exactly as many frames as repeating gives, so the duration and the audio
    sync don't change, and stopping and resuming work as before.
  - Scene cuts are detected and repeated instead of blended. Moments within 5% of an
    original frame show that frame, which saves RIFE runs at rates like 29.97 → 60.
  - Its weights (25 MB) are downloaded on first use. On OpenVINO, the converted model is
    saved in `weights/openvino_cache/`, so later runs start faster.
- Python API: `EnhanceOptions.interpolate`, `EnhanceResult.frames_interpolated`, and
  `framelift.interpolation` (`FrameInterpolator`, `Retimer`).

### Changed

- `--output-fps` and the summary table mention RIFE when it's used.

### Compatibility

- Every flag, default and API from 0.7 still works the same. Without `--interpolate`,
  the output is unchanged.

## [0.7.0] — 2026-10-15

### Added

- **`--no-ai`**: skip the AI entirely. No model is loaded and `--device` doesn't matter;
  each frame gets the pre-cleaning (`--profile`), a plain Lanczos resize (`--scale`) and
  the frame-rate change (`--output-fps`), at about the speed of decoding and encoding. To
  only change the frame rate: `--no-ai --same-resolution --output-fps 60`. Everything else
  works as usual: audio, `--pick-range`, stopping and resuming.
- Python API: `EnhanceOptions.ai` (`ai=False` is `--no-ai`) and `EnhanceOptions.uses_ai`.

### Changed

- **`--ai-strength 0` no longer runs the AI.** It used to run the model on every frame and
  then discard the result; now it works like `--no-ai`, with exactly the same output as
  before (checked frame by frame), in a fraction of the time (0.7 s instead of 42.7 s for
  10 frames at 720x480 on one CPU core, in our test).
- **OpenVINO (`--device igpu`, `--backend openvino`) takes and returns 8-bit frames.** The
  conversions around the network (to floats and back, with Real-ESRGAN's rounding) are
  compiled into the model, so on an Intel GPU they run on the GPU instead of the CPU, and a
  quarter of the data comes back. The output stays within 1 level out of 255 of PyTorch's,
  as before.
- Runs without the AI say "Processing" and "processed" instead of "Enhancing" and
  "enhanced", and their summary table shows "no AI" instead of the model and device.
- Without the AI, the first frame isn't checked for being black (the check is there to
  catch model and driver problems), so a video that opens on a black frame converts fine.

### Compatibility

- Every flag, default and API from 0.6 still works the same. With the AI on, the output is
  unchanged on PyTorch, and within 1 level out of 255 on OpenVINO, as before.

## [0.6.0] — 2026-10-10

### Added

- **`framelift tune` tries more ways to run the AI, and keeps the fastest:**
  - **OpenVINO on the CPU** (`--backend openvino`), Intel's runtime, against PyTorch with
    the tile and threads that already won;
  - **the Intel GPU** (`--device igpu`: Iris, UHD, Arc, through OpenVINO), with tile
    sizes and FP16 on and off, when OpenVINO can see one;
  - **2 AI workers** (`--workers 2`) against 1, on CPUs with 4 threads or more, with two
    short real runs (5 frames each). 2 are kept only if at least 5% faster.

  The report lists every configuration tried ("OpenVINO, tile 256", "Intel GPU, tile 512,
  FP16", "Real run, 2 workers × 4 threads ← faster"). OpenVINO is optional: without it,
  those tests are skipped.
- **`--device igpu`**: run the AI on an Intel GPU through OpenVINO. `auto` never picks
  it by itself; `tune` checks whether it's actually faster than the CPU.
- **`--backend {torch,openvino}`**: what runs the AI on the CPU. In our tests OpenVINO was
  ~13% faster on the default model (speed test) and ~20% on `realesrgan-x4plus`.
- **`--workers N`**: frames the AI works on at the same time, each with its share of the
  CPU threads (`--torch-threads` is now per worker; `0` splits PyTorch's default).
- Optional dependency group `intel` (`pip install -e ".[intel]"`, installs OpenVINO).
- Python API: `BenchmarkRun.device` / `.backend`, `RealRun.workers` / `.threads`,
  `TuneReport.real_runs`, `AIUpscaler.infer()` / `.finish()`.

### Changed

- **The AI runs in its own thread(s).** While it works on the next frame, the previous one
  is resized, blended and encoded, so the two overlap on machines with cores to spare.
  On a single core it's a draw (55.7 s vs 56.3 s for 12 frames in our tests).
- **framelift now runs the Real-ESRGAN networks itself** instead of through
  `RealESRGANer.enhance`: same tiling, padding and rounding (the output is bit-for-bit the
  same, checked against `RealESRGANer` in the tests), but with no state between calls, so
  several threads can share one model, and no per-tile console output to hide. The
  conversion back to 8 bits now happens on the device (on CUDA, 4x less to copy back).
- `--debug-timings`: `ai` is now the time spent *waiting* for the AI (it runs in parallel
  with the rest); `queue` is still the wait for the reader.
- `--torch-threads 0` now restores PyTorch's default instead of keeping whatever an earlier
  call in the same process had set.
- The "FP16 only applies to CUDA" note now says "GPUs" (it applies to `igpu` too).

### Compatibility

- Every flag, default and API from 0.5 still works the same, and with the defaults the
  output is bit-for-bit identical to 0.5, with 1 worker or 2.
- `--backend openvino` and `--device igpu` round a few values differently from PyTorch (at
  most 1 level out of 255 on CPU; FP16 on the Intel GPU, like `--half` on NVIDIA).
- OpenVINO caches compiled models in `<weights dir>/openvino_cache/` (already ignored by
  git, with the rest of `weights/`).

## [0.5.0] — 2026-10-10

### Added

- **Choose the part to enhance on a timeline: `framelift trim`.** A player window (VLC
  does the playback, Tkinter draws the window) shows the video above a timeline with a
  start and an end handle. Drag them, or use the keys (<kbd>I</kbd>/<kbd>O</kbd> at the
  playhead, arrows for one frame or, with <kbd>Shift</kbd>, one second, <kbd>Space</kbd>,
  "Play selection"), and confirm with <kbd>Enter</kbd>.
- **The chosen part becomes the input.** It's saved as a new file,
  `<input>_trim_<first>-<last frame>.mp4` (the original is never touched), so
  `--start-frame`, `--max-frames` and resuming after a stop refer to it. The cut:
  - starts and ends on the exact frames chosen, even between keyframes, and has exactly
    the same pixels as the original (lossless H.264, verified by MD5 of every frame);
  - keeps the audio in sync to the sample (lossless ALAC; converted to AAC in the
    enhanced output);
  - falls back to decoding from the start when seeking can't land exactly (e.g. MPEG-TS
    captures with few keyframes).
- **`--pick-range`** on the main command: pick, cut and enhance in one go. If the run
  stops, the resume hint points at the cut; picking the same range again reuses it.
- **`framelift trim --start/--end`**: the same cut by time, with no window (servers, SSH).
- Python API: `pick_range()`, `trim_video()`, `FrameRange` and `TrimResult`.
- Optional dependency group `picker` (`pip install -e ".[picker]"`, installs python-vlc).
  VLC itself is only needed for the window; nothing else in framelift needs it.
- `framelift tune` reports a **short real run** (`TuneReport.real_run`) and each preset's
  `basis` (`"measured"` or `"extrapolated"`).

### Changed

- **The default model is about 30% faster on CPU.** `realesr-animevideov3` now runs with
  its weights in channels-last layout, oneDNN's native one: 8.3 → 5.8 s per 720x480 frame
  in our tests, end to end. The RRDB models (`x4plus` and friends) were slower that way,
  so they're unchanged, as is everything on GPU.
- **`framelift tune` estimates times from a real run.** Three frames go through the whole
  pipeline with the preset's settings, so the estimates now include reading, clean-up,
  blending, encoding (with the encoder competing for the CPU) and loading the model. In our
  tests a 40-frame run took 7% longer than estimated (it used to be about 15% with the AI
  alone). On GPU both presets are timed this way; on CPU, `best` is built from one timed
  tile of the heavy model scaled to the frame's exact tiling, instead of from the ratio of
  two small crops.
- **`framelift tune` measures the AI's effect more carefully**: on crops from three
  frames instead of two, and with real picture around each crop (trimmed afterwards), so
  the AI and the clean-up filters don't react to artificial borders. On our test clip the
  third frame moved the suggested strength from 0.5 to 0.6; the context changed little.
- `--no-benchmark` also skips the short real run.
- The report's "Times are rough" note is replaced by what the times include.
- Docs: the CPU speed ratio between the default model and `realesrgan-x4plus` is now
  about 45x (it was 27x before the default model got faster).

### Compatibility

- Every flag, default and API from 0.4 still works the same.
- On **CPU with the default model**, the output is no longer bit-identical to the
  original script: the new memory layout rounds differently in about 0.002% of the
  values, by at most 1 level out of 255. GPU runs and the other models are unaffected.

## [0.4.0] — 2026-09-29

### Added

- **The audio is kept.** The input's audio is copied into the output automatically,
  trimmed to exactly the enhanced frames (`--start-frame`, `--max-frames`, Ctrl+C), in sync
  to within a millisecond, including across joined parts and 30 → 60 fps conversion.
  AAC, MP3, AC-3 and E-AC-3 are copied without re-encoding; other codecs (e.g. PCM from
  capture cards) are converted to AAC 192 kbps so every player can play them.
  `--no-audio` (or `audio=False`) turns it off, and `EnhanceResult.audio` reports what
  happened. The audio is added in a second pass after the video is finished, so a problem
  with it never costs the video: the MP4 is saved silent, with a warning.
- **Gallery** (`GALLERY.md` / `GALLERY.pt-BR.md`): a separate page with more before/after
  examples, linked from the top of both READMEs.
- `scripts/make_comparison.py`: turns an original video and framelift's output into the
  gallery's GIFs. It picks the sharpest steady scene and its most detailed area, keeps each
  GIF under 6 MB, and prints the Markdown to paste into the gallery.

### Changed

- By default the output now has audio. The video stream is still bit-identical to the
  original script's; `--no-audio` gives the old silent MP4.

## [0.3.0] — 2026-09-29

### Added

- **`framelift tune` adapts the AI strength to the footage.** Small crops are upscaled
  with and without AI and three things are measured: how much sharpness the AI brings
  back, whether it cleans noise or invents texture, and whether it preserves the picture's
  structure (SSIM). Those decide the AI strength, separately for each model. Across the
  calibration clips the suggestion now ranges from 0.4 (already-sharp source) to 0.7
  (blurry source), instead of the fixed 0.45.
- **Two presets with time estimates**: `fast` (the light video model) and `best` (a
  heavier model, `realesrgan-x4plus` or the anime model with `--anime`, plus `--pre-pad 10`,
  `--crf 18` and a slower x264 preset). `best` is recommended when it's estimated to finish
  within 10 minutes. The `best` command writes to `<output>_best.mp4`, so both can be
  compared.
- **Softness measurement** (Crété-Roffet no-reference blur metric), shown in the report.
- **CPU thread test**: all threads, half, and 4 are compared, because laptops that mix
  performance and efficiency cores are often faster with fewer threads.
- **Smarter test frame**: the sharpest *steady* frame is used, skipping fast motion and
  scene cuts, and detail is scored after a light blur so pixel noise doesn't pass for detail.
- The comparison sheet has one row per preset, with each preset's AI strength marked.
- New tune flags: `--no-best`, `--anime`. `--samples` now defaults to 6.
- Python API: `Preset`, `AIEffect`, `softness()`, `suggest_ai_strength()`, and
  `TuneReport.presets`, `.model_tests`, `.recommended_preset`, `.softness`.

### Changed

- `TuneReport.recommended` now returns the recommended preset's options.
- The README note claiming that compression hides grain from the noise check was wrong.
  A pixel-level look at the concert footage showed that it is soft, not grainy; the
  blotches in dark areas aren't pixel noise. The note now explains the difference.

### Unchanged

- `framelift -i … -o …` still produces output bit-identical to the original script.

## [0.2.0] — 2026-09-28

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

## [0.1.0] — 2026-09-26

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
