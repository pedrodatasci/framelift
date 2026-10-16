<div align="center">

# 🎞️ framelift

**Upscale and restore videos with Real-ESRGAN — on an NVIDIA GPU if you have one, on any CPU if you don't.**

[![Tests](https://github.com/pedrodatasci/framelift/actions/workflows/tests.yml/badge.svg)](https://github.com/pedrodatasci/framelift/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[Português 🇧🇷](README.pt-BR.md) · [🎬 More examples](GALLERY.md)

</div>

framelift takes an old, blurry or low-resolution video and runs every frame through
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), a neural network trained to
restore detail. Instead of handing you the raw AI output (which can look plasticky),
it **blends** it with a classic high-quality resize, so the result stays faithful to
the original footage.

```bash
framelift -i grandma_1998.mp4 -o grandma_1998_hd.mp4 --profile old_tv --scale 2
```

## Before / after

Two real clips, both upscaled from 480p to 1080p. Each GIF shows the same area side by
side: the original on the left (just stretched to the same size), framelift on the right.

**Color music video:** cleaner strings and frets, less compression blockiness.

![Before and after: color music video, zoomed in](docs/music_video_zoom.gif)

<details>
<summary>Full frame</summary>

![Before and after: color music video, full frame](docs/music_video_full.gif)

</details>

**Black-and-white concert footage:** grain and blocky artifacts cleaned up, crisper edges.
Enhanced on a laptop CPU.

![Before and after: black-and-white concert footage, zoomed in](docs/demo_zoom.gif)

<details>
<summary>Full frame</summary>

![Before and after: black-and-white concert footage, full frame](docs/demo_full.gif)

</details>

<sub>GIFs are compressed and scaled down for this page; the real MP4 output is sharper.</sub>

**[🎬 More examples in the gallery →](GALLERY.md)**

## Highlights

- **Works on any machine.** Uses CUDA when available and falls back to the CPU
  otherwise — Intel Iris laptops included.
- **Stop anytime, lose nothing.** Press <kbd>Ctrl</kbd>+<kbd>C</kbd> once and framelift
  finishes the current frame and saves a **playable** MP4 of everything done so far.
  It then tells you exactly how to resume.
- **Keeps the sound.** The original audio is copied over in sync, trimmed to the part
  that was enhanced, without re-encoding whenever possible.
- **Pick the part to enhance by eye.** `framelift trim` (or `--pick-range`) opens a player
  with a timeline: drag the start and end, and that stretch is cut out on the exact frame,
  without losing quality, and becomes your input.
- **Natural-looking results.** The `--ai-strength` blend lets you dial in anything from
  "subtle cleanup" to "full AI".
- **Pre-cleaning profiles** for common sources: phone footage, VHS/TV captures, very
  noisy video.
- **Uses what your machine has.** Besides NVIDIA GPUs, it can run the AI on Intel GPUs
  (Iris, UHD, Arc) and through Intel's OpenVINO on any CPU, and on many-core CPUs it can
  work on two frames at once. `framelift tune` measures which is fastest on *your* machine.
- **Fast pipeline.** Frames are decoded on a background thread while the AI works, and
  streamed straight into FFmpeg: no temporary image files on disk.
- **Smoother motion (RIFE).** Turn 24/30 fps into 60 fps with real in-between frames
  drawn by [RIFE](https://github.com/hzwer/Practical-RIFE) (`--interpolate`), or just by
  repeating frames. Either way the duration doesn't change, and it works along with the AI
  or on its own (`--no-ai`).
- **A real Python library**, not just a script: `enhance_video(...)`, typed options,
  structured results and specific exceptions.
- **Models download themselves** on first use.

## Contents

- [Before / after](#before--after) (and [more examples](GALLERY.md))
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Finding the best settings (`framelift tune`)](#finding-the-best-settings-framelift-tune)
- [Choosing the part to enhance (`framelift trim`)](#choosing-the-part-to-enhance-framelift-trim)
- [Recipes](#recipes)
- [Command-line reference](#command-line-reference)
- [Models](#models)
- [Pre-cleaning profiles](#pre-cleaning-profiles)
- [Performance tuning](#performance-tuning)
- [Stopping, resuming and joining parts](#stopping-resuming-and-joining-parts)
- [Audio](#audio)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Development](#development)
- [Coming from the old `main.py`](#coming-from-the-old-mainpy)
- [Credits and license](#credits-and-license)

## How it works

```
┌───────────────┐  queue   ┌─────────────┐      ┌─────────────────┐  pipe   ┌────────┐
│ reader thread │ ───────► │ Real-ESRGAN │ ───► │ blend with      │ ──────► │ FFmpeg │ ──► out.mp4
│ decode+clean  │(prefetch)│  (workers)  │      │ Lanczos resize  │  (raw)  │  H.264 │
└───────────────┘          └─────────────┘      └─────────────────┘         └────────┘
```

1. **Read and pre-clean.** A background thread decodes frames and applies the chosen
   [profile](#pre-cleaning-profiles), staying a few frames ahead of the AI.
2. **Upscale.** Real-ESRGAN restores detail and enlarges the frame to the target size. It
   runs in its own thread (or several, with `--workers`), so the AI is already on the next
   frame while the previous one is blended and encoded.
3. **Blend.** The AI result is mixed with a plain Lanczos resize of the same frame,
   according to `--ai-strength`.
4. **Encode.** Raw frames are piped into FFmpeg (libx264 or NVENC). The file is written
   under a temporary name and only renamed once FFmpeg finishes cleanly.

To raise the frame rate (`--output-fps`), frames are repeated in an even pattern. With
`--interpolate`, RIFE draws the extra frames instead, between steps 3 and 4: it estimates
how everything moves from one finished frame to the next and renders the moments in
between. The AI still runs once per original frame. Across a scene cut there's no motion
to follow, so those frames are repeated instead of blended.

A frame identical to the one before it (a still, a duplicated frame) doesn't go through
the AI again: its result would be the same, pixel for pixel, so the previous one is
reused, and RIFE has nothing to draw between the two.

Before committing to hours of work, framelift checks that the first frame isn't
black — both before and after the AI — which catches broken decodes, corrupt weights
and driver problems early.

With `--no-ai` (or `--ai-strength 0`), step 2 is skipped and no model is loaded: each
frame only gets the Lanczos resize. The same pipeline then resizes, pre-cleans or changes
the frame rate at about the speed of decoding and encoding. The black-frame check is
skipped too, since it's there to catch problems with the AI.

## Installation

You need **Python 3.9+** and **[FFmpeg](https://ffmpeg.org/download.html)** available
on your `PATH` (`ffmpeg -version` should work in a terminal).

```bash
git clone https://github.com/pedrodatasci/framelift.git
cd framelift
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Using an NVIDIA GPU?** Install a CUDA build of PyTorch *first*, using the command
that [pytorch.org](https://pytorch.org/get-started/locally/) gives you for your system.
Skip this step for CPU-only machines.

Then install framelift:

```bash
pip install -r requirements.txt    # optional: the exact, known-good versions
pip install -e .
framelift --version
```

This gives you the `framelift` command. `python -m framelift` works too.

**Optional, for Intel hardware:** `pip install -e ".[intel]"` installs
[OpenVINO](https://docs.openvino.ai), which lets framelift run the AI on an Intel GPU (Iris,
UHD, Arc) with `--device igpu`, or on the CPU with `--backend openvino` (often faster than
PyTorch, on any x86 CPU). `framelift tune` tries both when it's installed.

**Optional, for the visual range picker** ([`framelift trim`](#choosing-the-part-to-enhance-framelift-trim)):
install [VLC media player](https://www.videolan.org/vlc/) (64-bit if your Python is
64-bit) and its Python binding:

```bash
pip install -e ".[picker]"         # installs python-vlc
```

On Linux, Tkinter may also need `sudo apt install python3-tk`. Everything else works
without them.

## Quick start

```bash
# 1. Let framelift measure your video and machine (prints a fast and a best command)
framelift tune -i input.mp4

# 2. Try it on the first ~10 seconds (300 frames at 30 fps) to judge the look quickly
framelift -i input.mp4 -o preview.mp4 --max-frames 300

# 3. Happy with it? Run the whole thing
framelift -i input.mp4 -o output.mp4
```

The defaults are a good starting point: the fast `realesr-animevideov3` model, a 1.5x
upscale and a 45% AI blend.

## Finding the best settings (`framelift tune`)

Small changes in the settings can change the result a lot, and the right values depend on
*your* footage. Let framelift measure them:

```bash
framelift tune -i input.mp4
```

It takes a minute or two (longer on a slow CPU with big videos) and works in six steps:

1. **Looks at your footage.** On frames spread across the video it measures **softness**
   (how blurry the picture is), **noise** (which picks the clean-up profile) and
   resolution (which picks a scale that lands on Full HD without stretching more than 3x).
   For the tests it picks the sharpest *steady* frame, skipping fast motion and scene cuts.
2. **Measures your machine.** It upscales a real frame with each candidate setup and keeps
   the fastest one that fits in memory: tile sizes, FP16 on NVIDIA (double-checked without
   it, since FP16 is slower on some older cards) and, on CPU, **the number of threads**
   (laptops that mix fast and efficiency cores are often faster with fewer). If
   [OpenVINO](#installation) is installed, it also times **OpenVINO on the CPU** and, if
   there is one, **the Intel GPU** (with and without FP16), and keeps whichever is fastest.
   It also checks whether NVENC works.
3. **Measures what the AI does to your footage.** This is where it adapts to the input.
   Small crops from three frames are upscaled with and without AI (with some picture around
   them, so the AI doesn't trip over artificial borders), and three things are measured:
   - **sharpness:** how much the AI actually brings back compared with a plain resize;
   - **noise:** whether the AI cleans noise up or invents grain and texture;
   - **structure:** whether it restores the picture or starts redrawing details.

   Those decide the **AI strength**, separately for each model. A soft source where the AI
   recovers a lot gets a high strength. An already-sharp source, or a model that invents
   texture on this footage, gets a lower one.
4. **Builds two presets**:
   - **`fast`**: the light video model (`realesr-animevideov3`). Usually minutes.
   - **`best`**: a heavier model (`realesrgan-x4plus`, or the anime model with `--anime`)
     that recovers more detail, plus more careful settings: `--pre-pad 10` against edge
     artifacts, `--crf 18` so the encoder keeps more detail, and a slower x264 preset.
     On a CPU it can be around 45x slower.

5. **Times a short real run.** Three frames go through the whole pipeline with the
   preset's settings (reading, clean-up, AI, blend and encoding, with the encoder competing
   for the CPU as it would), so the time estimates include everything but the audio pass.
   On a CPU with 4 threads or more, it runs this twice (5 frames each): with **one AI
   worker** using all the threads, and with **two workers** sharing them, each on its own
   frame. Two workers are kept only if they're at least 5% faster.
   On a GPU both presets are timed this way. On a CPU, where a `best` frame can take
   minutes, one tile of the heavy model is timed instead and scaled to the frame's exact
   tiling. `best` is recommended when it's estimated to finish within 10 minutes (short
   clips, or a GPU); otherwise `fast` is.
6. **Makes a comparison sheet** (`input_tune.png`): the same 100% crop rendered with each
   preset (rows) and different AI strengths (columns), with each preset's pick marked.

Here's the real sheet for the color music video above. The bottom row is the `best` preset:
notice the sharper studs on the strap. Click it to see it at full size.

![Comparison sheet made by framelift tune](docs/tune_sheet.jpg)

And a report from a real run on another 720x480 clip, on a single-core test machine with
OpenVINO installed (so the times are long, and there was no worker test):

```
Your machine
  CPU (no NVIDIA GPU in use, so expect it to be slow)

Your video
  input.mp4  720x480 @ 29.970 fps, 130 frames (0m04s)
  Softness  slightly soft (0.35)
  Noise     very clean (0.4) → profile none
  Size      480p → 1080p → --scale 2.25
  Tested on frame 123 (the sharpest steady one)

What the AI does to it
  animevideov3  sharpness +0.15 · noise ±0.0 · structure 0.98  → AI strength 0.6
                recovers sharpness well
  x4plus        sharpness +0.18 · noise ±0.0 · structure 0.98  → AI strength 0.6
                recovers sharpness well

Speed test (animevideov3, 1620x1080 output)
  tile 256, FP32                       4.52 s/frame
  tile 128, FP32                       5.34 s/frame
  OpenVINO, tile 256, FP32             3.91 s/frame  ← fastest
  Real run, 3 frames with encoding: 4.06 s/frame + 4 s to start
  (real runs include reading, blending, encoding and loading the model)

Presets
  fast  animevideov3 · AI 0.6 · ~9 min measured  (light video model)  ← recommended
  best  x4plus · AI 0.6 · ~4h41 extrapolated  (heavier model, more careful encoding)
  Times include loading the model and encoding (not the audio: a few seconds).
  measured = a short real run of the preset; extrapolated = built from one.

Comparison sheet: /home/you/input_tune.png
  The same 100% crop of frame 123: columns are AI strengths
  (the presets' picks are marked), rows are the presets. Adjust to taste.

Commands
  fast (recommended):
    framelift -i input.mp4 -o input_enhanced.mp4 --scale 2.25 --ai-strength 0.6 --device cpu --backend openvino --x264-preset medium
  best:
    framelift -i input.mp4 -o input_enhanced_best.mp4 --model realesrgan-x4plus --scale 2.25 --ai-strength 0.6 --device cpu --backend openvino --pre-pad 10 --crf 18 --x264-preset slow
```

Good to know:

- **Faster vs. better is about the model, not the AI strength.** The AI strength is only a
  blend at the end, so it costs no time at all. The presets differ in model and encoding
  effort; each gets the strength that suits it on your footage.
- **Anything you pass is kept.** `framelift tune -i vhs.mp4 --scale 2 --profile old_tv`
  keeps your scale and profile, tunes everything else, and marks yours as "(yours)". With
  `--model`, only that model is tuned (no fast/best choice).
- **The measurements are a strong starting point; the sheet has the final word.** The
  strength rules were calibrated on a handful of clips (sharp, blurry and grainy synthetic
  video, plus real old footage), and taste still matters. Pick the tile of the sheet you
  like best and adjust `--ai-strength` to match.
- **Grain and blotches aren't the same thing.** Old footage often shows blotchy patches in
  dark areas, left by earlier compression. They aren't pixel noise, so the noise check
  doesn't count them and suggests no clean-up profile; step 3 measures how the AI deals
  with them. If they bother you, run tune again with `--profile soft_camera` and compare
  the sheets.
- **The time estimates come from a short real run**, so they include loading the model and
  encoding. In our tests the real run of 40 frames took 7% longer than estimated (205 s
  against 191 s, Python's own start-up included). On CPU, the `best` estimate is built from
  one timed tile of the heavy model rather than a whole frame, so take it as a good guide.
- **The first run downloads the heavier model** (67 MB). `--no-best` skips it and tunes
  only the `fast` preset.
- **On slow CPUs with Full HD or larger videos, the speed test itself can take several
  minutes.** `--no-benchmark` skips it (and the short real run) and still gives you the
  measurements, the presets (without time estimates) and the sheet.
- **Both presets write to different files** (`_enhanced.mp4` and `_enhanced_best.mp4`), so
  you can run both on a short clip and compare.

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input VIDEO` | *required* | Video to analyse. |
| `-o`, `--output MP4` | `<input>_enhanced.mp4` | Only used to write the suggested commands (`best` adds `_best`). |
| `--samples N` | `6` | Frames spread across the video that are inspected for noise, softness and motion. |
| `--sheet PNG` | `<input>_tune.png` | Where to save the comparison sheet (in the current folder by default). |
| `--no-sheet` | off | Don't make a comparison sheet. |
| `--no-benchmark` | off | Skip the speed test and the short real run: much faster on CPU, but no tile/FP16/thread tuning and no time estimates. |
| `--no-best` | off | Only tune the `fast` preset: skips the heavier model and its download. |
| `--anime` | off | The video is animation: the `best` preset uses `realesrgan-x4plus-anime-6B`. |
| `--model`, `--device`, `--gpu-id`, `--weights-dir`, `--torch-threads`, `--workers`, `--backend`, `--profile`, `--scale`, `--same-resolution`, `--ai-strength`, `--tile`, `--half`, `--encoder`, `--start-frame`, `--max-frames` | not set | Same meaning as in the main command. Whatever you pass is kept as-is and carried into the suggested commands. `--start-frame` and `--max-frames` change the time estimates. |
| `-v`, `-q` | | More detail / only the final report. |

Exit codes are the same as the main command; <kbd>Ctrl</kbd>+<kbd>C</kbd> cancels with code 130.

## Choosing the part to enhance (`framelift trim`)

Often only part of a video matters: one song of a concert, one scene of a tape. Pick it
by eye:

```bash
framelift trim -i concert.mp4
```

A player window opens (VLC does the playback, so it's smooth even with big files) with a
timeline under the video:

- **drag the two white handles** to set the start and end; the video follows, so you see
  the exact frame you're on. Click anywhere else on the timeline to move the playhead;
- or use the keys: <kbd>Space</kbd> play/pause, <kbd>I</kbd> / <kbd>O</kbd> set the start /
  end at the playhead, <kbd>←</kbd> <kbd>→</kbd> one frame, <kbd>Shift</kbd>+<kbd>←</kbd>
  <kbd>→</kbd> one second, <kbd>Home</kbd> / <kbd>End</kbd> jump to the start / end,
  <kbd>M</kbd> mute;
- **Play selection** plays just the chosen part and stops on its last frame;
- <kbd>Enter</kbd> (or **Use this range**) confirms, <kbd>Esc</kbd> cancels.

The chosen part is saved as a **new file**, `concert_trim_<first>-<last frame>.mp4` next to
the original, and framelift prints the commands to tune and enhance it. The original is
never touched. The cut copy:

- **starts and ends on the exact frames you chose**, even between keyframes;
- **has exactly the same pixels** as the original: the video is stored losslessly. The
  price is size, roughly 10–30x the original, so delete it once you're done;
- **keeps the audio in sync to the sample**, stored losslessly too (ALAC). It's converted to
  AAC when the enhanced video is saved.

From then on **the cut is your input**: `--start-frame` and `--max-frames` count within it,
so stopping halfway and [resuming](#stopping-resuming-and-joining-parts) works as usual.

To pick and enhance in one go, add `--pick-range` to the normal command:

```bash
framelift -i concert.mp4 -o song3_hd.mp4 --pick-range
```

If the run stops early, the resume hint points at the cut (`-i concert_trim_….mp4
--start-frame N`); run that without `--pick-range`. Choosing the same range again reuses
the cut already made instead of cutting it twice.

No desktop (a server, SSH)? Give the times instead and no window opens:

```bash
framelift trim -i concert.mp4 --start 1:30 --end 2:45.5
```

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input VIDEO` | *required* | Video to cut. |
| `-o`, `--output MP4` | `<input>_trim_<first>-<last>.mp4` | Where to save the cut (next to the input by default). |
| `--start TIME` | the beginning | Start time, without the window: seconds (`90.5`), `M:SS` (`1:30`) or `H:MM:SS` (`1:02:03`). |
| `--end TIME` | the end | End time, same formats. Give either one, or both, to skip the window. |
| `-v`, `-q` | | More detail / only warnings and errors. |

Exit codes: `0` saved, `1` error or the window was closed without choosing, `130`
<kbd>Ctrl</kbd>+<kbd>C</kbd>. The picker needs VLC and python-vlc (see
[Installation](#installation)); `--start`/`--end` don't.

## Recipes

| Goal | Command |
| --- | --- |
| Quick preview | `framelift -i in.mp4 -o prev.mp4 --max-frames 300` |
| Old VHS / TV capture | `framelift -i vhs.mp4 -o out.mp4 --profile old_tv --scale 2 --ai-strength 0.5` |
| Phone video, subtle cleanup | `framelift -i phone.mp4 -o out.mp4 --profile soft_camera --ai-strength 0.3` |
| Very noisy footage | `framelift -i noisy.mp4 -o out.mp4 --profile heavy_noise` |
| Anime / cartoons | `framelift -i ep01.mp4 -o out.mp4 --model realesrgan-x4plus-anime-6B --scale 2 --ai-strength 0.8` |
| Maximum detail (slow) | `framelift -i in.mp4 -o out.mp4 --model realesrgan-x4plus --ai-strength 0.6` |
| Restore without resizing | `framelift -i in.mp4 -o out.mp4 --same-resolution` |
| Convert to 60 fps | `framelift -i in.mp4 -o out.mp4 --output-fps 60` |
| Only change the frame rate (no AI, fast) | `framelift -i in.mp4 -o out.mp4 --no-ai --same-resolution --output-fps 60` |
| Smoother motion: 60 fps with RIFE | `framelift -i in.mp4 -o out.mp4 --output-fps 60 --interpolate` |
| Only smoother motion (no AI) | `framelift -i in.mp4 -o out.mp4 --no-ai --same-resolution --interpolate` |
| NVIDIA GPU, full speed | `framelift -i in.mp4 -o out.mp4 --device cuda --half --tile 768 --encoder nvenc` |
| Intel laptop (CPU) | `framelift -i in.mp4 -o out.mp4 --device cpu --tile 128 --torch-threads 8` |
| Resume after frame 1200 | `framelift -i in.mp4 -o part2.mp4 --start-frame 1201` |
| Only one part of the video | `framelift -i in.mp4 -o part.mp4 --pick-range` |

## Command-line reference

```
framelift -i INPUT -o OUTPUT [options]
```

Run `framelift --help` for the same information in your terminal.

### Input / output

| Flag | Default | Description |
| --- | --- | --- |
| `-i`, `--input VIDEO` | *required* | Video to enhance. Anything OpenCV/FFmpeg can read (MP4, MKV, AVI, MOV…). |
| `-o`, `--output MP4` | *required* | Where to save the result. Parent folders are created automatically. An existing file at this path is replaced **only** once the new one has finished successfully. |
| `--no-audio` | audio on | Leave the audio out. By default the input's audio is copied over, trimmed to match the enhanced part. See [Audio](#audio). |

### How the result looks

| Flag | Default | Description |
| --- | --- | --- |
| `--model NAME` | `realesr-animevideov3` | Real-ESRGAN model. See [Models](#models) or `framelift --list-models`. |
| `--scale FACTOR` | `1.5` | Output size relative to the input: `2` doubles width and height, `1.25` adds 25%. Values below 1 shrink the video (still cleaned by the AI). Ignored with `--same-resolution`. |
| `--same-resolution` | off | Keep the exact input resolution. Useful to restore quality without making the file bigger. |
| `--ai-strength N` | `0.45` | Share of the AI result in the final image, from `0` to `1`. `1.0` = pure AI (sharpest, can look artificial); `0.45` = balanced; `0.30` = natural; `0` = plain Lanczos resize, without running the AI at all (like `--no-ai`). Values outside 0–1 are clamped. |
| `--no-ai` | AI on | Skip the AI entirely: no model is loaded, and `--device` and the other AI settings don't matter. What's left is the pre-cleaning (`--profile`), a plain Lanczos resize (`--scale`) and the frame-rate change (`--output-fps`), at about the speed of decoding and encoding. `--no-ai --same-resolution --output-fps 60` only changes the frame rate. |
| `--profile NAME` | `none` | Clean-up filter applied *before* the AI. See [Pre-cleaning profiles](#pre-cleaning-profiles) or `framelift --list-profiles`. |
| `--output-fps FPS` | same as input | Output frame rate. Higher values repeat frames in an even pattern so the **duration stays the same** (e.g. 24 → 60 fps writes each frame 2 or 3 times). It can't be lower than the input's rate. On its own this doesn't create new motion, but it's handy when a platform or editor expects 60 fps; add `--interpolate` for real in-between frames. |
| `--interpolate` | off | Draw the extra frames with [RIFE](https://github.com/hzwer/Practical-RIFE) instead of repeating them: new in-between frames, so motion really looks smoother. Without `--output-fps`, it doubles the frame rate. Works with the AI (it runs on the finished frames, so the AI still runs once per original frame) or with `--no-ai`. Uses the same device as the AI (`--device`, `--half`, `--backend`); its weights (25 MB) are downloaded on first use. Scene cuts are repeated, not blended. Fast motion and things crossing each other (hands, hair) can show small artifacts. Pick an exact multiple of the input's rate (or leave `--output-fps` out): at 60 fps from 29.99, the in-between moments drift and RIFE draws up to ~75% more frames. framelift warns when that's the case and tells you the exact rate to use. |

### Which frames to process

| Flag | Default | Description |
| --- | --- | --- |
| `--start-frame N` | `1` | First frame to process, counting from **1**. Used to resume an interrupted run: if the last frame done was 1200, pass `1201`. |
| `--max-frames N` | all | Process at most `N` frames starting at `--start-frame`. Ideal for previews and for splitting long jobs into sessions. |
| `--pick-range` | off | Before enhancing, open the [range picker](#choosing-the-part-to-enhance-framelift-trim) to choose the part to enhance. That part is saved as `<input>_trim_<first>-<last>.mp4` and becomes the input; `--start-frame` and `--max-frames` then count within it. Needs VLC. |

### Hardware and performance

| Flag | Default | Description |
| --- | --- | --- |
| `--device {auto,cpu,cuda,igpu}` | `auto` | Where the AI runs. `auto` picks an NVIDIA GPU if PyTorch can see one, otherwise the CPU. `igpu` runs it on an Intel GPU (Iris, UHD, Arc) through OpenVINO (`pip install openvino`); `auto` never picks it on its own, `framelift tune` checks whether it's faster. `cuda` and `igpu` fail with a clear message if that GPU isn't usable. |
| `--gpu-id ID` | `0` | Which CUDA GPU to use on multi-GPU machines (`nvidia-smi` lists the IDs). |
| `--half` | off | Run the model in FP16 on the GPU (CUDA or `igpu`): faster and roughly half the memory, with no visible quality loss. Ignored (with a note) on CPU. |
| `--tile PX` | `256` | Split each frame into square tiles of this size before running the AI, which bounds memory use. **Lower it** if you run out of memory; **raise it** on a big GPU for speed. `0` processes the whole frame at once. See [Performance tuning](#performance-tuning). |
| `--tile-pad PX` | `10` | Overlap between neighbouring tiles, which hides seams. Rarely needs changing. |
| `--pre-pad PX` | `0` | Padding added around the whole frame before inference. A small value (e.g. `10`) can reduce artifacts at the very edges of the image. |
| `--torch-threads N` | `0` | CPU threads for the AI, **per worker** (OpenVINO on CPU uses it too). `0` lets PyTorch decide (usually all cores), split evenly between the workers. Setting it to your *physical* core count can help on some laptops. |
| `--workers N` | `1` | Frames the AI works on at the same time, each with its share of the CPU threads. `2` can be faster on CPUs with many cores (`framelift tune` measures it). Uses a bit more memory; the output is the same. |
| `--backend {torch,openvino}` | `torch` | What runs the AI on the CPU. `openvino` (Intel's runtime, `pip install openvino`) was ~13% faster in our tests on the default model and ~20% on `x4plus`, rounding a few values differently (at most 1 level out of 255). `framelift tune` compares them on your machine. |
| `--prefetch N` | `2` | How many pre-cleaned frames the reader thread keeps ready. Raise it (e.g. `4`–`8`) with slow profiles like `heavy_noise` on a fast GPU. Each frame costs a little RAM. |
| `--weights-dir DIR` | `weights` | Folder where model files are stored. Missing models are downloaded here automatically on first use (relative to the current directory). |

### Encoding

| Flag | Default | Description |
| --- | --- | --- |
| `--encoder {cpu,nvenc}` | `cpu` | `cpu` uses libx264 and works everywhere. `nvenc` uses the NVIDIA hardware encoder (faster, frees the CPU). If NVENC isn't usable — no NVIDIA GPU, running with `--device cpu`, or an FFmpeg build without it — framelift explains why and falls back to `cpu`. |
| `--crf N` | `20` | Quality target. **Lower = better quality and bigger files.** 18 is visually lossless for most footage, 18–22 is the sweet spot, above 26 artifacts start to show. With `nvenc` this value is passed as `-cq`. |
| `--x264-preset NAME` | `veryfast` | libx264 speed/size trade-off: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`. Slower presets give smaller files at the same quality. Since the AI is usually the bottleneck, `medium` often costs little extra time. Only used by the `cpu` encoder. |

### Diagnostics

| Flag | Default | Description |
| --- | --- | --- |
| `--debug-timings` | off | Periodically print how long each stage takes (waiting for the reader, waiting for the AI, blend, write). On CUDA it synchronizes the GPU so the numbers are real, which costs a little speed. |
| `--timing-every N` | `20` | With `--debug-timings`: report every `N` frames. |
| `--gc-every N` | `0` | With `--debug-timings`: run Python's garbage collector and release cached GPU memory every `N` frames. `0` = never. A diagnostic tool for suspected memory growth. |
| `-v`, `--verbose` | off | Show extra detail: FFmpeg commands, file paths, sanity-check values and full tracebacks on unexpected errors. |
| `-q`, `--quiet` | off | Only show warnings and errors (the progress bar stays). |

### Information

| Flag | Description |
| --- | --- |
| `--list-models` | Describe the available models and exit. |
| `--list-profiles` | Describe the pre-cleaning profiles and exit. |
| `--version` | Print the version and exit. |
| `-h`, `--help` | Show all options with examples. |

### Legacy (accepted but ignored)

| Flag | Why it still exists |
| --- | --- |
| `--frame-format` | Earlier versions saved frames as images. Frames now go straight to FFmpeg, so this does nothing; it's accepted so old commands keep running. |
| `--keep-temp` | Same story: there are no temporary frame files to keep anymore. |

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success — including a partial video saved after a single <kbd>Ctrl</kbd>+<kbd>C</kbd>. |
| `1` | Error (missing file, invalid option, FFmpeg failure, out of memory…). The message says what to do. |
| `2` | Invalid command-line usage (unknown flag, bad choice). |
| `130` | Stopped immediately with a second <kbd>Ctrl</kbd>+<kbd>C</kbd>. |

## Models

| Name | Best for | Speed | Notes |
| --- | --- | --- | --- |
| `realesr-animevideov3` *(default)* | Video in general, animation | ⚡⚡⚡ | Tiny network made for video, and by far the fastest. Start here. |
| `realesrgan-x4plus` | Real-world footage | 🐢 | The most detail, but slow: about 45x slower than the default on a CPU in our tests. Can invent textures; pair with a lower `--ai-strength`. |
| `realesrnet-x4plus` | Real-world footage, smoother look | 🐢 | Same size as x4plus, fewer invented textures. |
| `realesrgan-x4plus-anime-6B` | Anime, cartoons, illustrations | ⚡⚡ | Keeps flat colors and clean lines. |

All models are trained at 4x. framelift then resizes to your `--scale`, so you don't
have to output 4x-sized video. Weights (about 2.5 MB for the default model, 18 MB for the
anime model and 67 MB for the other two) are downloaded automatically from the official
Real-ESRGAN releases the first time a model is used.

## Pre-cleaning profiles

The AI enhances *everything*, including noise and compression blocks. A light
clean-up first usually gives a more natural result.

| Profile | What it does | Use it for | Cost |
| --- | --- | --- | --- |
| `none` *(default)* | Nothing. | Clean sources. | free |
| `minimal` | +2% contrast and saturation, a light sharpen. | Slightly dull footage. | ~free |
| `soft_camera` | Gentle edge-preserving smoothing (bilateral), mild color lift, sharpen. | Phone and webcam video with fine grain. | low |
| `old_tv` | Stronger bilateral smoothing, more contrast and color. | VHS, DVD rips, TV captures. | low |
| `heavy_noise` | Non-local-means denoising, then tone and sharpen. | Very grainy, low-light or heavily compressed video. | **high** (CPU) |

`heavy_noise` runs on the CPU in the reader thread. On a fast GPU it can become the
bottleneck; raising `--prefetch` helps a bit.

## Performance tuning

The AI step dominates the run time. The quickest way to tune it is
[`framelift tune`](#finding-the-best-settings-framelift-tune), which measures it on your
machine. Otherwise, rough guidance:

| Hardware | Suggested settings |
| --- | --- |
| NVIDIA, 8 GB+ VRAM | `--device cuda --half --tile 768` (or `--tile 0` if it fits) `--encoder nvenc` |
| NVIDIA, 4–6 GB VRAM | `--device cuda --half --tile 384` |
| Intel Iris / UHD / Arc | install OpenVINO and run `framelift tune`: it compares `--device igpu` with the CPU |
| CPU (Intel / AMD) | `--device cpu --tile 128` or `256`, default model; try `--backend openvino` and, with many cores, `--workers 2` (`framelift tune` tests both) |
| Apple | `--device cpu --tile 128` or `256`, default model, and see the thread tip below |

- **Out of memory?** Halve `--tile`. That's the main memory knob.
- **Where does the time go?** Add `--debug-timings`. A large `queue` time means the
  reader (profile) is the bottleneck; large `ai` (the time spent waiting for the AI) means
  the model is; large `write` means the encoder is (try `--encoder nvenc` or a faster
  `--x264-preset`).
- **The model matters more than anything else:** `realesr-animevideov3` was about 45x
  faster than `realesrgan-x4plus` on a CPU in our tests.
- **The default model is ~30% faster on CPU since 1.4.** It now runs in the memory layout
  that the CPU's math library prefers (channels-last); in our tests a 720x480 frame went from
  8.3 to 5.8 s. The heavier models were *slower* that way, so they're unchanged.
- **`--scale` barely affects speed.** The models always upscale 4x internally and resize to
  your scale at the end, so 1080p output costs about the same AI time as 720p.
- **The AI and the rest overlap.** The AI runs in its own thread: while it
  works on the next frame, the previous one is resized, blended and encoded. That only
  helps with cores to spare; on a single core it's a draw (the same 55–56 s for 12 frames
  in our tests). The output is bit-for-bit the same as before.
- **Two workers on many-core CPUs.** PyTorch spreads one frame over all the cores, but it
  gets less efficient as you add them, especially on laptops that mix fast and efficiency
  cores. `--workers 2` gives each of two frames half the threads instead. Whether that's
  faster depends on the CPU, so `framelift tune` measures it with two short real runs.
- **OpenVINO.** Intel's runtime is often faster than PyTorch on CPU (`--backend openvino`)
  and can use Intel GPUs (`--device igpu`). The first run compiles the model for each tile
  size (a few seconds on CPU, longer on a GPU); the result is cached in
  `weights/openvino_cache/`, so later runs start faster.
- **More CPU threads isn't always faster.** On laptops that mix performance and efficiency
  cores, a lower `--torch-threads` (say 4–6) can beat using them all. `framelift tune`
  tests this for you. Keep the laptop plugged in and out of power-saving mode, too.
- On CPU, expect roughly seconds per frame rather than frames per second. Test with
  `--max-frames` first to estimate the total time.

## Stopping, resuming and joining parts

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> **once**: framelift finishes the frame it's working on,
closes the MP4 properly and prints something like:

```
Saved a partial video /videos/out.mp4
  frames 1 → 1200 (1200 enhanced, 1200 written) in 42m10s
To continue from here, run again with:  --start-frame 1201
```

Press it **twice** to quit immediately (exit code 130). FFmpeg is still allowed to close
its file, which is left as `out.encoding_tmp.mp4`.

To continue, run the same command with a **new output name** and the suggested start:

```bash
framelift -i in.mp4 -o part2.mp4 --start-frame 1201
```

Then join the parts losslessly with FFmpeg (each part carries its own stretch of audio,
so the joined file stays in sync):

```bash
printf "file 'out.mp4'\nfile 'part2.mp4'\n" > parts.txt
ffmpeg -f concat -safe 0 -i parts.txt -c copy joined.mp4
```

> [!TIP]
> `--start-frame` together with `--max-frames` lets you split a long job into sessions
> on purpose, e.g. 5000 frames per night.

## Audio

framelift copies the input's audio into the output automatically, in sync:

- **Only the matching stretch.** With `--start-frame`, `--max-frames`, or after a
  <kbd>Ctrl</kbd>+<kbd>C</kbd>, the audio covers exactly the frames that were enhanced.
  Parts made this way can be [joined](#stopping-resuming-and-joining-parts) and stay in sync.
- **No quality loss when possible.** AAC, MP3, AC-3 and E-AC-3 are copied as they are.
  Other codecs (like the PCM audio common in AVI or MOV captures) are converted to AAC at
  192 kbps, because many players can't play them inside an MP4.
- **The video is never at risk.** The audio is added in a quick second pass after the video
  is finished. If the input has no audio, the output is silent; if adding it fails, the
  video is still saved, silent, with a warning.

Use `--no-audio` to leave it out (`audio=False` in Python). From Python, `result.audio`
tells you what happened: `"copied"`, `"converted"`, `"none"` (no audio in the input),
`"failed"` or `"disabled"`. In our tests, the audio lined up with the video to
within a millisecond, including after joining parts and after converting 30 → 60 fps.

## Python API

Everything the CLI does is available from Python, with the same option names
(`--ai-strength` → `ai_strength`).

### One call

```python
from framelift import enhance_video

result = enhance_video("in.mp4", "out.mp4", scale=2, profile="old_tv", ai_strength=0.5)

print(f"{result.frames_written} frames written in {result.elapsed_seconds:.0f}s")
if result.resume_from:
    print("Continue later with start_frame =", result.resume_from)
```

Without the AI (`--no-ai`), for example to only change the frame rate:

```python
enhance_video("in.mp4", "in_60fps.mp4", ai=False, same_resolution=True, output_fps=60)
```

With RIFE's in-between frames (`--interpolate`):

```python
result = enhance_video("in.mp4", "smooth.mp4", output_fps=60, interpolate=True)
print(result.frames_interpolated, "new frames drawn by RIFE")
```

### Reusable settings and batch processing

`VideoEnhancer` loads the model once and reuses it for every video:

```python
from pathlib import Path
from framelift import EnhanceOptions, VideoEnhancer

enhancer = VideoEnhancer(EnhanceOptions(scale=2, profile="soft_camera", device="cuda", half=True))

for clip in Path("raw").glob("*.mp4"):
    enhancer.enhance(clip, Path("enhanced") / clip.name)
```

### Stopping from code

Pass a `threading.Event`; setting it has the same effect as one Ctrl+C:

```python
import threading
from framelift import enhance_video

stop = threading.Event()
threading.Timer(600, stop.set).start()  # give up after 10 minutes, keep what's done

result = enhance_video("long.mp4", "out.mp4", stop_event=stop)
print("interrupted:", result.interrupted)
```

### Messages and progress

framelift uses the standard `logging` module under the `"framelift"` logger and prints
nothing on its own, apart from the progress bar. To see the same friendly output as the
CLI:

```python
import framelift

framelift.setup_console(verbosity=0)  # -1 = warnings only, 1 = debug
```

Pass `show_progress=False` to hide the progress bar.

### Handling errors

All errors inherit from `framelift.FrameliftError`:

| Exception | When |
| --- | --- |
| `InvalidOptionError` | An option can't work (also a `ValueError`). |
| `VideoReadError` | The input is missing, unreadable or has no frame rate. |
| `MissingDependencyError` | FFmpeg or the Real-ESRGAN packages aren't installed. |
| `OutOfMemoryError` | The model ran out of memory; retry with a smaller `tile`. |
| `EncodingError` | FFmpeg failed. |
| `BlackFrameError` | The first frame is black before or after the AI. |
| `NothingToDoError` | No frame was processed. |

```python
from framelift import EnhanceOptions, OutOfMemoryError, VideoEnhancer

options = EnhanceOptions(device="cuda", tile=1024)
while True:
    try:
        VideoEnhancer(options).enhance("in.mp4", "out.mp4")
        break
    except OutOfMemoryError:
        if options.tile <= 64:
            raise  # even tiny tiles don't fit: give up
        options.tile //= 2
        print("Out of memory, retrying with tile", options.tile)
```

### Finding settings from code

```python
from framelift import enhance_video, tune_video

report = tune_video("clip.mp4", sheet_path="clip_tune.png")
print(report.softness_label, "→ recommended:", report.recommended_preset)

for name, preset in report.presets.items():
    print(name, preset.options.model, preset.options.ai_strength, preset.estimated_seconds)

enhance_video("clip.mp4", "clip_hd.mp4", report.presets["fast"].options)
```

`report.recommended` is the recommended preset's options, and `report.model_tests` holds
what each model measurably does to the footage. Pass `options=EnhanceOptions(...)` as a
starting point and `keep={"profile", "scale"}` to pin settings you've already decided on.
`try_best=False` skips the heavier model, `content="anime"` picks the anime model for the
`best` preset, and `benchmark=False` skips the speed test. `report.real_run` holds the short
real run (`report.real_runs` has both when 1 and 2 workers were compared), each preset's
`basis` says whether its time was `"measured"` or `"extrapolated"`, and each entry of
`report.runs` says which `device` and `backend` it timed.

### Choosing the part to enhance from code

```python
from framelift import FrameRange, enhance_video, pick_range, trim_video

frames = pick_range("concert.mp4")  # opens the player window; None if cancelled
# frames = FrameRange(2701, 4500)   # or decide yourself (frames count from 1, both included)
if frames:
    cut = trim_video("concert.mp4", "song3.mp4", frames)
    enhance_video(cut.path, "song3_hd.mp4")
```

`trim_video` works without VLC; only `pick_range` needs it.

### Building blocks

The pieces are usable on their own too:

```python
import cv2
from framelift import (
    EnhanceOptions,
    apply_profile,
    estimate_noise,
    plan_run,
    probe_video,
    softness,
    suggest_size,
)

info = probe_video("in.mp4")  # fps, frame_count, width, height
plan = plan_run(info, EnhanceOptions(scale=2))  # sizes/ranges, without running anything
cleaned = apply_profile(cv2.imread("frame.png"), "old_tv")
noise = estimate_noise(cleaned)  # ~0 = clean, 10+ = very noisy
blur = softness(cleaned)  # ~0.15 = sharp, 0.5+ = soft
suggest_size(640, 360).scale  # 3.0 (360p → 1080p)
```

More in [`examples/`](examples/).

## Troubleshooting

**No sound in the output** — Check the messages at the end of the run. If the input has
no audio track, framelift says so. If adding the audio failed, it prints a warning; run
again with `-v` to see FFmpeg's error. framelift also needs `ffprobe`, which comes with
FFmpeg, to read the audio.

**`Couldn't find 'ffmpeg' on your PATH`** — Install FFmpeg and open a new terminal.
On Windows, add the folder containing `ffmpeg.exe` to `PATH`.

**`You asked for --device cuda, but PyTorch can't see a CUDA GPU`** — Either there's no
NVIDIA GPU (use `--device cpu`), or you have the CPU-only PyTorch build. Check with
`python -c "import torch; print(torch.cuda.is_available())"` and reinstall PyTorch from
[pytorch.org](https://pytorch.org/get-started/locally/).

**`Ran out of memory while upscaling`** — Lower `--tile` (256 → 128 → 64). On CUDA,
`--half` roughly halves memory use.

**`No module named 'torchvision.transforms.functional_tensor'`** — A known
incompatibility between `basicsr` and torchvision 0.17+. framelift patches it
automatically when it loads a model, so if you see it, it comes from code that imports
`basicsr` or `realesrgan` directly, not from framelift.

**`The first frame is almost completely black`** — Checked *before* the AI, it means
decoding or the profile is the problem: try `--profile none`, or re-encode the input
with FFmpeg. *After* the AI, suspect a corrupt weights file (delete it from `weights/`
to re-download) or a GPU driver issue (compare with `--device cpu`).

**`--output-fps … is lower than the input's`** — Lowering the frame rate isn't
supported. Leave `--output-fps` out, or convert the input first with
`ffmpeg -i in.mp4 -r 30 in30.mp4`.

**The range picker doesn't open** — It says what's missing: VLC (install it from
[videolan.org](https://www.videolan.org/vlc/), with the same 32/64-bit as Python) and
`pip install python-vlc`, Tkinter (`sudo apt install python3-tk` on Debian/Ubuntu), or a
desktop session. Without a desktop, use `framelift trim -i … --start … --end …`. On
macOS the window is untested; if the video stays black, cut by time instead.

**`--device igpu`: OpenVINO can't see an Intel GPU** — Install OpenVINO
(`pip install openvino`) and update the Intel graphics driver; on Linux the GPU also needs
the Intel compute runtime (`intel-opencl-icd`), and your user in the `render` group. Check
with `python -c "import openvino as ov; print(ov.Core().available_devices)"`: it should
list `GPU`.

**It's very slow** — See [Performance tuning](#performance-tuning). On CPU, this is
expected; use the default model and preview with `--max-frames`.

**The output is shorter than expected** — Some files report a wrong frame count in
their metadata. framelift processes until the decoder stops, and the summary shows what
was actually read.

## Project layout

```
framelift/
├── src/framelift/
│   ├── cli.py         # command-line interface and Ctrl+C handling
│   ├── pipeline.py    # VideoEnhancer / enhance_video: orchestrates a run
│   ├── tuning.py      # tune_video: speed test, AI-effect test, presets, sheet
│   ├── analysis.py    # noise, softness, motion and AI-effect measurements; sheet
│   ├── tune_cli.py    # the `framelift tune` subcommand
│   ├── trim.py        # FrameRange / trim_video: frame-exact, lossless cuts
│   ├── trim_cli.py    # the `framelift trim` subcommand and --pick-range
│   ├── picker.py      # pick_range: the player window with a timeline (Tk + VLC)
│   ├── planning.py    # RunPlan / EnhanceResult: sizes, ranges, frame rates
│   ├── options.py     # EnhanceOptions: every setting, validated
│   ├── video.py       # probing + background FrameReader
│   ├── filters.py     # pre-cleaning profiles (declarative recipes)
│   ├── frames.py      # resize, blend, brightness, fps cadence
│   ├── upscaler.py    # Real-ESRGAN wrapper
│   ├── interpolation.py  # RIFE in-between frames for --interpolate
│   ├── models.py      # model catalog + safe weight downloads
│   ├── device.py      # CUDA/CPU selection and PyTorch tuning
│   ├── encoder.py     # FFmpegWriter: raw frames → MP4
│   ├── console.py     # friendly terminal output that respects the progress bar
│   └── errors.py      # exception hierarchy
├── tests/             # pytest suite (runs without PyTorch)
├── examples/          # runnable usage examples
├── scripts/           # make_comparison.py: before/after GIFs for the gallery
├── docs/              # images and GIFs used by the READMEs and the gallery
├── GALLERY.md         # more before/after examples
└── main.py            # backwards-compatible entry point
```

Heavy dependencies are only imported when needed, so `framelift --help` and
`import framelift` are instant, and most of the test suite runs without PyTorch.

## Development

```bash
pip install -e ".[dev]"
pytest                 # full suite (the end-to-end tests need PyTorch + Real-ESRGAN)
pytest -m "not slow"   # fast unit tests only, no PyTorch needed
ruff check . && ruff format --check .
```

CI runs the fast suite and the linter on Python 3.9–3.12 for every push.

## Coming from the old `main.py`

Nothing breaks: every flag kept its name and default, and `python main.py …` still
works. The video was verified frame by frame (MD5 of every decoded frame) against the
original script across profiles, scales, frame-rate conversion, resume and tiling
settings. Two things differ in the output. It now keeps the audio (add `--no-audio` for
the old silent MP4). And on CPU, the default model runs about 30% faster in a
different memory layout, which rounds a handful of values differently: at most 1 level
out of 255, in about 0.002% of them, invisible but no longer bit-identical. GPU runs and
the other models are unaffected.

What changed — including a few bug fixes — is listed in the
[CHANGELOG](CHANGELOG.md).

## Credits and license

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause),
  which does the actual super-resolution. Model weights are distributed by that project
  under its terms.
- [RIFE](https://github.com/hzwer/Practical-RIFE) by Zhewei Huang et al. (MIT), which
  draws the in-between frames for `--interpolate`. framelift includes its network
  definition (version 4.26) and downloads the weights from
  [vs-rife](https://github.com/HolyWu/vs-rife) (MIT).
- [BasicSR](https://github.com/XPixelGroup/BasicSR), [PyTorch](https://pytorch.org),
  [OpenCV](https://opencv.org) and [FFmpeg](https://ffmpeg.org).

framelift itself is released under the [MIT License](LICENSE).
