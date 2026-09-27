"""Enhance every video in a folder, loading the model only once.

    python examples/batch_folder.py raw_videos/ enhanced_videos/

Already-finished videos are skipped, so you can stop and re-run it at any time.
"""

import sys
from pathlib import Path

import framelift
from framelift import EnhanceOptions, FrameliftError, VideoEnhancer

source_dir, target_dir = Path(sys.argv[1]), Path(sys.argv[2])
framelift.setup_console(verbosity=-1)  # warnings only; the progress bar is enough

enhancer = VideoEnhancer(EnhanceOptions(scale=2, profile="soft_camera", ai_strength=0.4))
videos = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".mov"})

for number, video in enumerate(videos, start=1):
    target = target_dir / f"{video.stem}_hd.mp4"
    if target.exists():
        print(f"[{number}/{len(videos)}] {video.name}: already done, skipping")
        continue

    print(f"[{number}/{len(videos)}] {video.name}")
    try:
        enhancer.enhance(video, target)
    except FrameliftError as exc:  # one bad file shouldn't stop the batch
        print(f"  skipped: {exc}")
