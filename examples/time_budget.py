"""Work on a long video for a fixed amount of time, then save and stop.

Run it again later: it picks up exactly where it left off, writing numbered parts.

    python examples/time_budget.py long_video.mp4 30   # 30 minutes per session
"""

import sys
import threading
from pathlib import Path

import framelift
from framelift import EnhanceOptions, VideoEnhancer

video = Path(sys.argv[1])
minutes = float(sys.argv[2])
framelift.setup_console()

# Each finished part records where the next one should start.
parts = sorted(video.parent.glob(f"{video.stem}_part*.mp4"))
progress_file = video.with_suffix(".next_frame")
start_frame = int(progress_file.read_text()) if progress_file.exists() else 1

stop = threading.Event()
timer = threading.Timer(minutes * 60, stop.set)
timer.start()

try:
    result = VideoEnhancer(EnhanceOptions(start_frame=start_frame)).enhance(
        video, video.with_name(f"{video.stem}_part{len(parts) + 1:03d}.mp4"), stop_event=stop
    )
finally:
    timer.cancel()

if result.resume_from:
    progress_file.write_text(str(result.resume_from))
    print(f"Session over. Next run continues from frame {result.resume_from}.")
else:
    progress_file.unlink(missing_ok=True)
    print("All frames done! Join the parts with FFmpeg's concat demuxer (see README).")
