"""The smallest useful script: enhance one video and report what happened.

python examples/basic.py input.mp4 output.mp4
"""

import sys

import framelift

framelift.setup_console()  # show the same friendly messages as the CLI

result = framelift.enhance_video(
    sys.argv[1],
    sys.argv[2],
    scale=2,
    profile="minimal",
    ai_strength=0.5,
)

print(f"\n{result.frames_written} frames in {result.elapsed_seconds:.0f}s → {result.output_path}")
