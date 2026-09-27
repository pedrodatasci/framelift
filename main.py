"""Backwards-compatible entry point.

Old commands like ``python main.py --input in.mp4 --output out.mp4`` keep
working, even without installing the package. New code should use the
``framelift`` command or ``python -m framelift`` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from framelift.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
