"""Allows ``python -m framelift``."""

import sys

from .cli import main

sys.exit(main())
