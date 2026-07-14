#!/usr/bin/env python3
"""Repository-local entry point for the private scratch-model workflow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rap_song_data.scratch.cli import main


if __name__ == "__main__":
    main()
