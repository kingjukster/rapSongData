"""Compatibility entry point for the quality-max local research lane."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rap_song_data.quality_max.runner import main


if __name__ == "__main__":
    main()
