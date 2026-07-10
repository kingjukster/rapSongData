"""Compatibility wrapper for :mod:`rap_song_data.corpus.cleaning`."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rap_song_data.corpus.cleaning import *  # noqa: F401,F403,E402
from rap_song_data.corpus.cleaning import main  # noqa: E402


if __name__ == "__main__":
    main()
