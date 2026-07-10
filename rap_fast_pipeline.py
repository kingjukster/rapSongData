"""Compatibility wrapper for the packaged pipeline.

New code should import :mod:`rap_song_data.cli` or use ``rap-pipeline``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rap_song_data.cli import *  # noqa: F401,F403,E402
from rap_song_data.cli import (  # noqa: E402
    _build_mutation_dataset,
    _build_preference_dataset,
    main,
)


if __name__ == "__main__":
    main()
