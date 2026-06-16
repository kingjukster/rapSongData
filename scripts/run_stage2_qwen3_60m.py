"""Run the Stage 2 practical Qwen2.5-7B local CUDA benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.run_stage2_qwen25_60m as runner


if __name__ == "__main__":
    runner.main()
