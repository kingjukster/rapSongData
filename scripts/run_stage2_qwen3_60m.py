"""Run the Stage 2 practical Qwen3 local CUDA benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.run_stage2_qwen25_60m as runner


runner.BASE_MODEL = "Qwen/Qwen3-8B"
runner.RUN_TITLE = "Stage 2 Qwen3 60-Minute Practical Benchmark"
runner.GENERATION_TITLE = "Stage 2 Qwen3 512 60m Generation Eval"
runner.CONFIG = Path("model/configs/local_cuda_qwen3_8b_stage2_512_60m.json")
runner.OUTPUT_DIR = Path("model/artifacts/stage2-qwen3-8b-cleaned-chunks-512-60m")
runner.GEN_MD = Path("reports/stage2_qwen3_8b_512_60m_generation.md")
runner.GEN_JSONL = Path("reports/stage2_qwen3_8b_512_60m_generation.jsonl")


if __name__ == "__main__":
    runner.main()
