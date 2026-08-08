#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/kingj/projects/rapSongData

PY="/mnt/c/Users/kingj/projects/rapSongData/.cuda-venv/Scripts/python.exe"

"$PY" model/run_fixed_generation_eval.py \
  --adapter-dir model/artifacts/qwen2.5-7b-rap-lora-full-song-openai-controlled-v2-384-500 \
  --output-md reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control.md \
  --output-jsonl reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control.jsonl \
  --run-summary-dir reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_summary \
  --title v2_384_500_hookcap96_trim_variety500_max230_control \
  --max-new-tokens 230 \
  --hook-max-new-tokens 96 \
  --temperature 0.74 \
  --top-p 0.86 \
  --repetition-penalty 1.20 \
  --seed 20260701 \
  --samples-per-prompt 100 \
  --sample-batch-size 5 \
  --load-in-4bit \
  --disable-thinking \
  --no-block-slurs \
  --drop-unfinished-final-line

"$PY" scripts/summarize_generation_jsonl.py \
  reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control.jsonl \
  > reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_summary/summary_metrics.json

"$PY" scripts/rank_generation_candidates.py \
  --input-jsonl reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control.jsonl \
  --output-md reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_ranked.md \
  --output-jsonl reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_ranked.jsonl \
  --top-per-prompt 20 \
  --min-lines 6 \
  --max-line-words 34
