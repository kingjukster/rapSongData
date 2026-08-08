#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/kingj/projects/rapSongData

PY="/mnt/c/Users/kingj/projects/rapSongData/.cuda-venv/Scripts/python.exe"
OUT_DIR="reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_chunks"
COMBINED_PREFIX="reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_chunked"

mkdir -p "$OUT_DIR" "${COMBINED_PREFIX}_summary"

for chunk in 1 2 3 4 5; do
  seed=$((20260700 + chunk))
  "$PY" model/run_fixed_generation_eval.py \
    --adapter-dir model/artifacts/qwen2.5-7b-rap-lora-full-song-openai-controlled-v2-384-500 \
    --output-md "$OUT_DIR/chunk_${chunk}.md" \
    --output-jsonl "$OUT_DIR/chunk_${chunk}.jsonl" \
    --run-summary-dir "$OUT_DIR/chunk_${chunk}_summary" \
    --title "v2_384_500_hookcap96_trim_variety500_max230_control_chunk_${chunk}" \
    --max-new-tokens 230 \
    --hook-max-new-tokens 96 \
    --temperature 0.74 \
    --top-p 0.86 \
    --repetition-penalty 1.20 \
    --seed "$seed" \
    --samples-per-prompt 20 \
    --sample-batch-size 5 \
    --load-in-4bit \
    --disable-thinking \
    --no-block-slurs \
    --drop-unfinished-final-line
done

"$PY" scripts/combine_jsonl.py \
  --input "$OUT_DIR/chunk_1.jsonl" \
  --input "$OUT_DIR/chunk_2.jsonl" \
  --input "$OUT_DIR/chunk_3.jsonl" \
  --input "$OUT_DIR/chunk_4.jsonl" \
  --input "$OUT_DIR/chunk_5.jsonl" \
  --output "${COMBINED_PREFIX}.jsonl"

"$PY" scripts/summarize_generation_jsonl.py \
  "${COMBINED_PREFIX}.jsonl" \
  > "${COMBINED_PREFIX}_summary/summary_metrics.json"

"$PY" scripts/rank_generation_candidates.py \
  --input-jsonl "${COMBINED_PREFIX}.jsonl" \
  --output-md "${COMBINED_PREFIX}_ranked.md" \
  --output-jsonl "${COMBINED_PREFIX}_ranked.jsonl" \
  --top-per-prompt 30 \
  --min-lines 6 \
  --max-line-words 34
