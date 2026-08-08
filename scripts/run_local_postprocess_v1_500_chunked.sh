#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/kingj/projects/rapSongData

PY="/mnt/c/Users/kingj/projects/rapSongData/.cuda-venv/Scripts/python.exe"
OUT_DIR="reports/generation_sweeps/local_postprocess_v1_500"
CHUNK_DIR="$OUT_DIR/chunks"

mkdir -p "$CHUNK_DIR" "$OUT_DIR"

for chunk in 1 2 3 4 5; do
  seed=$((20260800 + chunk))
  "$PY" model/run_fixed_generation_eval.py \
    --adapter-dir model/artifacts/qwen2.5-7b-rap-lora-full-song-openai-controlled-v2-384-500 \
    --output-md "$CHUNK_DIR/chunk_${chunk}.md" \
    --output-jsonl "$CHUNK_DIR/chunk_${chunk}.jsonl" \
    --run-summary-dir "$CHUNK_DIR/chunk_${chunk}_summary" \
    --title "local_postprocess_v1_500_chunk_${chunk}" \
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
    --drop-unfinished-final-line \
    --aggressive-drop-dangling-final-line \
    --trim-to-requested-line-count \
    --trim-hook-lines \
    --hook-line-cap 8 \
    --normalize-whitespace
done

"$PY" scripts/combine_jsonl.py \
  --input "$CHUNK_DIR/chunk_1.jsonl" \
  --input "$CHUNK_DIR/chunk_2.jsonl" \
  --input "$CHUNK_DIR/chunk_3.jsonl" \
  --input "$CHUNK_DIR/chunk_4.jsonl" \
  --input "$CHUNK_DIR/chunk_5.jsonl" \
  --output "$OUT_DIR/sweep.jsonl"

"$PY" scripts/summarize_generation_jsonl.py "$OUT_DIR/sweep.jsonl" > "$OUT_DIR/summary_metrics.json"

"$PY" scripts/curate_generation_pool.py \
  --input "$OUT_DIR/sweep.jsonl" \
  --out-dir "$OUT_DIR" \
  --top-per-prompt 5

"$PY" scripts/compare_generation_sweeps.py \
  --old reports/generation_sweeps/v2_384_500_hookcap96_trim_variety500_max230_control_chunked.jsonl \
  --new "$OUT_DIR/sweep.jsonl" \
  --out "$OUT_DIR/comparison_report.md"
