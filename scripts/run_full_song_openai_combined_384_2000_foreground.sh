#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/kingj/projects/rapSongData

OUT_DIR="model/artifacts/qwen2.5-7b-rap-lora-full-song-openai-combined-384-2000"
PY="/mnt/c/Users/kingj/projects/rapSongData/.cuda-venv/Scripts/python.exe"
CONFIG="model/configs/local_cuda_qwen2_5_7b_full_song_openai_combined_384_2000.json"

mkdir -p "$OUT_DIR"
printf '%s\n' "$PY -u model/train_local_cuda.py --config $CONFIG" > "$OUT_DIR/training_command.sh"
printf '%s\n' "$(date -Is)" > "$OUT_DIR/foreground_started_at.txt"

"$PY" -u model/train_local_cuda.py --config "$CONFIG" 2>&1 | tee "$OUT_DIR/training.log"
