#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/kingj/projects/rapSongData

OUT_DIR="model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-ai-labeled-balanced-384-2000"
RUN_DIR="${OUT_DIR}/run_wsl_launch"
LOG="${RUN_DIR}/train.log"
CONFIG="model/configs/local_cuda_qwen2_5_7b_mixed_sft_ai_labeled_balanced_384_2000.json"
PY="./.cuda-venv/Scripts/python.exe"

mkdir -p "${RUN_DIR}"
cp data/labels/rap_mixed_sft_ai_labeled_balanced_summary.json \
  "${RUN_DIR}/rap_mixed_sft_ai_labeled_balanced_summary.json"

COMMAND="${PY} model/train_local_cuda.py --config ${CONFIG} --skip-smoke-check"
printf '%s\n' "${COMMAND}" > "${RUN_DIR}/training_command.txt"

nohup ${COMMAND} > "${LOG}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" > "${RUN_DIR}/pid.txt"
printf 'pid=%s\nrun_dir=%s\nlog=%s\n' "${PID}" "${RUN_DIR}" "${LOG}"
