param(
  [string]$Python = ".\.cuda-venv\Scripts\python.exe",
  [string]$Root = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location $Root

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runTag = "dpo_control_probe_$timestamp"
$logDir = "model\artifacts\$runTag"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$pairPath = "data\preferences\rap_dpo_control_pairs.jsonl"
$summaryPath = "data\preferences\rap_dpo_control_pairs_summary.json"
$pairLog = Join-Path $logDir "build_dpo_pairs.log"
$trainDir = "model\artifacts\qwen2.5-7b-rap-lora-mixed-sft-500-dpo-control-probe"
$trainLog = Join-Path $logDir "dpo_train.log"
$evalLog = Join-Path $logDir "dpo_eval.log"

Write-Host "Building DPO control pairs..."
& $Python scripts\build_dpo_control_pairs.py `
  --output-path $pairPath `
  --summary-path $summaryPath `
  --seed 20260617 `
  2>&1 | Tee-Object -FilePath $pairLog

Write-Host "Running DPO probe..."
& $Python scripts\train_dpo_control_probe.py `
  --pair-path $pairPath `
  --output-dir $trainDir `
  --run-summary (Join-Path $trainDir "run_summary.json") `
  2>&1 | Tee-Object -FilePath $trainLog

Write-Host "Running fixed eval comparison..."
& $Python scripts\evaluate_dpo_probe.py `
  --models original_500=model\artifacts\qwen2.5-7b-rap-lora-mixed-sft-384-500-simple\checkpoint-500 `
  --models filtered_500=model\artifacts\qwen2.5-7b-rap-lora-mixed-sft-filtered-384-500-probe\checkpoint-500 `
  --models dpo_500=$trainDir\adapter `
  --output-jsonl (Join-Path $logDir "dpo_probe_eval.jsonl") `
  --output-md (Join-Path $logDir "dpo_probe_eval.md") `
  2>&1 | Tee-Object -FilePath $evalLog

Write-Host "All outputs logged under $logDir"
