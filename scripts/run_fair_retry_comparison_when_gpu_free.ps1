param(
    [string]$RepoRoot = "C:\Users\kingj\projects\rapSongData",
    [int]$MinimumFreeMiB = 8000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$root = Join-Path $RepoRoot "runs\fair_qwen3_4b_vs_olmo3_7b"
$statusPath = Join-Path $root "retry_orchestrator_status.json"
$logPath = Join-Path $root "retry_orchestrator.log"
$prompts = "configs\prompts\qwen3_4b_12line_v5_confirmation_prompts.json"
$train = "data\training\qwen3_4b_12line_section_mined_v51_prompt_aligned\train.jsonl"

function Write-RunLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line
}

function Write-Status([string]$Status, [string]$Stage, [string]$Message = "") {
    @{
        status = $Status
        stage = $Stage
        message = $Message
        updated_at = (Get-Date -Format o)
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath
}

function Invoke-Python([string[]]$Arguments, [bool]$AllowFailure = $false) {
    Write-RunLog ("COMMAND: " + $python + " " + ($Arguments -join " "))
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can promote a native program's harmless stderr
        # warnings to terminating ErrorRecord objects when the global policy is
        # Stop. Native success/failure is determined by its exit code instead.
        $ErrorActionPreference = "Continue"
        & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    Write-RunLog "EXIT: $code"
    if (-not $AllowFailure -and $code -ne 0) {
        throw "Python command failed with exit code $code"
    }
}

function Wait-ForGpu {
    Write-Status "waiting" "gpu_free" "Waiting for at least $MinimumFreeMiB MiB free VRAM."
    while ($true) {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
        $free = [int](($raw | Select-Object -First 1).Trim())
        Write-RunLog "GPU_FREE_MIB: $free"
        if ($free -ge $MinimumFreeMiB) {
            return
        }
        Start-Sleep -Seconds 15
    }
}

function Run-Sweep(
    [string]$Name,
    [string]$BaseModel,
    [string]$Revision,
    [string]$AdapterDir = ""
) {
    $dir = "runs\fair_qwen3_4b_vs_olmo3_7b\$Name"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $args = @(
        "scripts\run_qwen3_generation_sweep.py",
        "--base-model", $BaseModel,
        "--model-revision", $Revision,
        "--output-jsonl", "$dir\generations.jsonl",
        "--summary-json", "$dir\summary.json",
        "--run-manifest", "$dir\manifest.json",
        "--prompt-file", $prompts,
        "--num-candidates", "24",
        "--batch-size", "1",
        "--max-new-tokens", "320",
        "--temperature", "0.6",
        "--top-p", "0.95",
        "--top-k", "50",
        "--repetition-penalty", "1.0",
        "--no-repeat-ngram-size", "0",
        "--seed", "42",
        "--load-in-4bit",
        "--disable-thinking",
        "--block-slurs",
        "--enforce-target-line-count",
        "--underlength-retries", "2",
        "--no-resume",
        "--strict-row-seeds"
    )
    if ($AdapterDir) {
        $args += @("--adapter", "--adapter-dir", $AdapterDir)
    } else {
        $args += "--no-adapter"
    }
    Write-Status "running" $Name
    $completedSummary = $null
    if (Test-Path "$dir\summary.json") {
        try {
            $completedSummary = Get-Content "$dir\summary.json" -Raw | ConvertFrom-Json
        } catch {
            $completedSummary = $null
        }
    }
    if ($completedSummary -and $completedSummary.status -eq "complete" -and
        [int]$completedSummary.unique_output_rows -eq 24) {
        Write-RunLog "REUSE: completed generation for $Name"
    } else {
        Invoke-Python $args
    }
    Invoke-Python @(
        "scripts\rank_qwen3_quality.py",
        "--input", "$dir\generations.jsonl",
        "--prompts", $prompts,
        "--output-md", "$dir\quality_report.md",
        "--output-jsonl", "$dir\quality_ranked.jsonl",
        "--summary-json", "$dir\quality_summary.json",
        "--train-corpus", $train,
        "--top-n", "24"
    )
    Invoke-Python @(
        "scripts\evaluate_generation_outputs.py",
        "--input", "$dir\generations.jsonl",
        "--prompts", $prompts,
        "--train-corpus", $train,
        "--out", "$dir\structural_evaluation.json",
        "--sample-md", "$dir\structural_issues.md",
        "--max-samples", "24"
    )
}

function Run-Gate(
    [string]$Name,
    [string]$BaseRanked,
    [string]$CandidateRanked,
    [string]$BaseLabel,
    [string]$CandidateLabel
) {
    $dir = "runs\fair_qwen3_4b_vs_olmo3_7b\comparisons\$Name"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Invoke-Python @(
        "scripts\evaluate_balanced_quality_gate.py",
        "--base-ranked", $BaseRanked,
        "--adapter-ranked", $CandidateRanked,
        "--prompts", $prompts,
        "--profile", "confirmation",
        "--base-label", $BaseLabel,
        "--adapter-label", $CandidateLabel,
        "--output-json", "$dir\gate.json",
        "--output-md", "$dir\gate.md"
    ) $true
}

try {
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Write-RunLog "Retry comparison orchestrator started."
    Wait-ForGpu

    Run-Sweep "retry_qwen_v51_native_24" "Qwen/Qwen3-4B" `
        "1cfa9a7208912126459214e8b04321603b3df60c" `
        "model\artifacts\qwen3-4b-12line-section-v51-prompt-aligned-e1-lr2e5-s20260712"
    Run-Sweep "retry_olmo_base_native_24" "allenai/Olmo-3-7B-Instruct" `
        "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
    Run-Sweep "retry_olmo_v51_native_24" "allenai/Olmo-3-7B-Instruct" `
        "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc" `
        "model\artifacts\olmo3-7b-12line-section-v51-e1-lr2e5-s20260712"

    Run-Gate "retry_base_vs_base" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_qwen_base_native_24\quality_ranked.jsonl" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_olmo_base_native_24\quality_ranked.jsonl" `
        "qwen_retry_base" "olmo_retry_base"
    Run-Gate "retry_adapter_vs_adapter" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_qwen_v51_native_24\quality_ranked.jsonl" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_olmo_v51_native_24\quality_ranked.jsonl" `
        "qwen_retry_v51" "olmo_retry_v51"
    Run-Gate "retry_qwen_training_effect" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_qwen_base_native_24\quality_ranked.jsonl" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_qwen_v51_native_24\quality_ranked.jsonl" `
        "qwen_retry_base" "qwen_retry_v51"
    Run-Gate "retry_olmo_training_effect" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_olmo_base_native_24\quality_ranked.jsonl" `
        "runs\fair_qwen3_4b_vs_olmo3_7b\retry_olmo_v51_native_24\quality_ranked.jsonl" `
        "olmo_retry_base" "olmo_retry_v51"

    Write-Status "complete" "done" "All retry comparison runs and gates completed."
    Write-RunLog "Retry comparison orchestrator completed."
} catch {
    Write-Status "failed" "error" $_.Exception.Message
    Write-RunLog ("FAILED: " + $_.Exception.ToString())
    exit 1
}
