[CmdletBinding()]
param(
    [int]$MaxCycles = 6,
    [int]$SleepSeconds = 900,
    [int]$BatchRecords = 20,
    [int]$RawTargetTokens = 1000000,
    [int]$CandidateLimit = 5000,
    [double]$DelaySeconds = 2.0,
    [string]$RunId = (Get-Date -Format 'yyyyMMdd_HHmmss'),
    [switch]$SkipTokenizer,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runner = Join-Path $RepoRoot 'scripts/run_rap_scratch_safe.ps1'
$RunDir = Join-Path $RepoRoot "data/scratch/passive_gutenberg/$RunId"
$LogPath = Join-Path $RunDir 'passive_loop.jsonl'
$StopPath = Join-Path $RunDir 'STOP'
$PidPath = Join-Path $RunDir 'pid.txt'

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
[string]$PID | Set-Content -LiteralPath $PidPath -Encoding UTF8

function Write-LoopEvent {
    param([string]$Event, [hashtable]$Data = @{})
    $payload = [ordered]@{
        generated_at = (Get-Date).ToUniversalTime().ToString('o')
        event = $Event
        run_id = $RunId
        pid = $PID
    }
    foreach ($key in $Data.Keys) {
        $payload[$key] = $Data[$key]
    }
    ($payload | ConvertTo-Json -Depth 20 -Compress) | Add-Content -LiteralPath $LogPath -Encoding UTF8
}

function Convert-CommandOutputToJson {
    param([string[]]$Output)
    $text = ($Output -join "`n").Trim()
    $start = $text.IndexOf('{')
    $end = $text.LastIndexOf('}')
    if ($start -lt 0 -or $end -lt $start) {
        throw "Could not find JSON object in command output: $text"
    }
    return $text.Substring($start, $end - $start + 1) | ConvertFrom-Json
}

function Quote-ProcessArgument {
    param([string]$Value)
    '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-ScratchJson {
    param([string[]]$Arguments, [string]$Name)
    Write-LoopEvent -Event 'command_started' -Data @{ name = $Name; arguments = $Arguments }
    $started = Get-Date
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = 'powershell.exe'
    $process.StartInfo.WorkingDirectory = $RepoRoot
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $process.StartInfo.CreateNoWindow = $true
    $processArgs = @(
        '-NoProfile'
        '-ExecutionPolicy'
        'Bypass'
        '-File'
        (Quote-ProcessArgument $Runner)
    ) + ($Arguments | ForEach-Object { Quote-ProcessArgument $_ })
    $process.StartInfo.Arguments = ($processArgs -join ' ')
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $duration = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
    $stdoutPath = Join-Path $RunDir ("{0:yyyyMMdd_HHmmss}_{1}.log" -f (Get-Date), $Name)
    @(
        '--- stdout ---'
        $stdout
        '--- stderr ---'
        $stderr
    ) | Set-Content -LiteralPath $stdoutPath -Encoding UTF8
    if ($exitCode -ne 0) {
        Write-LoopEvent -Event 'command_failed' -Data @{
            name = $Name
            exit_code = $exitCode
            wall_seconds = $duration
            stdout_path = $stdoutPath
        }
        throw "$Name failed with exit code $exitCode"
    }
    $json = Convert-CommandOutputToJson -Output @($stdout)
    Write-LoopEvent -Event 'command_completed' -Data @{
        name = $Name
        wall_seconds = $duration
        stdout_path = $stdoutPath
        stderr_bytes = $stderr.Length
    }
    return $json
}

function Wait-OrStop {
    param([int]$Seconds)
    $remaining = $Seconds
    while ($remaining -gt 0) {
        if (Test-Path -LiteralPath $StopPath) {
            return $true
        }
        $chunk = [Math]::Min(10, $remaining)
        Start-Sleep -Seconds $chunk
        $remaining -= $chunk
    }
    return $false
}

if ($Once) {
    $MaxCycles = 1
    $SleepSeconds = 0
}

Write-LoopEvent -Event 'loop_started' -Data @{
    max_cycles = $MaxCycles
    sleep_seconds = $SleepSeconds
    batch_records = $BatchRecords
    raw_target_tokens = $RawTargetTokens
    candidate_limit = $CandidateLimit
    delay_seconds = $DelaySeconds
    skip_tokenizer = [bool]$SkipTokenizer
    stop_path = $StopPath
}

$cycle = 0
while ($MaxCycles -le 0 -or $cycle -lt $MaxCycles) {
    if (Test-Path -LiteralPath $StopPath) {
        Write-LoopEvent -Event 'loop_stopped_by_file' -Data @{ cycle = $cycle; stop_path = $StopPath }
        break
    }
    $cycle += 1
    try {
        Write-LoopEvent -Event 'cycle_started' -Data @{ cycle = $cycle }
        $acquire = Invoke-ScratchJson -Name "cycle_${cycle}_acquire" -Arguments @(
            'acquire-gutenberg',
            '--max-records', [string]$BatchRecords,
            '--raw-target-tokens', [string]$RawTargetTokens,
            '--candidate-limit', [string]$CandidateLimit,
            '--delay-seconds', [string]$DelaySeconds,
            '--skip-seen'
        )
        $snapshot = [string]$acquire.snapshot_id
        $accepted = [int]$acquire.counts.accepted_records
        $rawTokens = [int]$acquire.counts.approx_tokens

        Invoke-ScratchJson -Name "cycle_${cycle}_ingest" -Arguments @(
            'ingest-source',
            '--source', 'project_gutenberg_songbooks',
            '--snapshot-dir', "data/corpus_lake/normalized/project_gutenberg_songbooks/$snapshot",
            '--reason', "passive-gutenberg-cycle-$cycle-acquired"
        ) | Out-Null
        Invoke-ScratchJson -Name "cycle_${cycle}_audit" -Arguments @(
            'audit-source',
            '--source', 'project_gutenberg_songbooks',
            '--allow-conditional',
            '--reason', "passive-gutenberg-cycle-$cycle-needs-item-review"
        ) | Out-Null
        $review = Invoke-ScratchJson -Name "cycle_${cycle}_review" -Arguments @(
            'review-gutenberg',
            '--snapshot-id', $snapshot,
            '--allow-partial-admission'
        )
        $approved = [int]$review.counts.approved_records
        $approvedTokens = [int]$review.counts.approved_approx_tokens
        $quarantined = [int]$review.counts.quarantined_records

        if ($approved -gt 0) {
            Invoke-ScratchJson -Name "cycle_${cycle}_admit" -Arguments @(
                'admit-source',
                '--source', 'project_gutenberg_songbooks',
                '--rights-evidence', 'data/scratch/catalog/v2/sources/project_gutenberg_songbooks/review_manifest.json',
                '--reason', "passive-gutenberg-cycle-$cycle-admit-approved-subset"
            ) | Out-Null
            Invoke-ScratchJson -Name "cycle_${cycle}_build_core_profile" -Arguments @(
                'build-corpus',
                '--profile', 'scratch-core-open-v1',
                '--reason', "passive-gutenberg-cycle-$cycle-approved-subset"
            ) | Out-Null
            Invoke-ScratchJson -Name "cycle_${cycle}_build_private_profile" -Arguments @(
                'build-corpus',
                '--profile', 'scratch-private-extended-v1',
                '--reason', "passive-gutenberg-cycle-$cycle-approved-subset"
            ) | Out-Null
            Invoke-ScratchJson -Name "cycle_${cycle}_verify_core_profile" -Arguments @(
                'verify-profile',
                '--profile', 'scratch-core-open-v1'
            ) | Out-Null
            Invoke-ScratchJson -Name "cycle_${cycle}_verify_private_profile" -Arguments @(
                'verify-profile',
                '--profile', 'scratch-private-extended-v1'
            ) | Out-Null
            $materialized = Invoke-ScratchJson -Name "cycle_${cycle}_materialize_core" -Arguments @(
                'materialize-profile',
                '--profile', 'scratch-core-open-v1',
                '--force'
            )
            $trainTokens = $null
            $tokenGatePassed = $null
            if (-not $SkipTokenizer) {
                $tokenized = Invoke-ScratchJson -Name "cycle_${cycle}_tokenize_core" -Arguments @(
                    'train-tokenizer',
                    '--corpus-dir', 'data/scratch/profiles/scratch-core-open-v1',
                    '--output-dir', 'data/scratch/profiles/scratch-core-open-v1',
                    '--force'
                )
                $trainTokens = [int]$tokenized.acceptance.unique_training_tokens
                $tokenGatePassed = [bool]$tokenized.acceptance.token_gate_passed
            }
            Write-LoopEvent -Event 'cycle_completed' -Data @{
                cycle = $cycle
                snapshot_id = $snapshot
                accepted_records = $accepted
                raw_approx_tokens = $rawTokens
                approved_records = $approved
                approved_approx_tokens = $approvedTokens
                quarantined_records = $quarantined
                cumulative_retained_records = [int]$materialized.counts.retained
                cumulative_train_tokens = $trainTokens
                token_gate_passed = $tokenGatePassed
            }
        } else {
            Write-LoopEvent -Event 'cycle_completed_no_admission' -Data @{
                cycle = $cycle
                snapshot_id = $snapshot
                accepted_records = $accepted
                raw_approx_tokens = $rawTokens
                approved_records = $approved
                approved_approx_tokens = $approvedTokens
                quarantined_records = $quarantined
            }
        }
    } catch {
        Write-LoopEvent -Event 'cycle_failed' -Data @{
            cycle = $cycle
            error = $_.Exception.Message
        }
    }

    if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
        break
    }
    if ($SleepSeconds -gt 0) {
        Write-LoopEvent -Event 'sleep_started' -Data @{ cycle = $cycle; sleep_seconds = $SleepSeconds }
        if (Wait-OrStop -Seconds $SleepSeconds) {
            Write-LoopEvent -Event 'loop_stopped_by_file' -Data @{ cycle = $cycle; stop_path = $StopPath }
            break
        }
    }
}

Write-LoopEvent -Event 'loop_finished' -Data @{ cycles_completed = $cycle }
