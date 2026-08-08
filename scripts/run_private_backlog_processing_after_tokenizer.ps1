param(
    [string]$RunId = "",
    [string]$RepoRoot = "D:\Users\kingj\projects\rapSongData",
    [string]$Python = "D:\Users\kingj\projects\rapSongData\.venv\Scripts\python.exe",
    [string]$ActiveProfileDir = "C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v1_20260716_0742",
    [string]$OverflowRoot = "C:\Users\kingj\rapSongData_overflow_review",
    [int]$PollSeconds = 60,
    [switch]$SkipLrclib,
    [switch]$SkipMaterializeTokenize
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "20260716_backlog_after_tokenizer_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$RunDir = Join-Path $OverflowRoot ("backlog_processing\" + $RunId)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$StatusPath = Join-Path $RunDir "backlog_worker.jsonl"

function Write-Event {
    param(
        [string]$Event,
        [hashtable]$Data = @{}
    )
    $payload = [ordered]@{
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        event = $Event
        run_id = $RunId
    }
    foreach ($key in $Data.Keys) {
        $payload[$key] = $Data[$key]
    }
    ($payload | ConvertTo-Json -Compress -Depth 8) | Add-Content -Path $StatusPath -Encoding UTF8
}

function Get-FreeGiB {
    param([string]$DriveName)
    $drive = Get-PSDrive -Name $DriveName
    return [math]::Round($drive.Free / 1GB, 3)
}

function Invoke-Logged {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogPath
    )
    Write-Event -Event ($Name + "_started") -Data @{ command = ($Arguments -join " "); log_path = $LogPath }
    Push-Location $RepoRoot
    try {
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-Event -Event ($Name + "_completed") -Data @{ exit_code = $exitCode; log_path = $LogPath }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

function Get-TokenizerProcesses {
    $escaped = $ActiveProfileDir.Replace('\', '\\')
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*rap_scratch.py*" -and
        $_.CommandLine -like "*train-tokenizer*" -and
        $_.CommandLine -like "*$ActiveProfileDir*"
    })
}

Write-Event -Event "worker_started" -Data @{
    repo_root = $RepoRoot
    active_profile_dir = $ActiveProfileDir
    c_free_gib = (Get-FreeGiB -DriveName "C")
    d_free_gib = (Get-FreeGiB -DriveName "D")
}

$tokenManifest = Join-Path $ActiveProfileDir "tokenization_manifest.json"
while ($true) {
    $tokenizerProcesses = Get-TokenizerProcesses
    if ((Test-Path -LiteralPath $tokenManifest) -or $tokenizerProcesses.Count -eq 0) {
        Write-Event -Event "tokenizer_gate_open" -Data @{
            tokenization_manifest_exists = (Test-Path -LiteralPath $tokenManifest)
            active_tokenizer_processes = $tokenizerProcesses.Count
        }
        break
    }
    Write-Event -Event "tokenizer_gate_waiting" -Data @{
        active_tokenizer_processes = $tokenizerProcesses.Count
        c_free_gib = (Get-FreeGiB -DriveName "C")
        d_free_gib = (Get-FreeGiB -DriveName "D")
    }
    Start-Sleep -Seconds $PollSeconds
}

$SourceConfig = Join-Path $RunDir "private_lyrics_backlog_sources.json"
Invoke-Logged -Name "discover_backlog_sources" -LogPath (Join-Path $RunDir "discover_backlog_sources.log") -Arguments @(
    "scripts\discover_private_lyrics_backlog_sources.py",
    "--output-config", $SourceConfig
)

$SeedParquets = @(
    "data\corpus_lake\normalized\private_lyrics_exact_dedupe\20260715_genius_family_v1\admitted_unique.parquet",
    "data\corpus_lake\normalized\private_lyrics_exact_dedupe\20260715_remaining_local_v1\admitted_unique.parquet",
    "data\corpus_lake\raw\common_crawl_lyrics\20260715_commoncrawl_lyrics_v1\admitted_web_unique.parquet",
    "data\corpus_lake\raw\common_crawl_lyrics\20260715_commoncrawl_lyrics_serious_20260715_190814\admitted_web_unique.parquet",
    "data\corpus_lake\raw\common_crawl_lyrics\20260715_commoncrawl_songmeanings_20260715_192922\admitted_web_unique.parquet",
    "data\corpus_lake\raw\common_crawl_lyrics\20260715_commoncrawl_lyrics_backfill_20260715_174002\admitted_web_unique.parquet",
    "data\corpus_lake\raw\common_crawl_wet_lyrics\20260716_commoncrawl_wet_lyrics_v1_20260715_223722\admitted_wet_unique.parquet",
    "data\corpus_lake\normalized\hf_lyrics_midi_extracted\20260716_lyrics_midi_full_20260715_223056\admitted_unique.parquet"
) | Where-Object { Test-Path -LiteralPath (Join-Path $RepoRoot $_) }

$GenericSnapshot = $RunId + "_generic"
$GenericOutputRoot = Join-Path $OverflowRoot "normalized\private_lyrics_backlog_exact_dedupe"
$GenericArgs = @(
    "scripts\build_private_lyrics_dedupe_index.py",
    "--source-config", $SourceConfig,
    "--output-root", $GenericOutputRoot,
    "--snapshot-id", $GenericSnapshot,
    "--index-backend", "sqlite",
    "--chunksize", "25000",
    "--write-batch-size", "25000",
    "--progress-every", "250000"
)
foreach ($seed in $SeedParquets) {
    $GenericArgs += @("--seed-admitted-parquet", (Join-Path $RepoRoot $seed))
}

Invoke-Logged -Name "normalize_generic_backlog" -LogPath (Join-Path $RunDir "normalize_generic_backlog.log") -Arguments $GenericArgs
$GenericAdmitted = Join-Path $GenericOutputRoot ($GenericSnapshot + "\admitted_unique.parquet")

$LrclibAdmitted = $null
if (-not $SkipLrclib) {
    $LrclibSnapshot = $RunId + "_lrclib"
    $LrclibOutputRoot = Join-Path $OverflowRoot "normalized\lrclib_exact_dedupe"
    $LrclibDump = Join-Path $OverflowRoot "lrclib_db_dumps\20260716_lrclib_dump_v1_20260715_2320\lrclib-db-dump-20260624T025818Z.sqlite3.gz"
    $LrclibArgs = @(
        "scripts\normalize_lrclib_dump.py",
        "--compressed-dump", $LrclibDump,
        "--output-root", $LrclibOutputRoot,
        "--snapshot-id", $LrclibSnapshot,
        "--batch-size", "25000",
        "--write-batch-size", "25000",
        "--progress-every", "100000"
    )
    foreach ($seed in $SeedParquets) {
        $LrclibArgs += @("--seed-admitted-parquet", (Join-Path $RepoRoot $seed))
    }
    if (Test-Path -LiteralPath $GenericAdmitted) {
        $LrclibArgs += @("--seed-admitted-parquet", $GenericAdmitted)
    }
    foreach ($split in @("train.jsonl", "validation.jsonl", "test.jsonl")) {
        $splitPath = Join-Path $ActiveProfileDir $split
        if (Test-Path -LiteralPath $splitPath) {
            $LrclibArgs += @("--seed-jsonl", $splitPath)
        }
    }
    Invoke-Logged -Name "normalize_lrclib" -LogPath (Join-Path $RunDir "normalize_lrclib.log") -Arguments $LrclibArgs
    $LrclibAdmitted = Join-Path $LrclibOutputRoot ($LrclibSnapshot + "\admitted_unique.parquet")
}

if (-not $SkipMaterializeTokenize) {
    $cFree = Get-FreeGiB -DriveName "C"
    if ($cFree -lt 140) {
        Write-Event -Event "materialize_tokenize_skipped_low_disk" -Data @{ c_free_gib = $cFree }
    }
    else {
        $ProfileV2 = Join-Path $OverflowRoot ("scratch_profiles\scratch-private-lyric-lake-v2_backlog_" + $RunId)
        $V2RunDir = Join-Path $ProfileV2 "_runs\materialize_tokenize"
        New-Item -ItemType Directory -Force -Path $V2RunDir | Out-Null
        $MaterializeArgs = @(
            "scripts\materialize_private_lyric_lake_profile.py",
            "--output-dir", $ProfileV2,
            "--force"
        )
        if (Test-Path -LiteralPath $GenericAdmitted) {
            $MaterializeArgs += @("--extra-source", ("private_backlog_generic|" + $GenericAdmitted))
        }
        if ($LrclibAdmitted -and (Test-Path -LiteralPath $LrclibAdmitted)) {
            $MaterializeArgs += @("--extra-source", ("lrclib_private|" + $LrclibAdmitted + "|trackName,name,title|artistName,artist|lyrics"))
        }
        Invoke-Logged -Name "materialize_v2_profile" -LogPath (Join-Path $V2RunDir "materialize.log") -Arguments $MaterializeArgs
        Invoke-Logged -Name "tokenize_v2_profile" -LogPath (Join-Path $V2RunDir "tokenize.log") -Arguments @(
            "rap_scratch.py",
            "train-tokenizer",
            "--corpus-dir", $ProfileV2,
            "--output-dir", $ProfileV2,
            "--vocab-size", "32000",
            "--sequence-length", "512",
            "--min-train-tokens", "300000000",
            "--force"
        )
    }
}

Write-Event -Event "worker_completed" -Data @{
    c_free_gib = (Get-FreeGiB -DriveName "C")
    d_free_gib = (Get-FreeGiB -DriveName "D")
}
