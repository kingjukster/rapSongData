[CmdletBinding()]
param(
    [string]$RunId
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BaseDir = Join-Path $RepoRoot 'data/scratch/passive_wikisource'

if ($RunId) {
    $RunDir = Join-Path $BaseDir $RunId
} else {
    $RunDir = Get-ChildItem -LiteralPath $BaseDir -Directory -ErrorAction Stop |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not $RunDir) {
    throw "No passive Wikisource run directory found under $BaseDir"
}

$StopPath = Join-Path $RunDir 'STOP'
"stop requested $(Get-Date -Format o)" | Set-Content -LiteralPath $StopPath -Encoding UTF8
Write-Output "Stop requested for passive Wikisource run: $RunDir"
