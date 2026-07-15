[CmdletBinding()]
param()

function Invoke-WithScratchProfileRefreshLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$Owner,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Body,
        [int]$PollSeconds = 10,
        [int]$StaleMinutes = 180
    )

    $lockDir = Join-Path $RepoRoot 'data/scratch/profile_refresh.lock'
    $lockInfoPath = Join-Path $lockDir 'owner.json'
    $acquired = $false
    while (-not $acquired) {
        try {
            New-Item -ItemType Directory -Path $lockDir -ErrorAction Stop | Out-Null
            $payload = [ordered]@{
                owner = $Owner
                pid = $PID
                acquired_at = (Get-Date).ToUniversalTime().ToString('o')
                host = $env:COMPUTERNAME
            }
            $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $lockInfoPath -Encoding UTF8
            $acquired = $true
        } catch {
            $lock = Get-Item -LiteralPath $lockDir -ErrorAction SilentlyContinue
            if ($lock -and ((Get-Date) - $lock.LastWriteTime).TotalMinutes -gt $StaleMinutes) {
                Remove-Item -LiteralPath $lockDir -Recurse -Force -ErrorAction SilentlyContinue
                continue
            }
            Start-Sleep -Seconds $PollSeconds
        }
    }

    try {
        & $Body
    } finally {
        Remove-Item -LiteralPath $lockDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
