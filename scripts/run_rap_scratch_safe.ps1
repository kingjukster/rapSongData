[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScratchArguments
)

$ErrorActionPreference = 'Stop'
$WslRepo = '/mnt/d/Users/kingj/projects/rapSongData'
$Python = "$WslRepo/.venv/Scripts/python.exe"
$Entrypoint = 'rap_scratch.py'

if (-not $ScratchArguments -or $ScratchArguments.Count -eq 0) {
    throw 'Pass a rap-scratch subcommand and arguments, for example: build-corpus --limit 1000'
}

& wsl.exe --cd $WslRepo $Python $Entrypoint @ScratchArguments
if ($LASTEXITCODE -ne 0) {
    throw "rap-scratch failed with exit code $LASTEXITCODE"
}
