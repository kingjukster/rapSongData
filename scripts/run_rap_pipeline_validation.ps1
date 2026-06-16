param(
  [string]$DataRoot = "C:\Users\kingj\projects\rapSongData",
  [Alias("Input")]
  [string]$InputPath = "data/raw/your_file.jsonl",
  [string]$Output = "data/processed/rap_sections_labeled.parquet",
  [string]$RunDir = "data/run_logs",
  [string]$GenerationOut = "data/sft/rap_generation_sft.jsonl",
  [string]$MutationOut = "data/sft/rap_mutation_sft.jsonl",
  [string]$PreferenceOut = "data/preferences/rap_quality_pairs.jsonl",
  [string]$AuditOut = "data/reports/label_audit.md",
  [switch]$IncludeRisk,
  [string]$TrainScript = "model/train_local_cuda.py",
  [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $DataRoot

if ([string]::IsNullOrWhiteSpace($Python) -or $Python -eq "python") {
  $venvPy = Join-Path $DataRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPy) {
    $Python = $venvPy
  }
}

if (-not (Test-Path $InputPath)) {
  throw "Input file not found: $InputPath"
}
if ((Get-Item $InputPath).Length -le 0) {
  throw "Input file is empty: $InputPath"
}

function Invoke-PipelineStep {
  param(
    [Parameter(Mandatory=$true)][string]$Label,
    [Parameter(Mandatory=$true)][string]$Command
  )
  Write-Host "== Step $Label =="
  Write-Host $Command
  Invoke-Expression $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE."
  }
}

function Invoke-InspectorScript {
  param(
    [Parameter(Mandatory=$true)][string]$Label,
    [Parameter(Mandatory=$true)][string]$PythonCode
  )
  $tmp = Join-Path $env:TEMP ("rap_pipeline_tmp_" + [guid]::NewGuid().ToString("N") + ".py")
  Set-Content -Path $tmp -Value $PythonCode -Encoding UTF8
  & $Python $tmp
  $code = $LASTEXITCODE
  Remove-Item $tmp -Force
  if ($code -ne 0) {
    throw "$Label failed with exit code $code."
  }
}

Invoke-PipelineStep "1: curate (smoke)" "$Python rap_fast_pipeline.py curate --input `"$InputPath`" --output `"$Output`" --label-provider none --smoke"

Invoke-InspectorScript "Step 2: inspect parquet" @'
import json
p = "data/processed/rap_sections_labeled.parquet"
try:
    import pandas as pd

    df = pd.read_parquet(p)
    print("rows:", len(df))
    print("columns:", len(df.columns))
    print("section types:")
    print(df["section_type"].value_counts(dropna=False).head(20))
    print("\nquality:")
    print(df["quality_score"].describe())
    print("\nduplicate clusters:")
    print(df["duplicate_cluster_id"].nunique())
    print("\nnull-heavy columns:")
    print((df.isna().mean().sort_values(ascending=False).head(20) * 100).round(2))
except ModuleNotFoundError:
    try:
        import polars as pl
    except ModuleNotFoundError:
        print("Missing dependency for parquet inspection.")
        print("Install one: pip install polars pyarrow")
        print("or include pandas: pip install pandas pyarrow")
        raise SystemExit(3)

    df = pl.read_parquet(p)
    print("rows:", df.height)
    print("columns:", len(df.columns))
    print("section types:")
    print(
        df.select(pl.col("section_type"))
        .drop_nulls()
        .groupby("section_type")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(20)
    )
    print("\nquality:")
    if "quality_score" in df.columns:
        qdf = df.select(pl.col("quality_score")).describe()
        print(qdf)
    else:
        print("quality_score column missing")
    print("\nduplicate clusters:")
    if "duplicate_cluster_id" in df.columns:
        print(df.select(pl.col("duplicate_cluster_id").n_unique()).item())
    else:
        print("duplicate_cluster_id column missing")
    print("\nnull-heavy columns:")
    if df.height == 0:
        print([])
    else:
        null_rates = {
            col: round((df.select(pl.col(col).is_null().mean() * 100).item()), 2)
            for col in df.columns
        }
        top_null = sorted(null_rates.items(), key=lambda kv: kv[1], reverse=True)[:20]
        print(json.dumps(top_null, indent=2))
'@

if ($IncludeRisk) {
  $buildRiskArg = "--include-risk"
} else {
  $buildRiskArg = ""
}
Invoke-PipelineStep "3: build datasets" "$Python rap_fast_pipeline.py build-datasets --input `"$Output`" --generation-out `"$GenerationOut`" --mutation-out `"$MutationOut`" --preference-out `"$PreferenceOut`" $buildRiskArg"

Invoke-InspectorScript "Step 4: inspect JSONL outputs" @'
import json
from pathlib import Path
for path in [
    "data/sft/rap_generation_sft.jsonl",
    "data/sft/rap_mutation_sft.jsonl",
    "data/preferences/rap_quality_pairs.jsonl",
]:
    p = Path(path)
    print("\n", path)
    print("exists:", p.exists())
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        print("rows:", len(lines))
        if lines:
            obj = json.loads(lines[0])
            print("keys:", obj.keys())
            print(json.dumps(obj, indent=2)[:1000])
'@

Invoke-PipelineStep "5: smoke train" "$Python rap_fast_pipeline.py train --train-file `"$GenerationOut`" --train-script `"$TrainScript`" --run-dir `"$RunDir`" --smoke"

Invoke-PipelineStep "6: audit report" "$Python rap_fast_pipeline.py audit --input `"$Output`" --out `"$AuditOut`" --samples-per-bucket 20"

Write-Host "== Complete. Review outputs in:"
Write-Host " - $Output"
Write-Host " - $GenerationOut"
Write-Host " - $MutationOut"
Write-Host " - $PreferenceOut"
Write-Host " - $RunDir"
Write-Host " - $AuditOut"

Pop-Location
