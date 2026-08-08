# rap_fast_pipeline usage

The new pipeline is implemented in `rap_fast_pipeline.py`.

### 1) Curate lyrics into canonical schema

```bash
python rap_fast_pipeline.py curate \
  --input data/raw/your_raw_dataset.jsonl \
  --output data/processed/rap_sections_labeled.parquet \
  --label-provider none \
  --smoke
```

### Safe path for raw `song_lyrics.csv` (recommended)

For the original 9GB CSV source, run the CSV categorization + cleaning step first, then pipeline validation:

```powershell
.\scripts\run_rap_pipeline_from_csv_safe.ps1 -InputCsv data/song_lyrics.csv -IncludeRisk
```

That wrapper:
- Builds `data/rap_english_clean_categorized_with_families.parquet`
- Writes `data/cleaned/categorized_rap_corpus_cleaned.jsonl` via `clean_corpus`
- Reuses existing cleaned artifacts when unchanged
- Runs `run_rap_pipeline_validation.ps1` with smoke defaults

Use `-PrepareOnly` to stop after corpus build.

Outputs:
- `data/processed/rap_sections_labeled.parquet`
- `data/processed/rap_sections_labeled_stats.json`
- `.cache/rap_fast_pipeline_cache.json`

### 2) Build three dataset products

```bash
python rap_fast_pipeline.py build-datasets \
  --input data/processed/rap_sections_labeled.parquet \
  --generation-out data/sft/rap_generation_sft.jsonl \
  --mutation-out data/sft/rap_mutation_sft.jsonl \
  --preference-out data/preferences/rap_quality_pairs.jsonl
```

### Optional audit report

```bash
python rap_fast_pipeline.py audit \
  --input data/processed/rap_sections_labeled.parquet \
  --out data/reports/label_audit.md \
  --samples-per-bucket 20
```

### 3) Quick smoke test then baseline run

```bash
python rap_fast_pipeline.py train \
  --train-file data/sft/rap_generation_sft.jsonl \
  --run-dir data/run_logs \
  --smoke

python rap_fast_pipeline.py train \
  --train-file data/sft/rap_generation_sft.jsonl \
  --run-dir data/run_logs \
  --max-steps 1000 \
  --sequence-length 768
```

### 4) Controlled generation with telemetry

```bash
python rap_fast_pipeline.py generate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-file path/to/prompt.txt \
  --run-dir data/run_logs/generation
```

### 5) Mandatory logs per run
- `run_summary.md`
- `run_summary.json`
- `run_metrics_timeseries.csv`
- `generation_result.json` / `generation_result.txt` for generation runs

Defaults are aligned to your speed-first policy:
- 4-bit + bf16 path is expected by your trainer entrypoint.
- sequence length defaults: 512 for `--smoke`, 768 default baseline.
- gradient accumulation default: 4 (raise to 8 only when requested).
- optional `--comparability_note=smoke/default` is injected into command args.
