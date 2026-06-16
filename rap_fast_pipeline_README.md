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
