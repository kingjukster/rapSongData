# OpenAI Dataset Review

Use `scripts/review_sft_data_with_openai.py` to review SFT JSONL records in batches and write per-record quality decisions.

The raw dataset is never modified. The workflow is:

1. Review records into `data/reviews/*.jsonl`.
2. Apply reviews to create a new filtered JSONL.
3. Train against the filtered output only after inspecting the summary.

Default target:

```powershell
.\.cuda-venv\Scripts\python.exe scripts\review_sft_data_with_openai.py review `
  --input data\sft\rap_mixed_sft_train.jsonl `
  --output-reviews data\reviews\rap_mixed_sft_openai_reviews.jsonl `
  --candidate-mode heuristic `
  --batch-size 8 `
  --model gpt-5.4-mini
```

Dry run without API calls:

```powershell
.\.cuda-venv\Scripts\python.exe scripts\review_sft_data_with_openai.py review `
  --input data\sft\rap_mixed_sft_train.jsonl `
  --limit 20 `
  --batch-size 4 `
  --dry-run
```

Create a filtered dataset from saved reviews:

```powershell
.\.cuda-venv\Scripts\python.exe scripts\review_sft_data_with_openai.py apply `
  --input data\sft\rap_mixed_sft_train.jsonl `
  --reviews data\reviews\rap_mixed_sft_openai_reviews.jsonl `
  --output-kept data\sft\rap_mixed_sft_openai_filtered_train.jsonl `
  --output-dropped data\reviews\rap_mixed_sft_openai_dropped.jsonl `
  --summary-output data\reviews\rap_mixed_sft_openai_filter_summary.json
```

Notes:

- Set `OPENAI_API_KEY` in `.env` or the environment before live review.
- Override the model with `--model` or `OPENAI_REVIEW_MODEL`.
- `--candidate-mode heuristic` reviews only likely-bad rows first, which is the cheapest useful starting point.
- `--candidate-mode all` reviews every row.
- `--resume` is enabled by default, so interrupted review runs append only missing record IDs.
- The prompt explicitly does not drop records solely for profanity, rap slang, dark imagery, or the word `nigga`.
