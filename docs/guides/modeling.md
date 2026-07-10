# Rap Lyric Model Package

This folder contains the model-preparation layer for the cleaned rap lyrics corpus. It does not move or duplicate the large source datasets.

## Source Data

The cleaning pipeline writes the verified model-ready corpus here:

```text
data/model_ready/rap_lyrics_training_dataset.parquet
```

That Parquet file remains the authoritative input for model training data generation.

## Corpus Cleaning And Quality Audit

Before the next local QLoRA run, build the categorized corpus and run the repeatable cleaning/audit pass:

```powershell
python scripts/build_categorized_rap_corpus.py --clean --dedupe both --min-quality 0.70
```

You can also run the cleaner directly against an existing categorized corpus:

```powershell
rap-clean-corpus --source data/rap_english_clean_categorized_with_families.parquet --dedupe both --min-quality 0.70
```

Use audit-only mode when you want reports and review queues without writing train text:

```powershell
python scripts/build_categorized_rap_corpus.py --audit-only --dedupe both
```

The cleaner preserves raw data and writes derived artifacts under:

```text
data/cleaned/categorized_rap_corpus_cleaned.jsonl
data/cleaned/categorized_rap_corpus_train.jsonl
data/cleaned/categorized_rap_corpus_train.txt
data/cleaned/categorized_rap_corpus_validation.txt
data/cleaned/categorized_rap_corpus_review.jsonl
data/cleaned/categorized_rap_corpus_quarantine.jsonl
data/cleaned/categorized_rap_corpus_dropped.jsonl
data/cleaned/corpus_cleaning_summary.json
```

Reports are written under `reports/`:

```text
reports/corpus_cleaning_summary.json
reports/corpus_cleaning_report.md
reports/corpus_quality_by_rap_family.csv
reports/corpus_quality_by_rap_category.csv
reports/corpus_duplicates.csv
reports/corpus_before_after_samples.jsonl
```

Quality tiers:

- `gold`: cleanest records, included in training by default.
- `silver`: meets `--min-quality`, included in training by default.
- `bronze`: usable but noisier, excluded unless `--include-bronze` is passed.
- `review`: excluded and written to the review queue.
- `quarantine`: excluded and written to the quarantine queue.
- `drop`: excluded and written to the dropped queue, commonly for exact/near duplicates or unusable text.

The cleaner label-downs weak records instead of silently deleting them. Inspect `data/cleaned/categorized_rap_corpus_review.jsonl`, `data/cleaned/categorized_rap_corpus_quarantine.jsonl`, and `reports/corpus_before_after_samples.jsonl` before scaling training.

## Build Training JSONL

Run from the project root:

```powershell
rap-build-training-data
```

Optional overrides:

```powershell
rap-build-training-data --config configs/datasets/dataset_config.json
rap-build-training-data --source data/model_ready/rap_lyrics_training_dataset.parquet --output-dir model/data
```

The builder writes:

```text
model/data/train.jsonl
model/data/validation.jsonl
model/data/test.jsonl
model/data/dataset_manifest.json
```

## Training Format

Each JSONL row contains:

```json
{
  "id": "stable_hash",
  "task": "generate_song",
  "training_text": "...",
  "metadata": {
    "title": "...",
    "artist_clean": "...",
    "rap_family": "...",
    "rap_category": "...",
    "year": 2015,
    "log_views": 10.4
  }
}
```

The generated `training_text` uses control tokens for title, artist, rap family, rap category, year, views, and lyrics. `generate_verse` examples also include `structure`, `max_words_per_bar`, `theme`, `keywords`, and `rules` fields so the LoRA sees explicit short-verse instruction examples instead of only full-song continuation text.

Lyric targets are structurally annotated so the model can learn where the verse starts and ends, and where each bar starts. These markers are added as real tokenizer special tokens during training:

```text
<|lyrics|>
<|verse_start|>
<|bar_start|>first lyric line
<|bar_start|>second lyric line
<|verse_end|>
<|end|>
```

The special-token training path uses `configs/datasets/dataset_config.special_tokens.json`, which builds a verse-focused dataset under `model/data_special_tokens/`. The training script adds `<|verse_start|>`, `<|verse_end|>`, and `<|bar_start|>` to the tokenizer, resizes embeddings, and trains only those new token embeddings via PEFT `trainable_token_indices`.

The builder also supports two targeted training tasks:

```text
generate_clean_scene_verse
generate_architectural_verse
```

`generate_clean_scene_verse` mines stricter work/night/city scene chunks from the existing cleaned corpus and rejects explicit, violent, party, romance, ad-lib, and source-artifact patterns. `generate_architectural_verse` uses curated song metadata from `configs/datasets/lyrical_architecture_seeds.json` to tag matching corpus rows with architecture controls such as `narrative_architect`, `technical_density`, `abstract_metaphorical`, and `introspective_commentary`. These tasks are repeated in the generated JSONL with conservative repeat counts so the next fine-tune sees them often enough to matter.

## Full-Verse Dataset

For higher verse structure quality, prefer the full-verse path over fixed 12-line chunks. It extracts complete sections labeled as verses, keeps only clean 10-25 bar full verses, and assigns theme labels only when confidence is strong enough. Weak theme matches are labeled `general_vibe` instead of forcing a noisy category.

Build the full-verse section parquet:

```powershell
rap-build-verse-sections --source data/rap_english_clean_categorized_with_families.parquet --output model/data/full_verse_sections.parquet
```

Build JSONL from only those full verses:

```powershell
rap-build-training-data --config configs/datasets/dataset_config.full_verses.json
```

This writes:

```text
model/data_full_verses/train.jsonl
model/data_full_verses/validation.jsonl
model/data_full_verses/test.jsonl
model/data_full_verses/dataset_manifest.json
```

Each record uses the same special-token target format, plus full-verse controls:

```text
<|bar_count|>16
<|target_bars|>16
<|bar_count_range|>10-25
<|theme|>general_vibe
<|theme_confidence|>low
```

Use this dataset when the main goal is stronger 10-25 bar structure and fewer chorus/hook drifts.

## Fine-Tuning Target

The intended training target is a QLoRA fine-tune of a causal language model, with `Qwen/Qwen2.5-7B-Instruct` as the default base model. See:

```text
configs/training/qlora_config.example.json
```

## Local CUDA Training

This machine has an NVIDIA RTX 5070, so the primary path is now local CUDA instead of Runpod endpoints. Build the JSONL data first, then train directly from local files:

```powershell
rap-build-training-data --config configs/datasets/dataset_config.special_tokens.json
rap-train-local --config configs/training/local_cuda_config.example.json --train-path model/data_special_tokens/train.jsonl --validation-path model/data_special_tokens/validation.jsonl --add-structural-special-tokens --max-steps 50
```

When `data/cleaned/categorized_rap_corpus_train.txt` exists, `rap-train-local` prefers it automatically and logs the corpus path, record count, estimated tokens, included tiers, and report path at startup. It also copies `data/cleaned/corpus_cleaning_summary.json` into the LoRA output directory as `corpus_cleaning_summary.json`.

To force an older JSONL dataset, pass explicit paths:

```powershell
rap-train-local --train-path model/data_full_verses/train.jsonl --validation-path model/data_full_verses/validation.jsonl
```

For a faster cleaned-corpus run on the RTX 5070, build shorter chunk records from the gold/silver cleaned corpus:

```powershell
python scripts/build_cleaned_chunk_corpus.py --target-lines 16 --min-lines 8 --max-words 360 --max-chunks-per-record 4
```

This writes:

```text
data/cleaned/chunked/categorized_rap_corpus_train_chunks.txt
data/cleaned/chunked/categorized_rap_corpus_validation_chunks.txt
data/cleaned/chunked/cleaned_chunk_manifest.json
```

The chunked config keeps the same base model and LoRA shape, but uses `sequence_length=768` and `gradient_accumulation_steps=4`:

```powershell
rap-train-local --config configs/training/local_cuda_cleaned_chunks_fast.json --max-steps 1000 --timing-log-steps 25
```

This is not an apples-to-apples full-song baseline. It is the faster cleaned verse/chunk path for local iteration when full cleaned songs are too slow at the 1024-token cap.

The trainer prints periodic `[timing]` JSON lines with seconds per step, estimated examples/sec, estimated tokens/sec, elapsed minutes, and GPU memory. Control the frequency with:

```powershell
rap-train-local --timing-log-steps 5
```

Each local training run now also writes `training_summary.json` into the output artifact directory. It captures the core dataset/config values, trainer metrics, and a compact timing summary so runs can be compared without re-reading the full console log.

For full-verse training:

```powershell
rap-build-verse-sections --source data/rap_english_clean_categorized_with_families.parquet --output model/data/full_verse_sections.parquet
rap-build-training-data --config configs/datasets/dataset_config.full_verses.json
rap-train-local --config configs/training/local_cuda_config.example.json --train-path model/data_full_verses/train.jsonl --validation-path model/data_full_verses/validation.jsonl --output-dir model/artifacts/qwen2.5-7b-rap-lora-full-verses --add-structural-special-tokens --max-steps 50
```

Local defaults live here:

```text
configs/training/local_cuda_config.example.json
```

The local config uses 4-bit loading by default. That is the expected path for a 7B model on a 12 GB GPU. Install the local CUDA dependencies with:

```powershell
pip install -r requirements-local-cuda.txt
```

If PyTorch does not see CUDA after that, install the current Windows CUDA PyTorch wheel from the official PyTorch selector, then rerun:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Generate from the local adapter:

```powershell
rap-generate-local --adapter-dir model/artifacts/qwen2.5-7b-rap-lora-local --title "Night Shift" --theme "working late, loneliness, ambition, city lights" --keywords "night, shift, city, lights, work, ambition"
```

Local generation also prints a `[timing]` line with model load seconds, generation seconds, generated tokens, tokens/sec, and peak GPU memory.

For repeatable quality checks against a saved adapter, run the fixed-prompt eval:

```powershell
rap-evaluate-fixed --adapter-dir model/artifacts/qwen2.5-7b-rap-lora-cleaned-chunks-768-ga4-1000 --output-md reports/generation_eval_cleaned_1000.md --output-jsonl reports/generation_eval_cleaned_1000.jsonl
```

The report now includes a prompt-adherence summary plus per-prompt metrics for exact line-count matching, hook compactness, repeated-line ratio, and no-slur compliance. The eval blocks known scraped artifact phrases and known slur token sequences by default; use `--no-block-slurs` only when you need strict legacy comparability with older generation runs.

Use `--no-load-in-4bit` only if you are testing a much smaller base model or have enough VRAM.

## Runpod Flash Legacy

The Runpod Flash scripts are still available as a cloud fallback. They are no longer the primary workflow for this machine. Flash lets you write Python locally and execute `@Endpoint` functions remotely on Runpod GPUs.

Install and authenticate:

```powershell
pip install runpod-flash
flash login
```

There is also a `.env` option for local authentication:

```text
RUNPOD_API_KEY=...
```

Optional Slack notifications use an incoming webhook URL:

```text
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

When this is set, the Runpod training module sends training start, periodic progress, completion, and failure messages. The generation module sends generation start, completion, and failure messages with a short lyrics preview.

First run the GPU smoke test:

```powershell
flash deploy --python-version 3.12
python -m rap_song_data.training.runpod_flash_smoke
```

The deploy step registers the smoke endpoint in your Runpod Flash app. The endpoint uses `workers=(0, 1)`, so it should not keep a warm GPU running after the job finishes. The root `.gitignore` excludes the local datasets from the Flash build.

For QLoRA training, the remote worker needs downloadable dataset files. Because the files contain lyrics, keep them private. The easiest path is a private Hugging Face dataset repo.

Create a token at Hugging Face with read/write access, then add it to `.env`:

```text
HF_TOKEN=hf_...
```

Log in locally:

```powershell
hf auth login
```

Create a private dataset repo and upload the JSONL files:

```powershell
hf repo create rap-lyrics-jsonl --type dataset --private
hf upload YOUR_HF_USERNAME/rap-lyrics-jsonl model/data/train.jsonl train.jsonl --repo-type dataset
hf upload YOUR_HF_USERNAME/rap-lyrics-jsonl model/data/validation.jsonl validation.jsonl --repo-type dataset
```

Those commands upload:

```text
model/data/train.jsonl
model/data/validation.jsonl
```

Then run:

```powershell
$env:PYTHONIOENCODING="utf-8"
flash deploy --python-version 3.12
python -m rap_song_data.training.runpod_flash --hf-dataset-repo "YOUR_HF_USERNAME/rap-lyrics-jsonl" --hf-output-repo "YOUR_HF_USERNAME/rap-lyrics-lora-adapter" --max-steps 50
```

To control Slack progress frequency during training:

```powershell
python -m rap_song_data.training.runpod_flash --hf-dataset-repo "YOUR_HF_USERNAME/rap-lyrics-jsonl" --hf-output-repo "YOUR_HF_USERNAME/rap-lyrics-lora-adapter" --max-steps 1000 --slack-progress-steps 100
```

If you prefer signed URLs instead of Hugging Face, you can still run:

```powershell
python -m rap_song_data.training.runpod_flash --train-url "https://..." --validation-url "https://..." --max-steps 50
```

Training endpoint defaults live here:

```text
configs/training/runpod_flash_config.example.json
```

The example config uses a persistent Runpod network volume so adapters can be saved under:

```text
/runpod-volume/rap-lyrics/adapters/qwen2.5-7b-rap-lora
```

The local JSONL files are intentionally ignored by `model/.gitignore` and are not copied into the Flash build artifact.

After training uploads the adapter, download it locally:

```powershell
hf download YOUR_HF_USERNAME/rap-lyrics-lora-adapter --local-dir model/artifacts/rap-lyrics-lora-adapter
```
