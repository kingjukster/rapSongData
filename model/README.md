# Rap Lyric Model Package

This folder contains the model-preparation layer for the cleaned rap lyrics corpus. It does not move or duplicate the large source datasets.

## Source Data

The cleaning pipeline writes the verified model-ready corpus here:

```text
data/model_ready/rap_lyrics_training_dataset.parquet
```

That Parquet file remains the authoritative input for model training data generation.

## Build Training JSONL

Run from the project root:

```powershell
python model/build_training_data.py
```

Optional overrides:

```powershell
python model/build_training_data.py --config model/configs/dataset_config.json
python model/build_training_data.py --source data/model_ready/rap_lyrics_training_dataset.parquet --output-dir model/data
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

The special-token training path uses `model/configs/dataset_config.special_tokens.json`, which builds a verse-focused dataset under `model/data_special_tokens/`. The training script adds `<|verse_start|>`, `<|verse_end|>`, and `<|bar_start|>` to the tokenizer, resizes embeddings, and trains only those new token embeddings via PEFT `trainable_token_indices`.

The builder also supports two targeted training tasks:

```text
generate_clean_scene_verse
generate_architectural_verse
```

`generate_clean_scene_verse` mines stricter work/night/city scene chunks from the existing cleaned corpus and rejects explicit, violent, party, romance, ad-lib, and source-artifact patterns. `generate_architectural_verse` uses curated song metadata from `model/configs/lyrical_architecture_seeds.json` to tag matching corpus rows with architecture controls such as `narrative_architect`, `technical_density`, `abstract_metaphorical`, and `introspective_commentary`. These tasks are repeated in the generated JSONL with conservative repeat counts so the next fine-tune sees them often enough to matter.

## Full-Verse Dataset

For higher verse structure quality, prefer the full-verse path over fixed 12-line chunks. It extracts complete sections labeled as verses, keeps only clean 10-25 bar full verses, and assigns theme labels only when confidence is strong enough. Weak theme matches are labeled `general_vibe` instead of forcing a noisy category.

Build the full-verse section parquet:

```powershell
python model/build_verse_sections.py --source data/rap_english_clean_categorized_with_families.parquet --output model/data/full_verse_sections.parquet
```

Build JSONL from only those full verses:

```powershell
python model/build_training_data.py --config model/configs/dataset_config.full_verses.json
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

The intended first training target is a QLoRA fine-tune of a causal language model, with `Qwen/Qwen2.5-7B-Instruct` as the default base model. See:

```text
model/configs/qlora_config.example.json
```

This package only prepares data and configs. It does not download models or start training.

## Runpod Flash

Flash lets you write Python locally and execute `@Endpoint` functions remotely on Runpod GPUs.

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

When this is set, `model/runpod_flash_train.py` sends training start, periodic progress, completion, and failure messages. `model/runpod_flash_generate.py` sends generation start, completion, and failure messages with a short lyrics preview.

First run the GPU smoke test:

```powershell
flash deploy --python-version 3.12
python model/runpod_flash_smoke.py
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
python model/runpod_flash_train.py --hf-dataset-repo "YOUR_HF_USERNAME/rap-lyrics-jsonl" --hf-output-repo "YOUR_HF_USERNAME/rap-lyrics-lora-adapter" --max-steps 50
```

To control Slack progress frequency during training:

```powershell
python model/runpod_flash_train.py --hf-dataset-repo "YOUR_HF_USERNAME/rap-lyrics-jsonl" --hf-output-repo "YOUR_HF_USERNAME/rap-lyrics-lora-adapter" --max-steps 1000 --slack-progress-steps 100
```

If you prefer signed URLs instead of Hugging Face, you can still run:

```powershell
python model/runpod_flash_train.py --train-url "https://..." --validation-url "https://..." --max-steps 50
```

Training endpoint defaults live here:

```text
model/configs/runpod_flash_config.example.json
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
