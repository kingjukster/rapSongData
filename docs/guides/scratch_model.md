# Private scratch lyric-model workflow

This workflow is a parallel research track. It does not replace or modify the
existing Qwen/OLMo QLoRA datasets, adapters, generation defaults, or evaluation
artifacts.

The source lyrics have unknown redistribution rights. Every generated manifest
therefore marks the corpus and model as private research and blocks public
release. Raw data is never modified.

## Runtime

Run through WSL while using the CUDA-capable repository virtual environment:

```powershell
.\scripts\run_rap_scratch_safe.ps1 build-corpus
.\scripts\run_rap_scratch_safe.ps1 train-tokenizer
```

The compatibility launcher executes the following interpreter through WSL:

```text
/mnt/d/Users/kingj/projects/rapSongData/.venv/Scripts/python.exe
```

## Acceptance sequence

1. Build and audit the corpus. The full gate requires at least 500,000 retained
   songs.
2. Train the 32K byte-level BPE tokenizer and fixed 512-token shards. The full
   gate requires at least 300M unique training tokens.
3. Run a 20-step smoke test.
4. Resume that smoke checkpoint for at least one additional step.
5. Only after those checks pass, run the bounded 20-hour pretraining job.
6. Run one structured-SFT epoch with a two-hour cap.
7. Run fixed evaluation, extraction scanning, MAUVE, and create the blinded
   human-review packet.

Example smoke command:

```powershell
.\scripts\run_rap_scratch_safe.ps1 pretrain `
  --data-dir data/scratch/v1 `
  --output-dir model/artifacts/scratch-30m-v1-smoke `
  --config configs/training/scratch_30m_pretrain_v1.json `
  --smoke
```

The full command is identical without `--smoke`. Do not run it when either
data acceptance gate is false.

## Versioning and compliance development

Freeze loadable model releases before branching new SFT or control variants:

```powershell
.\scripts\run_rap_scratch_safe.ps1 freeze-checkpoint `
  --source model/artifacts/scratch-30m-v1-sft-full/checkpoint-00010150 `
  --name scratch-30m-sft-v1 `
  --kind sft
```

Use `analyze-compliance` to replay saved generations without spending GPU time,
then use `sweep-decoding` on a fixed stratified subset. Sweep budgets are always
capped to the positions remaining inside the model's 512-token context. Report
legacy evaluator compliance, corrected native compliance, and controlled-system
compliance separately.

Run the selected experimental router against semantic request records with:

```powershell
.\scripts\run_rap_scratch_safe.ps1 generate-controlled `
  --config configs/generation/scratch_30m_sft_v1_1_control.json `
  --requests configs/prompts/scratch_30m_router_smoke.json `
  --output-dir model/reports/scratch-30m-router-run
```

The router retains raw output beside controlled output and reports whether its
single underlength retry was attempted and accepted. It never extends beyond
the trained context window.

## Outputs

- `data/scratch/v1/corpus_manifest.json`
- `data/scratch/v1/tokenization_manifest.json`
- versioned binary token shards for base and SFT modes
- immutable timestamped checkpoints
- `run_summary.json`, `run_summary.md`, exact command, resolved configuration,
  and metric time series in every training output
- generation, MAUVE, extraction, and blinded-review artifacts in the evaluation
  output directory
