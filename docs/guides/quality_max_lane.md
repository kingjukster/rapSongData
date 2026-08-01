# Quality-Max Local Research Lane

This lane prioritizes the strongest visible lyric output that is practical on the
local RTX 4090. It is intentionally separate from the frozen 12-line comparison
baseline: candidate count, revision passes, model size, prompt format, and length
distribution can all change quality and therefore break strict comparability.

## What exists in v1

- Qwen3-14B loaded in 4-bit NF4 with BF16 compute and SDPA.
- Batched candidate generation sized for the 24 GB GPU.
- A second-pass verse improver applied to the best preliminary candidate.
- Four modes: original generation, style transfer, lyric mutation, and verse improvement.
- A soft 12-36 bar objective. Outputs are never truncated or rejected solely for length.
- Transparent local pre-ranking. Every candidate remains in `candidates.jsonl` for review.
- Exact command, resolved configuration, timing, throughput, peak VRAM, model, adapter,
  and output paths saved under `runs/quality_max/`.

The v1 default uses the existing Qwen3-14B 12-line adapter as an initial baseline.
That adapter has a 12-line training bias, so the lane must compare it with `--base-only`
before treating it as the best flexible-length system.

The adapter does not contain the base-model weights. The first run therefore requires
either a complete Hugging Face cache or network access to resolve `Qwen/Qwen3-14B`.
For a portable offline setup, set `model_path` in the configuration to a complete local
snapshot and run with `--local-files-only`. Failed launches retain `run_summary.json`,
`run_summary.md`, and `generation.log` instead of silently leaving an ambiguous run.

## Smoke first

Run from the repository root with the CUDA-capable environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py --smoke
```

The smoke creates two candidates without a revision pass. Review its `run_summary.md`,
`candidates.jsonl`, and `best.txt` before running the full profile:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py
```

## Flexible bar requests

The range and target influence prompting and ranking but do not impose a hard cutoff:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py `
  --min-bars 12 --max-bars 36 --target-bars 28 `
  --theme "rebuilding after a public failure" `
  --keywords "empty venue, voicemail, sunrise, receipts"
```

## Transformations

Style transfer uses high-level traits and explicitly prohibits copying recognizable
lyrics or signature phrases:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py `
  --mode style_transfer --source-file data\scratch\my_verse.txt `
  --artist-reference "a named artist" `
  --style "dense internal rhyme, restrained delivery, cinematic detail"
```

Mutation and improvement use the same source-file contract:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py `
  --mode mutate --source-file data\scratch\my_verse.txt

.\.venv\Scripts\python.exe scripts\run_quality_max_lane.py `
  --mode improve --source-file data\scratch\my_verse.txt
```

## Research sequence

1. Compare existing Qwen3-14B adapter versus `--base-only` on the same prompts.
2. Build a provenance-safe multi-task dataset using the mix in
   `configs/quality_max/research_program_v1.json`.
3. Train a smoke QLoRA adapter before any full run.
4. Evaluate blind human preference, length responsiveness, artifact rate, repetition,
   coherence, rhyme, imagery, and transformation faithfulness.
5. Promote only after human preference and per-family floors pass. Heuristic ranking
   alone cannot promote a model.

The multi-candidate and revision settings spend more generation time than the existing
single-pass baseline. That trade is intentional because this lane optimizes quality,
not strict speed or recipe equivalence.
