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

The adapter does not contain the base-model weights. The default configuration points to
the verified local snapshot at `D:/AI-Models/Qwen3-14B`, pinned to Hugging Face revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, and loads it offline. If the snapshot is
moved, update `model_path`; failed launches retain `run_summary.json`, `run_summary.md`,
and `generation.log` instead of silently leaving an ambiguous run.

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

The validated RTX 4090 smoke on 2026-08-01 loaded the Qwen3-14B adapter in 26.129
seconds and generated 482 effective tokens in 26.924 seconds (17.902 tokens/second
across a batch of two). Both candidates reached EOS instead of the 384-token cap;
the selected output contained 24 bars. PyTorch reported 9.744 GB peak allocated VRAM,
while device telemetry peaked at 11,907 MiB. Available system RAM briefly fell below
0.4 GB, so raw 27B-35B Transformers or QLoRA runs are not considered safe on the
current 32 GB host even when quantized weights appear to fit the GPU.

An already-local Gemma 4 26B-A4B Q4 GGUF also completed a one-candidate llama.cpp
smoke. It generated at 152 tokens/second and peaked at 15,829 MiB device VRAM, but
that single lyric sample was weaker than the Qwen3-14B winner. Treat larger MoE GGUF
models as fast inference-side teacher, critic, mutation, or revision candidates until
a same-prompt human evaluation demonstrates a quality win; they are not drop-in
replacements for this Transformers/PEFT training lane.

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
