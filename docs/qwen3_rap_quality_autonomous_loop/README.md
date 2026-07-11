# qwen3_rap_quality_autonomous_loop

This folder is the governance boundary for the autonomous quality-improvement loop.

## Canonical Sources (always referenced)
- Documentation milestone: C:\Users\kingj\projects\rapSongData\codex-bible\rapSongData Documentation and Governance Milestone Plan.pdf
- Comprehensive goal: C:\Users\kingj\projects\rapSongData\codex-bible\Comprehensive Codex Goal and Research Program for rapSongData.pdf
- Short goal summary: C:\Users\kingj\projects\rapSongData\.codex\attachments\573fb76a-8b55-488f-9bef-b4daae93c1c6\pasted-text.txt

## Operating Rules
- Never print secrets or dump `.env` values.
- Preserve unknown-license data as experimental-only.
- Keep all meaningful experiments in `state.json` and this folder.
- Every experiment must emit: run_manifest.json, config_snapshot/, metrics.json, summary.md, decision.md,
  stdout.log, stderr.log, sample_outputs.md.
- All loop artifacts are mirrored under `C:\Users\kingj\projects\rapSongData\codex-autonomous-loop-evidence` for repository-root evidence review.

## Current Objective
- Keep the incumbent baseline intact.
- Improve 12-line rap quality under quality gates.
- Stop only on explicit hard safety/legal blocker, time exhaustion, queue exhaustion, or promotion decision.
