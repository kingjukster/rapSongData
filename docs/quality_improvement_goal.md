# Rap generation quality goal

## Objective

Produce a Qwen3-4B adapter that improves original exact-12-line rap verses over
the base model on held-out prompts, with measurable gains in imagery, scene
coherence, rhyme, originality, and final-line payoff.

## Required gates

1. The refactored source, configs, prompts, and tests reproduce from a clean
   checkout.
2. Training data has no parse failures, normalized duplicate preference sides,
   requested/actual bar-count mismatches, or song/hash leakage across splits.
3. The primary SFT target is exactly 12 lyric lines and matches evaluation prompt
   structure; bar-level continuation data is not silently mixed into this task.
4. The calibrated set contains 100-300 provenance-verified automated-consensus
   examples. Automated labels are explicitly identified and are never represented
   as human judgments.
5. Qwen3-4B runs use the same dataset and seed for 1-, 2-, and 3-epoch variants,
   centered on a `5e-5` learning rate.
6. Base and adapters use the same held-out prompts, seeds, and decoding settings.

## Selection criteria

A candidate wins only when it:

- beats the base model in blind human or calibrated judge preference;
- preserves or improves raw exact-12-line compliance;
- reduces weak imagery, genericness, low rhyme, weak payoff, and scene drift;
- introduces no new safety, provenance, memorization, or split-leakage regression.

Training loss alone is not a selection criterion.

## Current phase

Human review is deprecated. The active path uses a provenance-aware automated
consensus dataset, pinned 1/2/3-epoch training matrix, raw development and
confirmation evaluation, a blinded multi-vote OpenAI judge, and fail-closed
promotion gates.

## Active handoff

1. Build the automated dataset with `python
   scripts/build_auto_calibrated_12line_sft.py`.
2. Run `python scripts/run_quality_goal_training_matrix.py`. It refuses to
   start unless provenance, automated-calibration, split-leakage, line-count, truncation,
   and held-out overlap gates pass.
3. Run `python scripts/run_quality_goal_evaluation_matrix.py --stage
   development`. The blinded three-vote automated judge selects the development
   adapter without exposing model identity.
4. Run `python scripts/run_quality_goal_evaluation_matrix.py --stage
   confirmation`. It consumes the automated development winner and applies the
   locked base-versus-adapter promotion gate.

No adapter is promoted unless every locked promotion gate passes.
