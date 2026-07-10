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
4. The calibrated set contains 100–300 genuinely human-reviewed examples, or the
   run remains blocked at the human-review gate without relabeling auto-judged
   examples as manual.
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

Reproducibility freeze and dataset-gate audit.
