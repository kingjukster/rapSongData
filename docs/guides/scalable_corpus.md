# Scalable scratch-corpus architecture

The corpus is organized around immutable source snapshots, rights partitions,
and deduplicated token counts. Song count is descriptive; token count is the
model-scaling control.

## Current scale

Scratch v1 contains 922,801 retained songs and 558,627,328 unique train tokens.
At a 20-token-per-parameter planning baseline, that nearly covers the current
30M-class model but covers only about 18.6% of a 150M target and 9.3% of a 300M target. Repeating
the existing shards can supply optimization steps, but does not add linguistic
or artist diversity.

Use a 30M -> 75M -> 150M -> 300M ladder. The 75M run is the cheapest useful
architecture-scaling checkpoint: it reveals whether extra capacity improves
held-out lyric quality before the data lake is large enough for 150M.

Run the versioned planner after each source snapshot:

```powershell
.\scripts\run_rap_scratch_safe.ps1 plan-corpus
```

It writes `data/scratch/catalog/v2/corpus_scale_plan.{json,md}` from the checked-in
source registry and the actual tokenizer manifest.

## Storage contract

```text
data/corpus_lake/
  raw/<source_id>/<snapshot_id>/             immutable downloads
  provenance/<source_id>/<snapshot_id>/      license, URLs, hashes, commands
  normalized/<source_id>/<snapshot_id>/      source-shaped Parquet
  dedup/<corpus_version>/                     exact and near-duplicate evidence
  partitions/<rights_partition>/<version>/   canonical song records
  mixtures/<mixture_id>/                     source weights and exclusions
  tokenized/<tokenizer_id>/<mixture_id>/      immutable binary shards
  catalogs/                                   source and corpus manifests
```

Large generated directories remain local. The checked-in registry defines what
may enter each partition; every local snapshot manifest records collection time,
source URL, license evidence, file SHA-256, row counts, rejection counts, and the
exact command.

## Canonical song record

Every normalized record should carry:

- `record_id`, `source_id`, `source_record_id`, and `snapshot_id`
- raw and normalized artist/title plus MusicBrainz IDs when matched
- language, year, genre/tags, section boundaries, and explicit-content flags
- `text_sha256`, normalized-text hash, and near-duplicate cluster ID
- rights tier, license identifier, provenance URL, collected timestamp, and
  release eligibility
- artist-family split key so an artist never crosses train/validation/test

No source-specific field should be smuggled into the training text. Keep source
metadata in Parquet and format training strings only while tokenizing.

## Source admission gates

1. Provenance and license evidence exist before text ingestion.
2. Raw files are immutable and hashed.
3. Normalization is source-specific but outputs the canonical schema.
4. Exact dedupe runs across all admitted partitions.
5. Near dedupe runs at song and stanza level.
6. Artist-family held-out splitting occurs after cross-source linking.
7. Token totals are measured with the target tokenizer.
8. Shards and checkpoints inherit the strictest source partition in their mix.

WASABI is useful metadata, but its public repository explicitly says full lyrics
cannot be redistributed. It must not be counted as 1.73M accessible training
lyrics. DALI is request-gated and small; admit it only after saving the granted
terms. The immediate engineering priority is MusicBrainz canonical metadata,
then a small clean Tier A ingestion pilot, then larger item-cleared collections.
