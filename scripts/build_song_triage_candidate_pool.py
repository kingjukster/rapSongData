#!/usr/bin/env python3
"""Build a strict, family-balanced candidate pool from nano song triage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


HARD_EXCLUDE = {"scrape_artifact", "non_lyric", "too_messy", "artist_or_metadata_leak"}
QUALITY_EXCLUDE = {"weak_craft", "generic", "repetitive", "incomplete"}
SUPPORT_TAGS = {"good_cadence", "coherent_story", "vivid_imagery", "strong_candidate"}


def family_memberships(row: dict[str, Any]) -> set[str]:
    """Return strict candidate families for a triage row."""
    tags = set(row.get("tags") or [])
    if row.get("decision") != "keep" or tags & (HARD_EXCLUDE | QUALITY_EXCLUDE):
        return set()

    families: set[str] = set()
    if {"technical_rhyme", "good_cadence"} <= tags:
        families.add("technical")
    if (
        "clean_candidate" in tags
        and "unsafe_clean_training" not in tags
        and bool(tags & SUPPORT_TAGS)
    ):
        families.add("clean")
    if "coherent_story" in tags and bool(tags & {"vivid_imagery", "good_cadence"}):
        families.add("story_support")
    return families


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Sort strongest evidence first, with stable source-rank tie breaking."""
    tags = set(row.get("tags") or [])
    positives = len(tags & {
        "technical_rhyme", "good_cadence", "coherent_story", "vivid_imagery",
        "strong_candidate", "clean_candidate",
    })
    return (
        -float(row.get("quality_floor") or 0),
        -float(row.get("confidence") or 0),
        -float(positives),
        float(row.get("rank") or 10**12),
        str(row.get("song_key") or ""),
    )


def load_and_select(
    path: Path,
    max_candidates: int,
    technical_cap: int,
    clean_cap: int,
    available_song_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = {
        "technical": [], "clean": [], "story_support": [],
    }
    counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            counts["input_rows"] += 1
            row = json.loads(line)
            key = str(row.get("song_key") or "")
            if not key:
                counts["missing_song_key"] += 1
                continue
            if available_song_keys is not None and key not in available_song_keys:
                counts["missing_from_review_pool"] += 1
                continue
            tags = set(row.get("tags") or [])
            if row.get("decision") != "keep":
                counts[f"decision_{row.get('decision', 'missing')}"] += 1
                continue
            counts["keep_rows"] += 1
            if tags & HARD_EXCLUDE:
                counts["hard_excluded"] += 1
                continue
            if tags & QUALITY_EXCLUDE:
                counts["quality_excluded"] += 1
                continue
            families = family_memberships(row)
            if not families:
                counts["no_target_family"] += 1
                continue
            normalized = dict(row)
            normalized["song_key"] = key
            normalized["candidate_families"] = sorted(families)
            normalized["source_line"] = line_number
            for family in families:
                eligible[family].append(normalized)

    for rows in eligible.values():
        rows.sort(key=rank_key)

    selected: dict[str, dict[str, Any]] = {}
    selection_order: list[str] = []

    def add(rows: list[dict[str, Any]], cap: int) -> None:
        for row in rows[:cap]:
            key = row["song_key"]
            if key not in selected and len(selected) < max_candidates:
                selected[key] = row
                selection_order.append(key)

    # Protect the two weak families first, then fill with story/melodic support.
    add(eligible["technical"], technical_cap)
    add(eligible["clean"], clean_cap)
    add(eligible["story_support"], len(eligible["story_support"]))

    result = [selected[key] for key in selection_order]
    stats = {
        "scan_counts": dict(sorted(counts.items())),
        "eligible_by_family": {name: len(rows) for name, rows in eligible.items()},
        "selected_unique": len(result),
        "selected_memberships": dict(sorted(Counter(
            family for row in result for family in row["candidate_families"]
        ).items())),
    }
    return result, stats


def materialize_lyrics(
    rows: list[dict[str, Any]], source: Path, output: Path, batch_size: int
) -> tuple[int, list[str]]:
    by_key = {row["song_key"]: row for row in rows}
    found: set[str] = set()
    materialized: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for source_row in batch.to_pylist():
            key = str(source_row.get("song_key") or "")
            review = by_key.get(key)
            if review is None:
                continue
            found.add(key)
            merged = dict(source_row)
            merged.update({
                "triage_decision": review.get("decision"),
                "triage_confidence": review.get("confidence"),
                "triage_quality_floor": review.get("quality_floor"),
                "triage_tags": review.get("tags") or [],
                "triage_reason": review.get("reason"),
                "candidate_families": review["candidate_families"],
            })
            materialized.append(merged)

    # Restore deterministic selection order rather than parquet source order.
    order = {row["song_key"]: index for index, row in enumerate(rows)}
    materialized.sort(key=lambda row: order[str(row["song_key"])])
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(materialized), output, compression="zstd")
    return len(materialized), sorted(set(by_key) - found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", type=Path, default=Path("data/reviews/song_triage_nano_completed_250k_deduped.jsonl"))
    parser.add_argument("--review-pool", type=Path, default=Path("data/review_pool/rap_song_review_pool_top250k.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/reviews/song_triage_candidate_pool_v1.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/reviews/song_triage_candidate_pool_v1_manifest.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/reviews/song_triage_candidate_pool_v1_summary.json"))
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--technical-cap", type=int, default=25000)
    parser.add_argument("--clean-cap", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    started = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    command = " ".join([sys.executable, *sys.argv])
    available_song_keys = {
        str(value)
        for value in pq.read_table(args.review_pool, columns=["song_key"])["song_key"].to_pylist()
        if value is not None
    }
    selected, stats = load_and_select(
        args.triage,
        args.max_candidates,
        args.technical_cap,
        args.clean_cap,
        available_song_keys,
    )
    if not selected:
        raise RuntimeError("No candidates passed the strict family filters")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for selection_rank, row in enumerate(selected, 1):
            record = dict(row)
            record["candidate_rank"] = selection_rank
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    output_rows, missing = materialize_lyrics(
        selected, args.review_pool, args.output, args.batch_size
    )
    ended = time.time()
    summary = {
        "schema_version": 1,
        "started_at_utc": started_iso,
        "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        "wall_time_seconds": round(ended - started, 3),
        "command": command,
        "inputs": {"triage": str(args.triage), "review_pool": str(args.review_pool)},
        "outputs": {"parquet": str(args.output), "manifest": str(args.manifest)},
        "config": {
            "max_candidates": args.max_candidates,
            "technical_cap": args.technical_cap,
            "clean_cap": args.clean_cap,
            "hard_exclude_tags": sorted(HARD_EXCLUDE),
            "quality_exclude_tags": sorted(QUALITY_EXCLUDE),
        },
        **stats,
        "materialized_rows": output_rows,
        "missing_from_review_pool_count": len(missing),
        "missing_from_review_pool_sample": missing[:50],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if missing or output_rows != len(selected):
        raise RuntimeError(f"Materialization incomplete: {output_rows}/{len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
