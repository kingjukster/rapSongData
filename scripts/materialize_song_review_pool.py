#!/usr/bin/env python3
"""Materialize a ranked song-review pool with lyrics from a metadata manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def read_manifest(path: Path, limit: int) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(selected) >= limit:
                break
            row = json.loads(line)
            key = str(row["song_key"])
            if key in selected:
                raise RuntimeError(f"Duplicate song_key in manifest: {key}")
            selected[key] = row
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/full_song_openai_ranked_manifest_top250k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/review_pool/rap_song_review_pool_top250k.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("data/review_pool/rap_song_review_pool_top250k_summary.json"))
    parser.add_argument("--limit", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    started_at = time.time()
    command = " ".join([sys.executable, *sys.argv])
    manifest = read_manifest(args.manifest, args.limit)
    found: set[str] = set()
    hash_mismatches: list[str] = []
    writer: pq.ParquetWriter | None = None
    output_rows = 0
    source_rows_scanned = 0
    columns = ["title", "tag", "artist", "year", "views", "id", "language_cld3", "language_ft", "language", "lyrics"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        parquet = pq.ParquetFile(args.input)
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
            source_rows_scanned += batch.num_rows
            rows = []
            for row in batch.to_pylist():
                key = str(row.get("id") or "")
                meta = manifest.get(key)
                if meta is None:
                    continue
                lyrics = str(row.get("lyrics") or "")
                current_hash = normalized_hash(lyrics)
                if current_hash != meta["text_hash"]:
                    hash_mismatches.append(key)
                    continue
                found.add(key)
                rows.append({
                    "song_key": key,
                    "source_id": row.get("id"),
                    "manifest_rank": int(meta["manifest_rank"]),
                    "rank_score": float(meta["rank_score"]),
                    "title": row.get("title"),
                    "artist": row.get("artist"),
                    "year": row.get("year"),
                    "views": row.get("views"),
                    "tag": row.get("tag"),
                    "language": row.get("language") or row.get("language_ft") or row.get("language_cld3"),
                    "word_count": int(meta["word_count"]),
                    "line_count": int(meta["line_count"]),
                    "header_count": int(meta["header_count"]),
                    "avg_line_words": float(meta["avg_line_words"]),
                    "text_hash": current_hash,
                    "lyrics": lyrics,
                })
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
            output_rows += len(rows)
    finally:
        if writer is not None:
            writer.close()

    missing = sorted(set(manifest) - found - set(hash_mismatches))
    ended_at = time.time()
    summary = {
        "schema_version": 1,
        "command": command,
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "wall_seconds": round(ended_at - started_at, 3),
        "input": str(args.input),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "requested_rows": len(manifest),
        "source_rows_scanned": source_rows_scanned,
        "output_rows": output_rows,
        "missing_rows": len(missing),
        "hash_mismatch_rows": len(hash_mismatches),
        "missing_examples": missing[:20],
        "hash_mismatch_examples": hash_mismatches[:20],
        "output_bytes": args.output.stat().st_size if args.output.exists() else 0,
        "criteria": {
            "source_tag": "rap",
            "language": "en",
            "minimum_words": 80,
            "minimum_lines": 8,
            "obvious_scrape_junk_rejected": True,
            "exact_text_deduplication": True,
            "ranked_by": "views, song length, line shape, section headers, metadata completeness",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if output_rows == len(manifest) and not missing and not hash_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
