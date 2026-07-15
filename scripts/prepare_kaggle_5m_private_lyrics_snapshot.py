from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rap_song_data.scratch.common import hash_file, hash_text, utc_now
from rap_song_data.scratch.corpus import content_flags, normalized_lyrics_key


SOURCE_ID = "kaggle_private_lyrics_5m_genius_scrape"
SOURCE_URL = "https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset"
DEFAULT_SNAPSHOT_ID = "20260715_credentialed_download_rap250k_pilot"
csv.field_size_limit(128 * 1024 * 1024)


def normalize_lyric(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [" ".join(line.split()) for line in text.split("\n")]
    compact = "\n".join(line for line in lines if line)
    while "\n\n\n" in compact:
        compact = compact.replace("\n\n\n", "\n\n")
    return compact.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def parse_tags(values: Iterable[str]) -> set[str]:
    tags: set[str] = set()
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip().lower()
            if item:
                tags.add(item)
    return tags


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def open_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="\n")


def write_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def path_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


class ExactDedupe:
    def __init__(self, sqlite_path: Path | None) -> None:
        self.sqlite_path = sqlite_path
        self.connection: sqlite3.Connection | None = None
        if sqlite_path and sqlite_path.exists():
            self.connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def existing_record_id(self, normalized_key: str) -> str | None:
        if self.connection is None:
            return None
        exact_hash = hash_text(normalized_key)
        row = self.connection.execute("SELECT record_id FROM exact_hashes WHERE hash = ?", (exact_hash,)).fetchone()
        return str(row[0]) if row else None


def iter_records(
    csv_path: Path,
    *,
    include_tags: set[str],
    min_words: int,
    max_approved_records: int | None,
    dedupe: ExactDedupe,
) -> tuple[Counter[str], Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen_in_snapshot: set[str] = set()
    retrieved_at = utc_now()

    normalized_dir = iter_records.normalized_dir
    review_dir = iter_records.review_dir
    records_path = normalized_dir / "records.jsonl"
    rejected_path = normalized_dir / "rejected.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    item_reviews_path = review_dir / "item_reviews.jsonl"

    with (
        csv_path.open("r", encoding="utf-8", errors="replace", newline="") as source,
        open_jsonl(records_path) as records_handle,
        open_jsonl(rejected_path) as rejected_handle,
        open_jsonl(approved_records_path) as approved_handle,
        open_jsonl(item_reviews_path) as reviews_handle,
    ):
        reader = csv.DictReader(source)
        for row_index, row in enumerate(reader):
            counts["scanned_rows"] += 1
            tag = str(row.get("tag") or "").strip().lower()
            tag_counts[tag or "unknown"] += 1
            if include_tags and tag not in include_tags:
                counts["filtered_tag"] += 1
                continue

            lyric = normalize_lyric(str(row.get("lyrics") or ""))
            if not lyric:
                counts["rejected_empty_lyrics"] += 1
                write_row(rejected_handle, {"row_index": row_index, "reason": "empty_lyrics", "tag": tag})
                continue

            normalized_key = normalized_lyrics_key(lyric)
            word_count = len(normalized_key.split())
            if word_count < min_words:
                counts["rejected_too_short"] += 1
                write_row(
                    rejected_handle,
                    {
                        "row_index": row_index,
                        "reason": "too_short",
                        "tag": tag,
                        "word_count": word_count,
                    },
                )
                continue

            normalized_sha256 = sha256_text(normalized_key)
            if normalized_sha256 in seen_in_snapshot:
                counts["rejected_duplicate_within_snapshot"] += 1
                write_row(
                    rejected_handle,
                    {
                        "row_index": row_index,
                        "reason": "duplicate_within_snapshot",
                        "tag": tag,
                        "normalized_text_sha256": normalized_sha256,
                    },
                )
                continue

            existing_id = dedupe.existing_record_id(normalized_key)
            if existing_id is not None:
                counts["rejected_duplicate_existing_exact"] += 1
                write_row(
                    rejected_handle,
                    {
                        "row_index": row_index,
                        "reason": "duplicate_existing_exact",
                        "tag": tag,
                        "existing_record_id": existing_id,
                        "normalized_text_sha256": normalized_sha256,
                    },
                )
                continue

            seen_in_snapshot.add(normalized_sha256)
            source_item_id = str(row.get("id") or row_index)
            title = str(row.get("title") or "").strip() or f"Kaggle 5M lyric {source_item_id}"
            artist = str(row.get("artist") or "").strip() or None
            record_id = hash_text(f"{SOURCE_ID}:{source_item_id}:{normalized_sha256}", digest_size=16)
            record = {
                "record_id": record_id,
                "source_id": SOURCE_ID,
                "source_item_id": source_item_id,
                "retrieved_at": retrieved_at,
                "source_url": SOURCE_URL,
                "source_url_hash": hash_text(SOURCE_URL, digest_size=16),
                "rights_tier": "private_unknown_rights",
                "rights_status": "private_use_only_underlying_rights_unverified",
                "license_id": "kaggle_package_unknown_underlying_lyrics_unverified",
                "license_evidence": SOURCE_URL,
                "license_evidence_scope": "dataset_page_plus_local_snapshot",
                "title": title,
                "authors": [artist] if artist else None,
                "artist_clean": artist,
                "genre": tag,
                "language": None,
                "issued": str(row.get("year") or "").strip() or None,
                "views": str(row.get("views") or "").strip() or None,
                "features": str(row.get("features") or "").strip() or None,
                "kaggle_id": source_item_id,
                "text_sha256": sha256_text(lyric),
                "normalized_text_sha256": normalized_sha256,
                "dedupe_cluster_id": None,
                "removal_key": f"{SOURCE_ID}:{source_item_id}",
                "split_group_id": artist,
                "split_group_status": "pending_deduplication",
                "split": None,
                "word_count": word_count,
                "line_count": len([line for line in lyric.splitlines() if line.strip()]),
                "approx_tokens": int(word_count * 1.3),
                "content_flags": content_flags(lyric),
                "lyrics": lyric,
            }
            review = {
                "source_id": SOURCE_ID,
                "source_item_id": source_item_id,
                "record_id": record_id,
                "rights_decision": "private_use_approved",
                "rights_tier": "private_unknown_rights",
                "rights_status": record["rights_status"],
                "license_id": record["license_id"],
                "license_evidence": SOURCE_URL,
                "quality_decision": "accepted",
                "issues": ["unknown_kaggle_license", "underlying_lyric_rights_unverified"],
                "approx_tokens": record["approx_tokens"],
            }
            write_row(records_handle, record)
            write_row(approved_handle, record)
            write_row(reviews_handle, review)
            counts["approved_records"] += 1
            counts["approved_approx_tokens"] += int(record["approx_tokens"])
            if len(examples) < 5:
                examples.append(
                    {
                        "source_item_id": source_item_id,
                        "title": title,
                        "artist": artist,
                        "tag": tag,
                        "word_count": word_count,
                        "lyrics_preview": lyric[:240],
                    }
                )
            if max_approved_records is not None and counts["approved_records"] >= max_approved_records:
                counts["stopped_at_approved_cap"] = 1
                break

    return counts, tag_counts, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-normalize the Kaggle 5M private lyric dataset.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path(
            "data/corpus_lake/raw/kaggle_private_lyrics/"
            "nikhilnayak123__5-million-song-lyrics-dataset/"
            "20260715_credentialed_download/ds2.csv"
        ),
    )
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--include-tags", nargs="*", default=["rap"])
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-approved-records", type=int, default=250_000)
    parser.add_argument("--dedupe-sqlite", type=Path, default=Path("data/scratch/v1/dedupe.sqlite3"))
    parser.add_argument("--normalized-root", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--catalog-root", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    args = parser.parse_args()

    started = time.monotonic()
    source_dir = args.catalog_root / SOURCE_ID
    review_dir = source_dir / "reviews" / args.snapshot_id
    normalized_dir = args.normalized_root / SOURCE_ID / args.snapshot_id
    iter_records.normalized_dir = normalized_dir
    iter_records.review_dir = review_dir

    dedupe = ExactDedupe(args.dedupe_sqlite)
    try:
        counts, tag_counts, examples = iter_records(
            args.raw_csv,
            include_tags=parse_tags(args.include_tags),
            min_words=args.min_words,
            max_approved_records=args.max_approved_records,
            dedupe=dedupe,
        )
    finally:
        dedupe.close()

    records_path = normalized_dir / "records.jsonl"
    rejected_path = normalized_dir / "rejected.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    item_reviews_path = review_dir / "item_reviews.jsonl"
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "prepare-kaggle-5m-private-lyrics-snapshot",
        "source_id": SOURCE_ID,
        "snapshot_id": args.snapshot_id,
        "training_eligibility": "private_only",
        "admission_allowed": True,
        "admission_scope": "approved_records_only",
        "profile_eligibility": ["scratch-private-extended-v1"],
        "rights_note": "Personal/private-use only. Kaggle package license is unknown; underlying lyric redistribution rights are unverified.",
        "filters": {
            "include_tags": sorted(parse_tags(args.include_tags)),
            "min_words": args.min_words,
            "max_approved_records": args.max_approved_records,
            "dedupe_sqlite": str(args.dedupe_sqlite) if args.dedupe_sqlite else None,
        },
        "counts": dict(counts),
        "tag_counts_seen": dict(tag_counts.most_common()),
        "sample_records_redacted": examples,
        "inputs": {
            "raw_csv": {
                "path": str(args.raw_csv),
                "bytes": args.raw_csv.stat().st_size,
            }
        },
        "outputs": {
            "records": path_manifest(records_path),
            "rejected": path_manifest(rejected_path),
            "item_reviews": path_manifest(item_reviews_path),
            "approved_records": path_manifest(approved_records_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(review_dir / "review_manifest.json", manifest)
    write_json(source_dir / "review_manifest.json", manifest)
    write_json(normalized_dir / "normalization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
