from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


SOURCE_ID = "kaggle_private_lyrics_english_5genres_500k"
SOURCE_URL = "https://www.kaggle.com/datasets/d3stron/english-music-lyrics-5-genres-500k"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hash_text(value: str, *, digest_size: int = 32) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:digest_size] if digest_size < 64 else digest


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_lyric(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_csv_records(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    file_manifests: list[dict[str, Any]] = []
    retrieved_at = utc_now()
    for csv_path in sorted(raw_dir.glob("*.csv")):
        file_manifests.append(path_manifest(csv_path))
        split_hint = "test" if "test" in csv_path.stem.lower() else None
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                lyric = normalize_lyric(row.get("Lyric", ""))
                genre = str(row.get("genre") or "").strip()
                if not lyric:
                    rejected.append(
                        {
                            "file": csv_path.name,
                            "row_index": index,
                            "reason": "empty_lyric",
                        }
                    )
                    continue
                source_item_id = f"{csv_path.name}:{index}"
                title = f"{genre or 'Unknown'} lyric {index}"
                record_id = hash_text(f"{SOURCE_ID}:{source_item_id}:{hash_text(lyric, digest_size=16)}", digest_size=16)
                rows.append(
                    {
                        "record_id": record_id,
                        "source_id": SOURCE_ID,
                        "source_item_id": source_item_id,
                        "retrieved_at": retrieved_at,
                        "source_url": SOURCE_URL,
                        "source_url_hash": hash_text(SOURCE_URL, digest_size=16),
                        "rights_tier": "private_unknown_rights",
                        "rights_status": "private_use_only_underlying_rights_unverified",
                        "license_id": "kaggle_package_mit_underlying_lyrics_unverified",
                        "license_evidence": SOURCE_URL,
                        "license_evidence_scope": "dataset_page_plus_local_snapshot",
                        "title": title,
                        "authors": None,
                        "artist_clean": None,
                        "genre": genre,
                        "language": "en",
                        "issued": None,
                        "text_sha256": hash_text(lyric, digest_size=32),
                        "normalized_text_sha256": hash_text(lyric.lower(), digest_size=32),
                        "dedupe_cluster_id": None,
                        "removal_key": f"{SOURCE_ID}:{source_item_id}",
                        "split_group_id": None,
                        "split_group_status": "pending_deduplication",
                        "split": split_hint,
                        "word_count": len(WORD_RE.findall(lyric)),
                        "approx_tokens": int(len(WORD_RE.findall(lyric)) * 1.3),
                        "lyrics": lyric,
                    }
                )
    return rows, rejected, file_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the Kaggle private lyrics 5-genre snapshot.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("data/corpus_lake/normalized"),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("data/scratch/catalog/v2/sources"),
    )
    parser.add_argument("--snapshot-id", default="20260715_credentialed_download")
    args = parser.parse_args()

    started = time.monotonic()
    records, rejected, file_manifests = iter_csv_records(args.raw_dir)
    normalized_dir = args.normalized_root / SOURCE_ID / args.snapshot_id
    review_dir = args.catalog_root / SOURCE_ID / "reviews" / args.snapshot_id
    records_path = normalized_dir / "records.jsonl"
    rejected_path = normalized_dir / "rejected.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    item_reviews_path = review_dir / "item_reviews.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(rejected_path, rejected)
    write_jsonl(approved_records_path, records)
    reviews = [
        {
            "source_id": SOURCE_ID,
            "source_item_id": row["source_item_id"],
            "record_id": row["record_id"],
            "rights_decision": "private_use_approved",
            "rights_tier": "private_unknown_rights",
            "rights_status": "private_use_only_underlying_rights_unverified",
            "license_id": row["license_id"],
            "license_evidence": row["license_evidence"],
            "quality_decision": "accepted",
            "issues": ["underlying_lyric_rights_unverified"],
            "approx_tokens": row["approx_tokens"],
        }
        for row in records
    ]
    write_jsonl(item_reviews_path, reviews)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "operation": "prepare-kaggle-private-lyrics-snapshot",
        "source_id": SOURCE_ID,
        "snapshot_id": args.snapshot_id,
        "training_eligibility": "private_only",
        "admission_allowed": True,
        "admission_scope": "private_snapshot_records",
        "profile_eligibility": ["scratch-private-extended-v1"],
        "rights_note": "Personal/private-use only. Kaggle package license observed as MIT; underlying lyric redistribution rights are unverified.",
        "counts": {
            "records": len(records),
            "rejected_records": len(rejected),
            "approx_tokens": sum(int(row["approx_tokens"]) for row in records),
        },
        "inputs": {
            "raw_dir": str(args.raw_dir),
            "files": file_manifests,
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
    write_json(args.catalog_root / SOURCE_ID / "review_manifest.json", manifest)
    write_json(normalized_dir / "normalization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
