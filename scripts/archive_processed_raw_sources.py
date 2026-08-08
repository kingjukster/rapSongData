from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_CANDIDATES: list[tuple[str, str, str]] = [
    (
        "d_raw/huggingface_lyrics",
        r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw\huggingface_lyrics",
        "HF bulk and overnight raw snapshots; processed by private backlog exact dedupe.",
    ),
    (
        "d_raw/kaggle_private_lyrics",
        r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw\kaggle_private_lyrics",
        "Kaggle aggressive/overnight raw snapshots; processed by private backlog exact dedupe.",
    ),
    (
        "d_raw/common_crawl_lyrics",
        r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw\common_crawl_lyrics",
        "Common Crawl URL-range raw/admitted outputs; included as seeded admitted sources in v2 profile.",
    ),
    (
        "d_raw/common_crawl_wet_lyrics",
        r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw\common_crawl_wet_lyrics",
        "Common Crawl WET probe raw/admitted outputs; included as seeded admitted source in v2 profile.",
    ),
    (
        "d_raw/source_surveys",
        r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw\source_surveys",
        "Source-survey scratch outputs from acquisition planning.",
    ),
    (
        "c_overflow/huggingface_lyrics",
        r"C:\Users\kingj\rapSongData_overflow_review\huggingface_lyrics",
        "HF clean-songs overflow raw data; materialized into v2 profile.",
    ),
    (
        "c_overflow/lrclib_db_dumps",
        r"C:\Users\kingj\rapSongData_overflow_review\lrclib_db_dumps",
        "LRCLIB compressed dump; processed by lrclib exact dedupe.",
    ),
]


ALLOWED_DELETE_ROOTS = [
    Path(r"D:\Users\kingj\projects\rapSongData\data\corpus_lake\raw").resolve(),
    Path(r"C:\Users\kingj\rapSongData_overflow_review\huggingface_lyrics").resolve(),
    Path(r"C:\Users\kingj\rapSongData_overflow_review\lrclib_db_dumps").resolve(),
]


PROCESSED_EVIDENCE = [
    r"C:\Users\kingj\rapSongData_overflow_review\normalized\private_lyrics_backlog_exact_dedupe\20260716_backlog_after_tokenizer_v1_20260716_084703_generic\dedupe_summary.json",
    r"C:\Users\kingj\rapSongData_overflow_review\normalized\lrclib_exact_dedupe\20260716_backlog_after_tokenizer_v1_20260716_084703_lrclib\lrclib_normalization_summary.json",
    r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v2_1_outlier_filtered_20260717_0156\corpus_manifest.json",
    r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v2_1_outlier_filtered_20260717_0156\tokenization_manifest.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_inventory(root: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return count, total


def is_allowed_delete(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in ALLOWED_DELETE_ROOTS)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--delete-after-verify", action="store_true")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--per-source",
        action="store_true",
        help="Write one verified zip per source and optionally delete each source after its own verification.",
    )
    args = parser.parse_args()

    started = time.monotonic()
    run_dir = args.archive_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    archive_path = run_dir / f"{args.run_id}.zip"
    manifest_path = run_dir / "archive_manifest.json"
    status_path = run_dir / "archive_status.json"

    candidates: list[dict[str, Any]] = []
    for archive_prefix, raw_path, note in RAW_CANDIDATES:
        path = Path(raw_path)
        if not path.exists():
            candidates.append(
                {
                    "archive_prefix": archive_prefix,
                    "path": str(path),
                    "exists": False,
                    "note": note,
                }
            )
            continue
        files, bytes_total = file_inventory(path)
        candidates.append(
            {
                "archive_prefix": archive_prefix,
                "path": str(path),
                "exists": True,
                "file_count": files,
                "bytes": bytes_total,
                "gib": round(bytes_total / 1024**3, 6),
                "note": note,
                "delete_allowed": is_allowed_delete(path),
            }
        )

    evidence = []
    for evidence_path in PROCESSED_EVIDENCE:
        path = Path(evidence_path)
        evidence.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at": utc_now(),
        "archive_path": str(archive_path),
        "delete_after_verify_requested": args.delete_after_verify,
        "compression": {"method": "ZIP_DEFLATED", "level": args.compression_level},
        "processed_evidence": evidence,
        "candidates": candidates,
    }
    write_status(manifest_path, manifest)
    write_status(
        status_path,
        {
            "status": "archiving",
            "started_at": manifest["started_at"],
            "archive_path": str(archive_path),
            "processed_files": 0,
            "processed_bytes": 0,
        },
    )

    expected_files = sum(int(c.get("file_count") or 0) for c in candidates if c.get("exists"))
    expected_bytes = sum(int(c.get("bytes") or 0) for c in candidates if c.get("exists"))
    processed_files = 0
    processed_bytes = 0
    archives: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    verified = False
    bad_member = None
    archived_files = 0
    archive_bytes = 0

    def delete_source(candidate: dict[str, Any]) -> None:
        path = Path(str(candidate["path"]))
        if not bool(candidate.get("delete_allowed")) or not is_allowed_delete(path):
            deleted.append({"path": str(path), "deleted": False, "reason": "delete_not_allowed"})
            return
        if not path.exists():
            deleted.append({"path": str(path), "deleted": False, "reason": "already_missing"})
            return
        shutil.rmtree(path)
        deleted.append({"path": str(path), "deleted": True})

    def write_candidate_zip(candidate: dict[str, Any], destination: Path) -> dict[str, Any]:
        nonlocal processed_files, processed_bytes
        root = Path(str(candidate["path"]))
        prefix = str(candidate["archive_prefix"]).replace("\\", "/").strip("/")
        source_files = int(candidate.get("file_count") or 0)
        source_bytes = int(candidate.get("bytes") or 0)
        source_processed_files = 0
        source_processed_bytes = 0
        with zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=args.compression_level,
            allowZip64=True,
        ) as zf:
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(root).as_posix()
                arcname = f"{prefix}/{relative}"
                size = file_path.stat().st_size
                zf.write(file_path, arcname)
                processed_files += 1
                processed_bytes += size
                source_processed_files += 1
                source_processed_bytes += size
                if processed_files % 50 == 0:
                    write_status(
                        status_path,
                        {
                            "status": "archiving",
                            "started_at": manifest["started_at"],
                            "archive_path": str(archive_path),
                            "current_archive": str(destination),
                            "current_source": str(root),
                            "processed_files": processed_files,
                            "expected_files": expected_files,
                            "processed_bytes": processed_bytes,
                            "expected_bytes": expected_bytes,
                            "source_processed_files": source_processed_files,
                            "source_expected_files": source_files,
                            "source_processed_bytes": source_processed_bytes,
                            "source_expected_bytes": source_bytes,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                    )
        with zipfile.ZipFile(destination, mode="r") as zf:
            candidate_bad_member = zf.testzip()
            candidate_archived_files = len([info for info in zf.infolist() if not info.is_dir()])
        candidate_verified = candidate_bad_member is None and candidate_archived_files == source_files
        result = {
            "path": str(destination),
            "source_path": str(root),
            "archive_prefix": candidate["archive_prefix"],
            "archive_bytes": destination.stat().st_size,
            "source_bytes": source_bytes,
            "archived_files": candidate_archived_files,
            "expected_files": source_files,
            "verified": candidate_verified,
            "bad_member": candidate_bad_member,
        }
        archives.append(result)
        if args.delete_after_verify and candidate_verified:
            delete_source(candidate)
        return result

    if args.per_source:
        for candidate in candidates:
            if not candidate.get("exists"):
                continue
            safe_name = str(candidate["archive_prefix"]).replace("\\", "_").replace("/", "__").replace(":", "_")
            current_archive = run_dir / f"{args.run_id}__{safe_name}.zip"
            result = write_candidate_zip(candidate, current_archive)
            if not result["verified"]:
                break
        verified = bool(archives) and all(item["verified"] for item in archives) and processed_files == expected_files
        archive_bytes = sum(int(item["archive_bytes"]) for item in archives)
        archived_files = sum(int(item["archived_files"]) for item in archives)
        bad_member = next((item["bad_member"] for item in archives if item["bad_member"]), None)
    else:
        result = write_candidate_zip(
            {
                "archive_prefix": args.run_id,
                "path": ".",
                "file_count": 0,
                "bytes": 0,
            },
            archive_path,
        )
        # Preserve the original single-zip behavior by explicitly writing all candidates
        # into one archive. This branch is retained for backward compatibility, but the
        # outage-safe workflow should use --per-source.
        archives.append(result)
        verified = False
        raise RuntimeError("single-archive mode is disabled for this large raw-data workflow; use --per-source")

    completed = {
        **manifest,
        "ended_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "expected_files": expected_files,
        "archived_files": archived_files,
        "expected_bytes": expected_bytes,
        "archive_bytes": archive_bytes,
        "archive_gib": round(archive_bytes / 1024**3, 6),
        "archives": archives,
        "verification": {
            "verified": verified,
            "bad_member": bad_member,
            "file_count_matches": archived_files == expected_files,
        },
        "deleted_sources": deleted,
        "status": "complete" if verified else "verify_failed",
    }
    write_status(manifest_path, completed)
    write_status(status_path, completed)
    print(json.dumps(completed, indent=2, ensure_ascii=False))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
