from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCES: list[dict[str, str]] = [
    {
        "id": "scratch_profile_v2_backlog_superseded",
        "path": r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v2_backlog_20260716_backlog_after_tokenizer_v1_20260716_084703",
        "reason": "Superseded by scratch-private-lyric-lake-v2_1_outlier_filtered_20260717_0156.",
    },
    {
        "id": "scratch_profile_v1_superseded",
        "path": r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v1_20260716_0742",
        "reason": "Superseded by v2/v2.1 profiles.",
    },
    {
        "id": "autonomous_loop_evidence_runs",
        "path": r"C:\Users\kingj\projects\rapSongData\codex-autonomous-loop-evidence\runs",
        "reason": "Large historical autonomous-loop evidence runs, not active model input.",
    },
    {
        "id": "qwen3_rap_quality_autonomous_loop_runs",
        "path": r"C:\Users\kingj\projects\rapSongData\runs\qwen3_rap_quality_autonomous_loop",
        "reason": "Large historical generation/evaluation run artifacts, not active scratch training input.",
    },
]


DO_NOT_TOUCH = [
    Path(r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles\scratch-private-lyric-lake-v2_1_outlier_filtered_20260717_0156").resolve(),
    Path(r"C:\Users\kingj\rapSongData_overflow_review\training_runs\scratch_100m_v2_1_resume_500m_20260717_1000\checkpoint-00030518").resolve(),
]


ALLOWED_DELETE_ROOTS = [
    Path(r"C:\Users\kingj\rapSongData_overflow_review\scratch_profiles").resolve(),
    Path(r"C:\Users\kingj\projects\rapSongData\codex-autonomous-loop-evidence").resolve(),
    Path(r"C:\Users\kingj\projects\rapSongData\runs").resolve(),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory(root: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return files, total


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_within(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def overlaps_protected(path: Path) -> bool:
    resolved = path.resolve()
    for protected in DO_NOT_TOUCH:
        if resolved == protected or resolved in protected.parents or protected in resolved.parents:
            return True
    return False


def archive_source(
    *,
    source: dict[str, str],
    output_dir: Path,
    status_path: Path,
    compression_level: int,
    delete_after_verify: bool,
    totals: dict[str, int],
) -> dict[str, Any]:
    source_id = source["id"]
    root = Path(source["path"])
    if not root.exists():
        return {"source_id": source_id, "path": str(root), "exists": False, "skipped": True}
    if overlaps_protected(root):
        raise RuntimeError(f"Refusing to archive protected active path: {root}")
    if not is_within(root, ALLOWED_DELETE_ROOTS):
        raise RuntimeError(f"Refusing source outside allowed roots: {root}")
    expected_files, expected_bytes = inventory(root)
    archive_path = output_dir / f"{source_id}.zip"
    source_processed_files = 0
    source_processed_bytes = 0
    prefix = source_id
    started = time.monotonic()
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as zf:
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            size = file_path.stat().st_size
            arcname = f"{prefix}/{file_path.relative_to(root).as_posix()}"
            zf.write(file_path, arcname)
            source_processed_files += 1
            source_processed_bytes += size
            totals["processed_files"] += 1
            totals["processed_bytes"] += size
            if source_processed_files % 250 == 0:
                write_json(
                    status_path,
                    {
                        "status": "archiving",
                        "current_source_id": source_id,
                        "current_source_path": str(root),
                        "current_archive": str(archive_path),
                        "source_processed_files": source_processed_files,
                        "source_expected_files": expected_files,
                        "source_processed_bytes": source_processed_bytes,
                        "source_expected_bytes": expected_bytes,
                        "total_processed_files": totals["processed_files"],
                        "total_expected_files": totals["expected_files"],
                        "total_processed_bytes": totals["processed_bytes"],
                        "total_expected_bytes": totals["expected_bytes"],
                        "updated_at": utc_now(),
                    },
                )
    with zipfile.ZipFile(archive_path, mode="r") as zf:
        bad_member = zf.testzip()
        archived_files = len([info for info in zf.infolist() if not info.is_dir()])
    verified = bad_member is None and archived_files == expected_files
    deleted = False
    if verified and delete_after_verify:
        shutil.rmtree(root)
        deleted = True
    return {
        "source_id": source_id,
        "path": str(root),
        "reason": source["reason"],
        "exists": True,
        "expected_files": expected_files,
        "archived_files": archived_files,
        "expected_bytes": expected_bytes,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_gib": round(archive_path.stat().st_size / 1024**3, 6),
        "verified": verified,
        "bad_member": bad_member,
        "deleted": deleted,
        "wall_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--delete-after-verify", action="store_true")
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args()

    started = time.monotonic()
    output_dir = args.archive_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "archive_status.json"
    manifest_path = output_dir / "archive_manifest.json"
    candidates: list[dict[str, Any]] = []
    expected_files = 0
    expected_bytes = 0
    for source in SOURCES:
        root = Path(source["path"])
        row: dict[str, Any] = {**source, "exists": root.exists()}
        if root.exists():
            files, size = inventory(root)
            row.update({"file_count": files, "bytes": size, "gib": round(size / 1024**3, 6)})
            expected_files += files
            expected_bytes += size
        candidates.append(row)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at": utc_now(),
        "archive_root": str(args.archive_root),
        "output_dir": str(output_dir),
        "delete_after_verify_requested": args.delete_after_verify,
        "compression": {"method": "ZIP_DEFLATED", "level": args.compression_level},
        "protected_paths": [str(path) for path in DO_NOT_TOUCH],
        "candidates": candidates,
    }
    write_json(manifest_path, manifest)
    totals = {
        "expected_files": expected_files,
        "expected_bytes": expected_bytes,
        "processed_files": 0,
        "processed_bytes": 0,
    }
    write_json(status_path, {"status": "started", **totals, "started_at": manifest["started_at"]})
    archives = []
    status = "complete"
    for source in SOURCES:
        result = archive_source(
            source=source,
            output_dir=output_dir,
            status_path=status_path,
            compression_level=args.compression_level,
            delete_after_verify=args.delete_after_verify,
            totals=totals,
        )
        archives.append(result)
        if result.get("exists") and not result.get("verified"):
            status = "verify_failed"
            break
    completed = {
        **manifest,
        "ended_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "expected_files": expected_files,
        "archived_files": sum(int(item.get("archived_files") or 0) for item in archives),
        "expected_bytes": expected_bytes,
        "archive_bytes": sum(int(item.get("archive_bytes") or 0) for item in archives),
        "archive_gib": round(sum(int(item.get("archive_bytes") or 0) for item in archives) / 1024**3, 6),
        "archives": archives,
        "verification": {
            "verified": status == "complete"
            and all((not item.get("exists")) or bool(item.get("verified")) for item in archives),
            "file_count_matches": expected_files == sum(int(item.get("archived_files") or 0) for item in archives),
        },
        "deleted_count": sum(1 for item in archives if item.get("deleted")),
        "status": status,
    }
    write_json(manifest_path, completed)
    write_json(status_path, completed)
    print(json.dumps(completed, indent=2, ensure_ascii=False))
    return 0 if completed["verification"]["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
