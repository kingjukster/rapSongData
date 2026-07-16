"""Download a queue of Kaggle lyric datasets with raw manifests.

This is intentionally a raw-acquisition helper: it preserves downloaded files,
unzips Kaggle bundles into a snapshot directory, and writes hashes/provenance so
later normalizers/dedupe runs can decide what to admit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_RAW_ROOT = Path("data/corpus_lake/raw/kaggle_private_lyrics")
DEFAULT_OVERFLOW_ROOT = Path("C:/Users/kingj/rapSongData_overflow_review/kaggle_private_lyrics")
DEFAULT_SNAPSHOT_ID = "20260716_aggressive_queue"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify_ref(ref: str) -> str:
    return ref.strip().replace("/", "__").replace(" ", "_")


def drive_free_gb(path: Path) -> float:
    anchor = Path(path.anchor or ".").resolve()
    usage = shutil.disk_usage(anchor)
    return usage.free / (1024**3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in {"download_manifest.json", "download_command.json"}:
            continue
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def load_queue(args: argparse.Namespace) -> list[str]:
    refs: list[str] = []
    if args.queue_json:
        payload = json.loads(args.queue_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("refs") or payload.get("kaggle_refs") or payload.get("datasets") or []
        else:
            values = payload
        for value in values:
            if isinstance(value, str):
                refs.append(value)
            elif isinstance(value, dict) and value.get("kaggle_ref"):
                refs.append(str(value["kaggle_ref"]))
            elif isinstance(value, dict) and value.get("ref"):
                refs.append(str(value["ref"]))
    refs.extend(args.ref or [])
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        ref = ref.strip()
        if ref and ref not in seen:
            deduped.append(ref)
            seen.add(ref)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", action="append", default=[], help="Kaggle dataset ref, repeatable.")
    parser.add_argument("--queue-json", type=Path, help="Optional JSON list/dict of Kaggle refs.")
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--overflow-root", type=Path, default=DEFAULT_OVERFLOW_ROOT)
    parser.add_argument("--min-d-free-gb", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refs = load_queue(args)
    if not refs:
        raise SystemExit("No Kaggle refs provided.")

    api = KaggleApi()
    api.authenticate()
    started = time.time()
    results: list[dict[str, Any]] = []
    print(json.dumps({"event": "queue_started", "refs": refs, "snapshot_id": args.snapshot_id}), flush=True)
    for ref in refs:
        started_ref = time.time()
        root = args.raw_root
        d_free = drive_free_gb(Path("D:/"))
        storage = "D:"
        if d_free < args.min_d_free_gb:
            root = args.overflow_root
            storage = "C_OVERFLOW"
        out_dir = root / slugify_ref(ref) / args.snapshot_id
        manifest_path = out_dir / "download_manifest.json"
        if manifest_path.exists() and not args.force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            results.append({"ref": ref, "status": "skipped_existing", "manifest": str(manifest_path)})
            print(json.dumps({"event": "dataset_skipped_existing", "ref": ref, "manifest": str(manifest_path)}), flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        command = {
            "generated_at_utc": utc_now(),
            "ref": ref,
            "snapshot_id": args.snapshot_id,
            "storage": storage,
            "d_free_gb_at_start": round(d_free, 3),
            "raw_root": str(root),
        }
        (out_dir / "download_command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        status = "completed"
        error = None
        try:
            print(json.dumps({"event": "dataset_download_started", "ref": ref, "out_dir": str(out_dir), "storage": storage}), flush=True)
            api.dataset_download_files(ref, path=str(out_dir), unzip=True, quiet=False)
        except Exception as exc:  # pragma: no cover - operational path
            status = "failed"
            error = repr(exc)
            print(json.dumps({"event": "dataset_download_failed", "ref": ref, "error": error}), flush=True)
        files = file_manifest(out_dir)
        manifest = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "ref": ref,
            "snapshot_id": args.snapshot_id,
            "url": f"https://www.kaggle.com/datasets/{ref}",
            "rights_partition": "private_unknown_rights",
            "rights_note": "Private/personal-use only. Kaggle package license does not prove underlying lyric rights.",
            "status": status,
            "storage": storage,
            "output_dir": str(out_dir),
            "file_count": len(files),
            "downloaded_bytes": sum(int(row["bytes"]) for row in files),
            "files": files,
            "error": error,
            "wall_seconds": round(time.time() - started_ref, 3),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append({"ref": ref, "status": status, "manifest": str(manifest_path), "bytes": manifest["downloaded_bytes"]})
        print(
            json.dumps(
                {
                    "event": "dataset_completed",
                    "ref": ref,
                    "status": status,
                    "files": len(files),
                    "bytes": manifest["downloaded_bytes"],
                    "wall_seconds": manifest["wall_seconds"],
                }
            ),
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_id": args.snapshot_id,
        "results": results,
        "wall_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
