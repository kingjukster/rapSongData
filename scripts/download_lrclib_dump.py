"""Download the latest LRCLIB SQLite database dump with provenance.

LRCLIB exposes both an API and public database dumps. For corpus acquisition,
the dump is the polite path: one large resumable download instead of millions
of API calls. This helper only preserves the raw compressed SQLite artifact and
writes a manifest; later normalization/dedupe should decide what to admit.
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

import requests


DUMP_INDEX_URL = "https://lrclib-db-dumps.bu3nnyut4y9jfkdg.workers.dev"
DUMP_BASE_URL = "https://db-dumps.lrclib.net"
DEFAULT_OUTPUT_ROOT = Path("C:/Users/kingj/rapSongData_overflow_review/lrclib_db_dumps")
DEFAULT_SNAPSHOT_ID = "20260716_lrclib_dump_v1"
DEFAULT_USER_AGENT = "rapSongData-lrclib-dump/0.1 (private research; local acquisition)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def drive_free_gb(path: Path) -> float:
    anchor = Path(path.anchor or ".").resolve()
    usage = shutil.disk_usage(anchor)
    return usage.free / (1024**3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dump_index(user_agent: str) -> list[dict[str, Any]]:
    response = requests.get(DUMP_INDEX_URL, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    objects = payload.get("objects") if isinstance(payload, dict) else payload
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"Unexpected LRCLIB dump index response: {payload!r}")
    return objects


def choose_dump(objects: list[dict[str, Any]], requested_key: str | None) -> dict[str, Any]:
    if requested_key:
        for obj in objects:
            if obj.get("key") == requested_key:
                return obj
        raise RuntimeError(f"Requested dump key not found in index: {requested_key}")
    return max(objects, key=lambda obj: str(obj.get("uploaded") or obj.get("key") or ""))


def stream_download(url: str, dest: Path, expected_size: int | None, user_agent: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return
    headers = {"User-Agent": user_agent}
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if existing and response.status_code == 416:
            partial.replace(dest)
            return
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
            headers.pop("Range", None)
            response.close()
            with requests.get(url, headers=headers, stream=True, timeout=120) as retry:
                retry.raise_for_status()
                with partial.open(mode) as handle:
                    for chunk in retry.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        else:
            response.raise_for_status()
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    partial.replace(dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dump-key", help="Specific dump key from the LRCLIB dump index. Defaults to latest upload.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--skip-sha256", action="store_true", help="Skip final full-file hash.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = args.output_root / args.snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_id": args.snapshot_id,
        "output_root": str(args.output_root),
        "dump_index_url": DUMP_INDEX_URL,
        "dump_base_url": DUMP_BASE_URL,
        "requested_dump_key": args.dump_key,
        "user_agent": args.user_agent,
        "d_free_gb_at_start": round(drive_free_gb(Path("D:/")), 3),
        "c_free_gb_at_start": round(drive_free_gb(Path("C:/")), 3),
        "rights_partition": "private_unknown_rights",
        "rights_note": "LRCLIB dump is openly downloadable, but underlying lyric rights remain unverified; keep private/personal-use only.",
    }
    (out_dir / "download_command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"event": "dump_index_started", "url": DUMP_INDEX_URL}), flush=True)
    objects = fetch_dump_index(args.user_agent)
    selected = choose_dump(objects, args.dump_key)
    key = str(selected["key"])
    expected_size = int(selected["size"]) if selected.get("size") is not None else None
    url = f"{DUMP_BASE_URL}/{key}"
    dest = out_dir / key
    print(
        json.dumps(
            {
                "event": "dump_download_started",
                "key": key,
                "url": url,
                "expected_size": expected_size,
                "dest": str(dest),
            }
        ),
        flush=True,
    )
    status = "completed"
    error = None
    try:
        stream_download(url, dest, expected_size, args.user_agent)
    except Exception as exc:  # pragma: no cover - operational path
        status = "failed"
        error = repr(exc)
        print(json.dumps({"event": "dump_download_failed", "error": error}), flush=True)

    sha256 = None
    if status == "completed" and dest.exists() and not args.skip_sha256:
        print(json.dumps({"event": "sha256_started", "path": str(dest)}), flush=True)
        sha256 = sha256_file(dest)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_id": args.snapshot_id,
        "source_id": "lrclib_db_dump",
        "source_url": "https://lrclib.net/db-dumps",
        "dump_index_url": DUMP_INDEX_URL,
        "dump_url": url,
        "dump_key": key,
        "dump_uploaded": selected.get("uploaded"),
        "expected_size_bytes": expected_size,
        "downloaded_size_bytes": dest.stat().st_size if dest.exists() else 0,
        "sha256": sha256,
        "status": status,
        "error": error,
        "output_dir": str(out_dir),
        "compressed_sqlite_path": str(dest),
        "rights_partition": "private_unknown_rights",
        "rights_note": "Private/personal-use only. Public LRCLIB availability does not prove underlying lyric rights.",
        "d_free_gb_at_end": round(drive_free_gb(Path("D:/")), 3),
        "c_free_gb_at_end": round(drive_free_gb(Path("C:/")), 3),
        "wall_seconds": round(time.time() - started, 3),
    }
    (out_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "dump_completed", "status": status, "bytes": manifest["downloaded_size_bytes"], "wall_seconds": manifest["wall_seconds"]}), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
