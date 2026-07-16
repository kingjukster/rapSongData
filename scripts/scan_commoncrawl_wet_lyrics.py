"""Stream Common Crawl WET files for lyric-domain text.

This complements ``acquire_commoncrawl_lyrics.py``. The CDX/range fetcher is
precise but slow; this script takes CDX hits, maps their WARC files to matching
WET files, streams those WET files, and admits lyric-shaped text after exact
dedupe against existing private lyric parquets.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from acquire_commoncrawl_lyrics import COMMON_CRAWL_DATA, is_plausible_lyrics
from build_private_lyrics_dedupe_index import (
    ADMITTED_SCHEMA,
    SourceSpec,
    fingerprint_text,
    make_admitted_record,
    seed_hashes_from_parquet,
    stable_hash,
)


DEFAULT_OUTPUT_ROOT = Path("data/corpus_lake/raw/common_crawl_wet_lyrics")
DEFAULT_SNAPSHOT_ID = "20260716_commoncrawl_wet_lyrics_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def warc_to_wet_path(filename: str) -> str | None:
    if "/warc/" not in filename or not filename.endswith(".warc.gz"):
        return None
    return filename.replace("/warc/", "/wet/").removesuffix(".warc.gz") + ".warc.wet.gz"


def iter_cdx_wet_paths(paths: list[Path]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wet = warc_to_wet_path(str(row.get("filename") or ""))
                if wet:
                    counts[wet] += 1
    return counts.most_common()


def parse_warc_headers(stream: gzip.GzipFile) -> Iterator[tuple[dict[str, str], bytes]]:
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if not line.startswith(b"WARC/"):
            continue
        headers: dict[str, str] = {}
        while True:
            header_line = stream.readline()
            if not header_line:
                return
            if not header_line.strip():
                break
            text = header_line.decode("utf-8", errors="replace")
            if ":" in text:
                key, value = text.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        try:
            length = int(headers.get("content-length") or "0")
        except ValueError:
            length = 0
        body = stream.read(length) if length > 0 else b""
        yield headers, body


def normalize_wet_text(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def host_matches(url: str, domains: tuple[str, ...]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def url_has_lyric_hint(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(token in path for token in ("lyric", "lyrics", "/songs/view/"))


def source_id_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower().lstrip("www.")
    safe = re.sub(r"[^a-z0-9]+", "_", host).strip("_") or "unknown"
    return f"commoncrawl_wet_{safe}"


def write_parquet_batch(writer: pq.ParquetWriter | None, parquet_path: Path, rows: list[dict[str, Any]]) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=ADMITTED_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(parquet_path, ADMITTED_SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scan_wet_file(
    *,
    wet_path: str,
    domains: tuple[str, ...],
    timeout_seconds: int,
) -> Iterator[tuple[str, str]]:
    url = f"{COMMON_CRAWL_DATA}/{wet_path}"
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as gz:
            for headers, body in parse_warc_headers(gz):
                if headers.get("warc-type") != "conversion":
                    continue
                target_url = headers.get("warc-target-uri") or ""
                if not target_url or not host_matches(target_url, domains):
                    continue
                if not url_has_lyric_hint(target_url):
                    continue
                text = normalize_wet_text(body)
                if is_plausible_lyrics(text):
                    yield target_url, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--cdx-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--wet-path", action="append", default=[])
    parser.add_argument("--domain", action="append", default=["lyrics.com", "songlyrics.com", "songmeanings.com", "azlyrics.com", "genius.com"])
    parser.add_argument("--seed-admitted-parquet", action="append", type=Path, default=[])
    parser.add_argument("--max-wet-files", type=int, default=25)
    parser.add_argument("--max-admitted", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--write-batch-size", type=int, default=1000)
    parser.add_argument("--seed-batch-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw_wet_captures.jsonl"
    rejected_jsonl = output_dir / "rejected_wet_captures.jsonl"
    parquet_path = output_dir / "admitted_wet_unique.parquet"
    duplicate_csv_path = output_dir / "duplicate_examples.csv"
    summary_path = output_dir / "wet_scan_summary.json"
    command_path = output_dir / "command.json"
    for stale in (raw_jsonl, rejected_jsonl, parquet_path, duplicate_csv_path, summary_path):
        if stale.exists():
            stale.unlink()
    command_path.write_text(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "args": {
                    key: str(value) if isinstance(value, Path) else [str(v) for v in value] if isinstance(value, list) else value
                    for key, value in vars(args).items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    wet_counts = Counter(dict(iter_cdx_wet_paths(args.cdx_jsonl)))
    for wet_path in args.wet_path:
        wet_counts[wet_path] += 1
    selected_wet_paths = [path for path, _ in wet_counts.most_common(args.max_wet_files or None)]
    if not selected_wet_paths:
        raise SystemExit("No WET paths selected. Provide --cdx-jsonl or --wet-path.")

    started = time.time()
    seen_hashes: dict[str, tuple[str, str]] = {}
    seed_records_loaded = 0
    for seed in args.seed_admitted_parquet:
        print(f"[wet] seed_started path={seed}", flush=True)
        seed_records_loaded += seed_hashes_from_parquet(
            parquet_path=seed,
            conn=None,
            seen_hashes=seen_hashes,
            batch_size=max(1, args.seed_batch_size),
        )
        print(f"[wet] seed_completed cumulative={seed_records_loaded}", flush=True)

    stats = Counter()
    by_source: dict[str, Counter[str]] = {}
    pending: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    duplicate_csv = duplicate_csv_path.open("w", encoding="utf-8", newline="")
    duplicate_writer = csv.DictWriter(duplicate_csv, ["duplicate_hash", "duplicate_url", "canonical_record_id", "canonical_source_id"])
    duplicate_writer.writeheader()
    try:
        for wet_index, wet_path in enumerate(selected_wet_paths, start=1):
            print(f"[wet] file_started index={wet_index}/{len(selected_wet_paths)} path={wet_path}", flush=True)
            file_started = time.time()
            try:
                captures = scan_wet_file(wet_path=wet_path, domains=tuple(args.domain), timeout_seconds=args.timeout_seconds)
                for target_url, lyrics in captures:
                    stats["captures"] += 1
                    source_id = source_id_for_url(target_url)
                    source_stats = by_source.setdefault(source_id, Counter())
                    source_stats["captures"] += 1
                    normalized_hash = stable_hash(fingerprint_text(lyrics))
                    append_jsonl(raw_jsonl, {"wet_path": wet_path, "url": target_url, "source_id": source_id, "normalized_hash": normalized_hash, "lyrics": lyrics})
                    existing = seen_hashes.get(normalized_hash)
                    if existing:
                        stats["duplicates"] += 1
                        source_stats["duplicates"] += 1
                        duplicate_writer.writerow(
                            {
                                "duplicate_hash": normalized_hash,
                                "duplicate_url": target_url,
                                "canonical_record_id": existing[0],
                                "canonical_source_id": existing[1],
                            }
                        )
                        continue
                    spec = SourceSpec(
                        source_id=source_id,
                        source_family="common_crawl_wet",
                        path=Path(wet_path),
                        format="wet_stream",
                        lyric_columns=("lyrics",),
                    )
                    admitted = make_admitted_record(
                        spec=spec,
                        source_path=f"{wet_path}::{target_url}",
                        source_row=stats["captures"],
                        row={"title": "", "artist": "", "lyrics": lyrics},
                        normalized_hash=normalized_hash,
                        lyrics=lyrics,
                    )
                    seen_hashes[normalized_hash] = (admitted["record_id"], source_id)
                    pending.append(admitted)
                    stats["admitted_records"] += 1
                    stats["admitted_approx_tokens"] += int(admitted["approx_tokens"])
                    source_stats["admitted"] += 1
                    if len(pending) >= args.write_batch_size:
                        writer = write_parquet_batch(writer, parquet_path, pending)
                        pending.clear()
                    if args.max_admitted and stats["admitted_records"] >= args.max_admitted:
                        print(f"[wet] max_admitted_reached max={args.max_admitted}", flush=True)
                        raise KeyboardInterrupt
                stats["wet_files_completed"] += 1
                print(
                    f"[wet] file_completed path={wet_path} elapsed={time.time() - file_started:.1f}s "
                    f"captures={stats['captures']} admitted={stats['admitted_records']} duplicates={stats['duplicates']}",
                    flush=True,
                )
            except Exception as exc:
                stats["wet_file_errors"] += 1
                append_jsonl(rejected_jsonl, {"stage": "wet_file", "wet_path": wet_path, "error": repr(exc)})
                print(f"[wet] file_failed path={wet_path} error={exc!r}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if pending:
            writer = write_parquet_batch(writer, parquet_path, pending)
            pending.clear()
        if writer is not None:
            writer.close()
        duplicate_csv.close()

    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_id": args.snapshot_id,
        "output_dir": str(output_dir),
        "raw_wet_captures_jsonl": str(raw_jsonl),
        "rejected_wet_captures_jsonl": str(rejected_jsonl),
        "admitted_parquet": str(parquet_path) if parquet_path.exists() else None,
        "duplicate_examples_csv": str(duplicate_csv_path),
        "rights_partition": "private_unknown_rights",
        "seed_admitted_parquet": [str(seed) for seed in args.seed_admitted_parquet],
        "seed_records_loaded": seed_records_loaded,
        "selected_wet_paths": selected_wet_paths,
        "domains": list(args.domain),
        "stats": dict(stats),
        "sources": {source_id: dict(source_stats) for source_id, source_stats in by_source.items()},
        "wall_seconds": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
