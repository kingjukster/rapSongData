"""Acquire lyric candidates from Common Crawl with provenance and exact dedupe.

This is the web-scale lane for the private lyric corpus lake. It queries the
Common Crawl CDX index for lyric-like URL patterns, fetches archived WARC
records by HTTP range request, extracts likely lyric text, and writes:

- raw_captures.jsonl: every successfully extracted capture with provenance
- admitted_web_unique.parquet: rows not already present in seed dedupe indexes
- rejected_captures.jsonl: fetch/extract/duplicate failures for auditability
- acquisition_summary.json / command.json: run metadata

The script is intentionally polite: bounded limits, request delay, timeouts, and
no live-site scraping.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from build_private_lyrics_dedupe_index import (
    ADMITTED_SCHEMA,
    SourceSpec,
    fingerprint_text,
    make_admitted_record,
    seed_hashes_from_parquet,
    stable_hash,
)


DEFAULT_OUTPUT_ROOT = Path("data/corpus_lake/raw/common_crawl_lyrics")
DEFAULT_SNAPSHOT_ID = "20260715_commoncrawl_lyrics_v1"
COMMON_CRAWL_INDEX = "https://index.commoncrawl.org"
COMMON_CRAWL_DATA = "https://data.commoncrawl.org"

DEFAULT_CRAWLS = (
    "CC-MAIN-2024-18",
    "CC-MAIN-2024-10",
    "CC-MAIN-2023-50",
    "CC-MAIN-2023-40",
    "CC-MAIN-2022-49",
)

DEFAULT_PATTERNS = (
    "lyrics.com/lyric/*",
    "www.lyrics.com/lyric/*",
    "www.songlyrics.com/*/*-lyrics/",
    "songlyrics.com/*/*-lyrics/",
    "genius.com/*",
    "azlyrics.com/lyrics/*",
    "www.azlyrics.com/lyrics/*",
)

BLOCKLIST_TEXT_RE = re.compile(
    r"(access denied|captcha|cloudflare|enable javascript|temporarily unavailable|privacy policy|terms of use)",
    re.IGNORECASE,
)
SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_LINE_RE = re.compile(r"[ \t]+")


class TextStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.parts.append(unescape(f"&#{name};"))

    def text(self) -> str:
        return "\n".join(self.parts)


@dataclass
class ExtractResult:
    lyrics: str
    extractor: str
    title: str = ""
    artist: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_cdx_records(
    *,
    crawl: str,
    pattern: str,
    limit: int,
    timeout_seconds: int,
) -> Iterator[dict[str, Any]]:
    api = f"{COMMON_CRAWL_INDEX}/{crawl}-index"
    params = [
        ("url", pattern),
        ("output", "json"),
        ("filter", "status:200"),
        ("filter", "mime:text/html"),
        ("limit", str(limit)),
    ]
    response = requests.get(api, params=params, timeout=timeout_seconds)
    if response.status_code == 404:
        return
    response.raise_for_status()
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def fetch_warc_body(record: dict[str, Any], timeout_seconds: int) -> bytes:
    offset = int(record["offset"])
    length = int(record["length"])
    url = f"{COMMON_CRAWL_DATA}/{record['filename']}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    decompressed = gzip.decompress(response.content)
    # WARC headers, then HTTP response headers, then body.
    parts = decompressed.split(b"\r\n\r\n", 2)
    if len(parts) < 3:
        parts = decompressed.split(b"\n\n", 2)
    if len(parts) < 3:
        raise ValueError("Could not split WARC/HTTP/body payload")
    return parts[2]


def strip_html(fragment: str) -> str:
    fragment = BR_RE.sub("\n", fragment)
    fragment = TAG_RE.sub("", fragment)
    text = unescape(fragment)
    lines = [WHITESPACE_LINE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def generic_visible_text(html: str) -> str:
    cleaned = SCRIPT_STYLE_RE.sub("", html)
    parser = TextStripper()
    parser.feed(cleaned)
    lines = [WHITESPACE_LINE_RE.sub(" ", line).strip() for line in parser.text().splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return strip_html(match.group(1))[:300]


def extract_lyrics(url: str, html_bytes: bytes) -> ExtractResult | None:
    html = html_bytes.decode("utf-8", errors="replace")
    title = extract_title(html)
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    extractor_patterns: list[tuple[str, re.Pattern[str]]] = []
    if "lyrics.com" in host:
        extractor_patterns.append(
            (
                "lyrics.com:pre#lyric-body-text",
                re.compile(r"<pre[^>]+id=[\"']lyric-body-text[\"'][^>]*>(.*?)</pre>", re.IGNORECASE | re.DOTALL),
            )
        )
    if "songlyrics.com" in host:
        extractor_patterns.append(
            (
                "songlyrics.com:p#songLyricsDiv",
                re.compile(r"<p[^>]+id=[\"']songLyricsDiv[\"'][^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL),
            )
        )
    if "azlyrics.com" in host:
        extractor_patterns.append(
            (
                "azlyrics.com:main-lyric-div",
                re.compile(
                    r"<!--\s*Usage of azlyrics\.com content.*?-->\s*<div[^>]*>(.*?)</div>",
                    re.IGNORECASE | re.DOTALL,
                ),
            )
        )
    if "genius.com" in host:
        extractor_patterns.append(
            (
                "genius.com:lyrics-container",
                re.compile(r"<div[^>]+data-lyrics-container=[\"']true[\"'][^>]*>(.*?)</div>", re.IGNORECASE | re.DOTALL),
            )
        )

    for extractor, pattern in extractor_patterns:
        matches = pattern.findall(html)
        if not matches:
            continue
        text = "\n".join(strip_html(match) for match in matches)
        if is_plausible_lyrics(text):
            return ExtractResult(lyrics=text, extractor=extractor, title=title)

    if parsed.path.lower().endswith("lyrics/") or "lyric" in parsed.path.lower() or "-lyrics" in parsed.path.lower():
        text = generic_visible_text(html)
        if is_plausible_lyrics(text):
            return ExtractResult(lyrics=text, extractor="generic_visible_text", title=title)
    return None


def is_plausible_lyrics(text: str) -> bool:
    if not text:
        return False
    if BLOCKLIST_TEXT_RE.search(text[:3000]):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    fingerprint = fingerprint_text(text)
    if len(fingerprint) < 80:
        return False
    if len(text) < 240:
        return False
    return True


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def source_id_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower().lstrip("www.")
    safe = re.sub(r"[^a-z0-9]+", "_", host).strip("_") or "unknown"
    return f"commoncrawl_{safe}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--crawl", action="append", default=[], help="Common Crawl id, repeatable. Defaults to a curated recent/backfill set.")
    parser.add_argument("--pattern", action="append", default=[], help="CDX URL pattern, repeatable.")
    parser.add_argument("--records-per-pattern", type=int, default=100)
    parser.add_argument("--max-admitted", type=int, default=0, help="Stop after this many new unique records; 0 means no cap.")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--write-batch-size", type=int, default=500)
    parser.add_argument("--seed-admitted-parquet", action="append", type=Path, default=[])
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def write_parquet_batch(writer: pq.ParquetWriter | None, parquet_path: Path, rows: list[dict[str, Any]]) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=ADMITTED_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(parquet_path, ADMITTED_SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def main() -> int:
    args = parse_args()
    crawls = tuple(args.crawl or DEFAULT_CRAWLS)
    patterns = tuple(args.pattern or DEFAULT_PATTERNS)
    output_dir = args.output_root / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = output_dir / "raw_captures.jsonl"
    rejected_jsonl = output_dir / "rejected_captures.jsonl"
    cdx_jsonl = output_dir / "cdx_records.jsonl"
    parquet_path = output_dir / "admitted_web_unique.parquet"
    summary_path = output_dir / "acquisition_summary.json"
    command_path = output_dir / "command.json"
    duplicate_csv_path = output_dir / "duplicate_examples.csv"

    for path in (raw_jsonl, rejected_jsonl, cdx_jsonl, duplicate_csv_path):
        if path.exists():
            path.unlink()

    command_path.write_text(
        json.dumps({"generated_at_utc": utc_now(), "args": jsonable_args(args), "output_dir": str(output_dir)}, indent=2),
        encoding="utf-8",
    )

    start = time.time()
    seen_hashes: dict[str, tuple[str, str]] = {}
    seed_records_loaded = 0
    for seed_path in args.seed_admitted_parquet or []:
        print(f"[cc] seed_started path={seed_path}", flush=True)
        seed_records_loaded += seed_hashes_from_parquet(
            parquet_path=seed_path,
            conn=None,
            seen_hashes=seen_hashes,
            batch_size=50_000,
        )
        print(f"[cc] seed_completed cumulative={seed_records_loaded}", flush=True)

    duplicate_csv = duplicate_csv_path.open("w", encoding="utf-8", newline="")
    duplicate_writer = csv.DictWriter(
        duplicate_csv,
        ["duplicate_hash", "duplicate_url", "canonical_record_id", "canonical_source_id"],
    )
    duplicate_writer.writeheader()

    writer: pq.ParquetWriter | None = None
    pending: list[dict[str, Any]] = []
    stats = {
        "cdx_records": 0,
        "fetch_attempts": 0,
        "fetch_errors": 0,
        "extract_failures": 0,
        "duplicates": 0,
        "admitted_records": 0,
        "admitted_approx_tokens": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    visited_digests: set[str] = set()

    try:
        for crawl in crawls:
            for pattern in patterns:
                print(f"[cc] query_started crawl={crawl} pattern={pattern}", flush=True)
                try:
                    records = list(
                        iter_cdx_records(
                            crawl=crawl,
                            pattern=pattern,
                            limit=args.records_per_pattern,
                            timeout_seconds=args.timeout_seconds,
                        )
                    )
                except Exception as exc:
                    append_jsonl(rejected_jsonl, {"stage": "cdx_query", "crawl": crawl, "pattern": pattern, "error": repr(exc)})
                    print(f"[cc] query_failed crawl={crawl} pattern={pattern} error={exc!r}", flush=True)
                    continue
                print(f"[cc] query_completed crawl={crawl} pattern={pattern} records={len(records)}", flush=True)
                for record in records:
                    digest = record.get("digest") or f"{record.get('filename')}:{record.get('offset')}"
                    if digest in visited_digests:
                        continue
                    visited_digests.add(digest)
                    stats["cdx_records"] += 1
                    append_jsonl(cdx_jsonl, {"crawl": crawl, "pattern": pattern, **record})
                    url = record.get("url", "")
                    source_id = source_id_for_url(url)
                    source_stats = by_source.setdefault(source_id, {"fetched": 0, "admitted": 0, "duplicates": 0, "rejected": 0})
                    time.sleep(max(0.0, args.request_delay_seconds))
                    stats["fetch_attempts"] += 1
                    source_stats["fetched"] += 1
                    try:
                        body = fetch_warc_body(record, timeout_seconds=args.timeout_seconds)
                        extracted = extract_lyrics(url, body)
                    except Exception as exc:
                        stats["fetch_errors"] += 1
                        source_stats["rejected"] += 1
                        append_jsonl(rejected_jsonl, {"stage": "fetch_or_parse", "crawl": crawl, "url": url, "record": record, "error": repr(exc)})
                        continue
                    if extracted is None:
                        stats["extract_failures"] += 1
                        source_stats["rejected"] += 1
                        append_jsonl(rejected_jsonl, {"stage": "extract", "crawl": crawl, "url": url, "record": record})
                        continue

                    fingerprint = fingerprint_text(extracted.lyrics)
                    normalized_hash = stable_hash(fingerprint)
                    existing = seen_hashes.get(normalized_hash)
                    raw_record = {
                        "crawl": crawl,
                        "pattern": pattern,
                        "url": url,
                        "timestamp": record.get("timestamp"),
                        "digest": record.get("digest"),
                        "filename": record.get("filename"),
                        "offset": record.get("offset"),
                        "length": record.get("length"),
                        "extractor": extracted.extractor,
                        "title": extracted.title,
                        "lyrics": extracted.lyrics,
                        "normalized_hash": normalized_hash,
                    }
                    append_jsonl(raw_jsonl, raw_record)

                    if existing:
                        stats["duplicates"] += 1
                        source_stats["duplicates"] += 1
                        duplicate_writer.writerow(
                            {
                                "duplicate_hash": normalized_hash,
                                "duplicate_url": url,
                                "canonical_record_id": existing[0],
                                "canonical_source_id": existing[1],
                            }
                        )
                        duplicate_csv.flush()
                        continue

                    spec = SourceSpec(
                        source_id=source_id,
                        source_family="common_crawl",
                        path=Path(record.get("filename", "")),
                        format="warc_range",
                        lyric_columns=("lyrics",),
                    )
                    admitted = make_admitted_record(
                        spec=spec,
                        source_path=f"{record.get('filename')}#{record.get('offset')}:{record.get('length')}",
                        source_row=int(record.get("offset") or 0),
                        row={"title": extracted.title, "artist": "", "lyrics": extracted.lyrics},
                        normalized_hash=normalized_hash,
                        lyrics=extracted.lyrics,
                    )
                    seen_hashes[normalized_hash] = (admitted["record_id"], source_id)
                    stats["admitted_records"] += 1
                    stats["admitted_approx_tokens"] += int(admitted["approx_tokens"])
                    source_stats["admitted"] += 1
                    pending.append(admitted)
                    if len(pending) >= args.write_batch_size:
                        writer = write_parquet_batch(writer, parquet_path, pending)
                        pending.clear()

                    if args.max_admitted and stats["admitted_records"] >= args.max_admitted:
                        print(f"[cc] max_admitted_reached max={args.max_admitted}", flush=True)
                        raise KeyboardInterrupt
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
        "raw_captures_jsonl": str(raw_jsonl),
        "rejected_captures_jsonl": str(rejected_jsonl),
        "cdx_records_jsonl": str(cdx_jsonl),
        "admitted_parquet": str(parquet_path) if parquet_path.exists() else None,
        "duplicate_examples_csv": str(duplicate_csv_path),
        "rights_partition": "private_unknown_rights",
        "seed_records_loaded": int(seed_records_loaded),
        "crawls": list(crawls),
        "patterns": list(patterns),
        "stats": stats,
        "sources": by_source,
        "wall_seconds": round(time.time() - start, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
