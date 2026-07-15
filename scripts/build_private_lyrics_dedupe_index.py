"""Build a streaming exact-dedupe admission index for private lyric sources.

The corpus lake keeps raw snapshots untouched. This script reads selected raw
song-level sources, normalizes lyric text into a stable fingerprint, writes one
canonical admitted record per unique lyric, and records duplicate/source stats.

The first preset intentionally targets the large Genius-family mirrors because
they are high-value and highly overlapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_OUTPUT_ROOT = Path("data/corpus_lake/normalized/private_lyrics_exact_dedupe")
DEFAULT_SNAPSHOT_ID = "20260715_genius_family_v1"

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"</?[a-z][^>]{0,200}>", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
FINGERPRINT_RE = re.compile(r"[^a-z0-9']+")
TRAILING_EMBED_RE = re.compile(r"\s*\d*\s*embed\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_family: str
    path: Path
    format: str
    lyric_columns: tuple[str, ...]
    title_columns: tuple[str, ...] = ("title", "Title", "SName", "song", "track_name")
    artist_columns: tuple[str, ...] = ("artist", "Artist", "ALink", "artist_name")
    genre_columns: tuple[str, ...] = ("tag", "genre", "Genre")
    language_columns: tuple[str, ...] = ("language", "language_ft", "language_cld3")
    year_columns: tuple[str, ...] = ("year", "Year")
    include_glob: str | None = None


GENIUS_FAMILY_SOURCES = (
    SourceSpec(
        source_id="hf_dr3dre_genius_song_lyrics_cleaned",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/Dr3dre__Genius-song-lyrics-cleaned/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("lyrics", "lyrics_clean"),
        include_glob="*.parquet",
    ),
    SourceSpec(
        source_id="kaggle_carlosgdcj_genius_language_info",
        source_family="kaggle",
        path=Path(
            "data/corpus_lake/raw/kaggle_private_lyrics/carlosgdcj__genius-song-lyrics-with-language-information/20260715_credentialed_download/song_lyrics.csv"
        ),
        format="csv",
        lyric_columns=("lyrics",),
    ),
    SourceSpec(
        source_id="kaggle_nikhilnayak123_5m_song_lyrics",
        source_family="kaggle",
        path=Path(
            "data/corpus_lake/raw/kaggle_private_lyrics/nikhilnayak123__5-million-song-lyrics-dataset/20260715_credentialed_download/ds2.csv"
        ),
        format="csv",
        lyric_columns=("lyrics",),
    ),
    SourceSpec(
        source_id="hf_amishshah_song_lyrics_min",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/amishshah__song_lyrics/20260715_hf_public_snapshot/song_lyrics_min.csv"),
        format="csv",
        lyric_columns=("lyrics",),
    ),
    SourceSpec(
        source_id="hf_theelderemo_genius_lyrics_cleaned",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/theelderemo__genius-lyrics-cleaned/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("lyrics",),
        include_glob="*.parquet",
    ),
)

ADMITTED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_family", pa.string()),
        ("source_path", pa.string()),
        ("source_row", pa.int64()),
        ("title", pa.string()),
        ("artist", pa.string()),
        ("genre", pa.string()),
        ("language", pa.string()),
        ("year", pa.string()),
        ("lyrics", pa.string()),
        ("normalized_hash", pa.string()),
        ("char_count", pa.int64()),
        ("word_count", pa.int64()),
        ("approx_tokens", pa.int64()),
        ("rights_partition", pa.string()),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def first_present(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column in row:
            value = row.get(column)
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text:
                    return text
    return ""


def clean_text_for_output(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        if pd.isna(value):
            return ""
        text = str(value)
    else:
        text = value
    if "&" in text:
        text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = TRAILING_EMBED_RE.sub("", text)
    return text.strip()


def fingerprint_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    normalized = CONTROL_RE.sub("", normalized)
    normalized = TRAILING_EMBED_RE.sub("", normalized)
    normalized = FINGERPRINT_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def choose_lyrics(row: dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column in row:
            cleaned = clean_text_for_output(row.get(column))
            if cleaned:
                return cleaned
    return ""


def iter_source_rows(spec: SourceSpec, chunksize: int) -> Iterator[tuple[str, int, dict[str, Any]]]:
    if spec.format == "csv":
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        row_offset = 0
        for chunk in pd.read_csv(spec.path, chunksize=chunksize, encoding_errors="replace", low_memory=False):
            for row in chunk.to_dict("records"):
                yield str(spec.path), row_offset, row
                row_offset += 1
        return

    if spec.format == "parquet_dir":
        files = sorted(spec.path.glob(spec.include_glob or "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {spec.path}")
        for file_path in files:
            parquet = pq.ParquetFile(file_path)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=chunksize):
                table = pa.Table.from_batches([batch])
                for row in table.to_pylist():
                    yield str(file_path), row_offset, row
                    row_offset += 1
        return

    raise ValueError(f"Unsupported source format for {spec.source_id}: {spec.format}")


def make_admitted_record(
    *,
    spec: SourceSpec,
    source_path: str,
    source_row: int,
    row: dict[str, Any],
    normalized_hash: str,
    lyrics: str,
) -> dict[str, Any]:
    char_count = len(lyrics)
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", lyrics)
    record_id = f"priv_lyrics_{normalized_hash[:20]}"
    return {
        "record_id": record_id,
        "source_id": spec.source_id,
        "source_family": spec.source_family,
        "source_path": source_path,
        "source_row": int(source_row),
        "title": first_present(row, spec.title_columns),
        "artist": first_present(row, spec.artist_columns),
        "genre": first_present(row, spec.genre_columns),
        "language": first_present(row, spec.language_columns),
        "year": first_present(row, spec.year_columns),
        "lyrics": lyrics,
        "normalized_hash": normalized_hash,
        "char_count": char_count,
        "word_count": len(words),
        "approx_tokens": max(1, char_count // 4),
        "rights_partition": "private_unknown_rights",
    }


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lyric_hashes (
            normalized_hash TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_row INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS duplicate_examples (
            duplicate_hash TEXT NOT NULL,
            duplicate_source_id TEXT NOT NULL,
            duplicate_source_path TEXT NOT NULL,
            duplicate_source_row INTEGER NOT NULL,
            canonical_record_id TEXT NOT NULL,
            canonical_source_id TEXT NOT NULL
        )
        """
    )
    return conn


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Private lyrics exact dedupe report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        f"Preset: `{summary['preset']}`",
        "",
        "## Totals",
        "",
        f"- Input rows scanned: {summary['input_rows_scanned']:,}",
        f"- Empty/invalid lyric rows: {summary['empty_lyrics']:,}",
        f"- Unique admitted records: {summary['unique_records']:,}",
        f"- Exact duplicate rows: {summary['duplicate_rows']:,}",
        f"- Approx admitted tokens: {summary['admitted_approx_tokens']:,}",
        "",
        "## By source",
        "",
        "| Source | Scanned | Unique admitted | Duplicates | Empty |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source_id, stats in summary["sources"].items():
        lines.append(
            f"| `{source_id}` | {stats.get('scanned', 0):,} | {stats.get('unique', 0):,} | "
            f"{stats.get('duplicates', 0):,} | {stats.get('empty', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw source files are preserved; this stage writes a deduped private-use normalized index.",
            "- Deduplication is exact over a normalized lyric fingerprint, not semantic near-dedupe.",
            "- All admitted rows remain in `private_unknown_rights` until underlying lyric rights are separately verified.",
        ]
    )
    (output_dir / "dedupe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["genius_family"], default="genius_family")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--max-rows-per-source", type=int, default=0, help="Smoke-test cap per source; 0 means no cap.")
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Source id to skip for this run; repeatable. Useful for known source-level duplicate mirrors.",
    )
    parser.add_argument(
        "--index-backend",
        choices=["memory", "sqlite"],
        default="memory",
        help="memory is fastest and expected to fit this workstation; sqlite is safer but much slower.",
    )
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main() -> int:
    args = parse_args()
    excluded_sources = set(args.exclude_source or [])
    sources = tuple(spec for spec in GENIUS_FAMILY_SOURCES if spec.source_id not in excluded_sources)
    output_dir = args.output_root / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "dedupe_index.sqlite"
    duplicate_csv_path = output_dir / "duplicate_examples.csv"
    parquet_path = output_dir / "admitted_unique.parquet"
    summary_path = output_dir / "dedupe_summary.json"
    command_path = output_dir / "command.json"

    command_path.write_text(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "args": jsonable_args(args),
                "preset": args.preset,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start = time.time()
    conn = init_db(db_path) if args.index_backend == "sqlite" else None
    seen_hashes: dict[str, tuple[str, str]] = {}
    duplicate_csv = duplicate_csv_path.open("w", encoding="utf-8", newline="")
    duplicate_writer = csv.DictWriter(
        duplicate_csv,
        [
            "duplicate_hash",
            "duplicate_source_id",
            "duplicate_source_path",
            "duplicate_source_row",
            "canonical_record_id",
            "canonical_source_id",
        ],
    )
    duplicate_writer.writeheader()
    writer: pq.ParquetWriter | None = None
    pending_records: list[dict[str, Any]] = []
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    duplicate_examples_written = 0
    admitted_tokens = 0

    try:
        for spec in sources:
            seen_for_source = 0
            print(f"[dedupe] source_started source_id={spec.source_id} path={spec.path}", flush=True)
            for source_path, source_row, row in iter_source_rows(spec, args.chunksize):
                if args.max_rows_per_source and seen_for_source >= args.max_rows_per_source:
                    break
                seen_for_source += 1
                stats = source_stats[spec.source_id]
                stats["scanned"] += 1

                lyrics = choose_lyrics(row, spec.lyric_columns)
                fingerprint = fingerprint_text(lyrics)
                if len(fingerprint) < 80:
                    stats["empty"] += 1
                    continue
                normalized_hash = stable_hash(fingerprint)

                if args.index_backend == "sqlite":
                    assert conn is not None
                    existing = conn.execute(
                        "SELECT record_id, source_id FROM lyric_hashes WHERE normalized_hash = ?",
                        (normalized_hash,),
                    ).fetchone()
                else:
                    existing = seen_hashes.get(normalized_hash)
                if existing:
                    stats["duplicates"] += 1
                    if duplicate_examples_written < 100_000:
                        duplicate_writer.writerow(
                            {
                                "duplicate_hash": normalized_hash,
                                "duplicate_source_id": spec.source_id,
                                "duplicate_source_path": source_path,
                                "duplicate_source_row": source_row,
                                "canonical_record_id": existing[0],
                                "canonical_source_id": existing[1],
                            }
                        )
                        duplicate_examples_written += 1
                    continue

                admitted = make_admitted_record(
                    spec=spec,
                    source_path=source_path,
                    source_row=source_row,
                    row=row,
                    normalized_hash=normalized_hash,
                    lyrics=lyrics,
                )
                if args.index_backend == "sqlite":
                    assert conn is not None
                    conn.execute(
                        "INSERT INTO lyric_hashes VALUES (?, ?, ?, ?, ?)",
                        (normalized_hash, admitted["record_id"], spec.source_id, source_path, source_row),
                    )
                else:
                    seen_hashes[normalized_hash] = (admitted["record_id"], spec.source_id)
                stats["unique"] += 1
                admitted_tokens += int(admitted["approx_tokens"])
                pending_records.append(admitted)

                if len(pending_records) >= args.write_batch_size:
                    table = pa.Table.from_pylist(pending_records, schema=ADMITTED_SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, ADMITTED_SCHEMA, compression="zstd")
                    writer.write_table(table)
                    pending_records.clear()

                if stats["scanned"] % args.progress_every == 0:
                    if conn is not None:
                        conn.commit()
                    duplicate_csv.flush()
                    elapsed = time.time() - start
                    print(
                        f"[dedupe] progress source_id={spec.source_id} scanned={stats['scanned']} "
                        f"unique={stats['unique']} duplicates={stats['duplicates']} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            if conn is not None:
                conn.commit()
            duplicate_csv.flush()
            print(
                f"[dedupe] source_completed source_id={spec.source_id} scanned={source_stats[spec.source_id]['scanned']} "
                f"unique={source_stats[spec.source_id]['unique']} duplicates={source_stats[spec.source_id]['duplicates']} "
                f"empty={source_stats[spec.source_id]['empty']}",
                flush=True,
            )

        if pending_records:
            table = pa.Table.from_pylist(pending_records, schema=ADMITTED_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, ADMITTED_SCHEMA, compression="zstd")
            writer.write_table(table)
            pending_records.clear()

    finally:
        if writer is not None:
            writer.close()
        duplicate_csv.close()
        if conn is not None:
            conn.commit()

    total_scanned = sum(stats["scanned"] for stats in source_stats.values())
    total_unique = sum(stats["unique"] for stats in source_stats.values())
    total_duplicates = sum(stats["duplicates"] for stats in source_stats.values())
    total_empty = sum(stats["empty"] for stats in source_stats.values())
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "preset": args.preset,
        "snapshot_id": args.snapshot_id,
        "output_dir": str(output_dir),
        "admitted_parquet": str(parquet_path),
        "sqlite_index": str(db_path) if args.index_backend == "sqlite" else None,
        "duplicate_examples_csv": str(duplicate_csv_path),
        "rights_partition": "private_unknown_rights",
        "dedupe_method": "exact_sha256_over_nfkc_lower_alnum_apostrophe_fingerprint",
        "index_backend": args.index_backend,
        "input_rows_scanned": int(total_scanned),
        "empty_lyrics": int(total_empty),
        "unique_records": int(total_unique),
        "duplicate_rows": int(total_duplicates),
        "admitted_approx_tokens": int(admitted_tokens),
        "wall_seconds": round(time.time() - start, 3),
        "sources": {source_id: dict(stats) for source_id, stats in source_stats.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary)
    if conn is not None:
        conn.close()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
