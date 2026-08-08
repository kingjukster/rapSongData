"""Normalize an LRCLIB SQLite dump into the private exact-dedupe lyric lane.

The raw LRCLIB dump is kept compressed as source-of-truth. This script
decompresses it into the output snapshot when needed, discovers the lyric table
schema, streams lyric rows, exact-dedupes against seed corpora, and writes an
``admitted_unique.parquet`` compatible with the private lyric lake.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from build_private_lyrics_dedupe_index import (
    ADMITTED_SCHEMA,
    SourceSpec,
    clean_text_for_output,
    fingerprint_text,
    make_admitted_record,
    seed_hashes_from_parquet,
    stable_hash,
)


DEFAULT_COMPRESSED_DUMP = Path(
    "C:/Users/kingj/rapSongData_overflow_review/lrclib_db_dumps/"
    "20260716_lrclib_dump_v1_20260715_2320/lrclib-db-dump-20260624T025818Z.sqlite3.gz"
)
DEFAULT_OUTPUT_ROOT = Path("C:/Users/kingj/rapSongData_overflow_review/normalized/lrclib_exact_dedupe")
DEFAULT_SNAPSHOT_ID = "20260716_lrclib_normalized_v1"

TIMESTAMP_RE = re.compile(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def drive_free_gib(path: Path) -> float:
    usage = shutil.disk_usage(Path(path.anchor or "."))
    return usage.free / (1024**3)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_synced_lyrics(value: str) -> str:
    lines: list[str] = []
    for line in value.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = TIMESTAMP_RE.sub("", line).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def decompress_gzip(source: Path, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    with gzip.open(source, "rb") as src, partial.open("wb") as out:
        shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
    partial.replace(dest)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        if row and not str(row[0]).startswith("sqlite_")
    ]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()]


def first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    normalized = {column.lower().replace("_", ""): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def discover_lrclib_table(conn: sqlite3.Connection) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for table in list_tables(conn):
        columns = table_columns(conn, table)
        plain = first_column(columns, ("plainLyrics", "plain_lyrics", "lyrics", "text"))
        synced = first_column(columns, ("syncedLyrics", "synced_lyrics", "lrc"))
        if not plain and not synced:
            continue
        title = first_column(columns, ("trackName", "name", "title", "songName", "song"))
        artist = first_column(columns, ("artistName", "artist", "artists"))
        album = first_column(columns, ("albumName", "album"))
        duration = first_column(columns, ("duration", "durationSeconds"))
        instrumental = first_column(columns, ("instrumental", "isInstrumental"))
        table_info = {
            "table": table,
            "columns": columns,
            "plain": plain,
            "synced": synced,
            "title": title,
            "artist": artist,
            "album": album,
            "duration": duration,
            "instrumental": instrumental,
        }
        if plain:
            return table_info
        best = best or table_info
    if best:
        return best
    raise RuntimeError("No LRCLIB table with plainLyrics/syncedLyrics-like columns was found")


def iter_lrclib_rows(
    conn: sqlite3.Connection,
    table_info: dict[str, Any],
    *,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    selected = [
        value
        for value in (
            table_info.get("plain"),
            table_info.get("synced"),
            table_info.get("title"),
            table_info.get("artist"),
            table_info.get("album"),
            table_info.get("duration"),
            table_info.get("instrumental"),
        )
        if value
    ]
    unique_selected = list(dict.fromkeys(selected))
    query = (
        f"SELECT rowid AS __rowid__, "
        + ", ".join(quote_identifier(column) for column in unique_selected)
        + f" FROM {quote_identifier(str(table_info['table']))}"
    )
    cursor = conn.execute(query)
    column_names = [description[0] for description in cursor.description]
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for values in rows:
            yield dict(zip(column_names, values))


def seed_hashes_from_jsonl(
    *,
    jsonl_path: Path,
    conn: sqlite3.Connection,
    batch_size: int,
) -> int:
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    loaded = 0
    pending: list[tuple[str, str, str, str, int]] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
        for row_number, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            lyrics = clean_text_for_output(row.get("lyrics") or row.get("text"))
            if not lyrics:
                continue
            normalized_hash = stable_hash(fingerprint_text(lyrics))
            record_id = str(row.get("record_id") or f"seed_jsonl_{loaded}")
            source_id = str(row.get("source_id") or "seed_jsonl")
            pending.append((normalized_hash, record_id, source_id, str(jsonl_path), row_number))
            loaded += 1
            if len(pending) >= batch_size:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO lyric_hashes
                    (normalized_hash, record_id, source_id, source_path, source_row)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    pending,
                )
                conn.commit()
                pending.clear()
                print(f"[lrclib] seed_jsonl_progress path={jsonl_path} loaded={loaded}", flush=True)
    if pending:
        conn.executemany(
            """
            INSERT OR IGNORE INTO lyric_hashes
            (normalized_hash, record_id, source_id, source_path, source_row)
            VALUES (?, ?, ?, ?, ?)
            """,
            pending,
        )
        conn.commit()
    return loaded


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
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
    return conn


def write_batch(writer: pq.ParquetWriter | None, path: Path, records: list[dict[str, Any]]) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(records, schema=ADMITTED_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(path, ADMITTED_SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressed-dump", type=Path, default=DEFAULT_COMPRESSED_DUMP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--seed-admitted-parquet", action="append", type=Path, default=[])
    parser.add_argument("--seed-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--write-batch-size", type=int, default=25_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-fingerprint-chars", type=int, default=80)
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--force-decompress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_root / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / args.compressed_dump.name.removesuffix(".gz")
    admitted_path = output_dir / "admitted_unique.parquet"
    summary_path = output_dir / "lrclib_normalization_summary.json"
    command_path = output_dir / "command.json"
    duplicate_path = output_dir / "duplicate_examples.csv"
    index_path = output_dir / "dedupe_index.sqlite"

    command_path.write_text(
        json.dumps(
            {
                "generated_at_utc": utc_now(),
                "command": {
                    "compressed_dump": str(args.compressed_dump),
                    "output_root": str(args.output_root),
                    "snapshot_id": args.snapshot_id,
                    "seed_admitted_parquet": [str(path) for path in args.seed_admitted_parquet],
                    "seed_jsonl": [str(path) for path in args.seed_jsonl],
                    "batch_size": args.batch_size,
                    "write_batch_size": args.write_batch_size,
                    "max_rows": args.max_rows,
                    "min_fingerprint_chars": args.min_fingerprint_chars,
                    "min_lines": args.min_lines,
                },
                "c_free_gib_at_start": round(drive_free_gib(Path("C:/")), 3),
                "d_free_gib_at_start": round(drive_free_gib(Path("D:/")), 3),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[lrclib] decompress_started source={args.compressed_dump} dest={sqlite_path}", flush=True)
    decompress_gzip(args.compressed_dump, sqlite_path, force=args.force_decompress)
    print(f"[lrclib] decompress_completed sqlite_bytes={sqlite_path.stat().st_size}", flush=True)

    seed_conn = init_db(index_path)
    seed_records_loaded = 0
    for seed_path in args.seed_admitted_parquet:
        print(f"[lrclib] seed_parquet_started path={seed_path}", flush=True)
        seed_records_loaded += seed_hashes_from_parquet(
            parquet_path=seed_path,
            conn=seed_conn,
            seen_hashes={},
            batch_size=max(1, args.batch_size),
        )
        print(f"[lrclib] seed_parquet_completed cumulative={seed_records_loaded}", flush=True)
    seed_jsonl_records_loaded = 0
    for seed_path in args.seed_jsonl:
        print(f"[lrclib] seed_jsonl_started path={seed_path}", flush=True)
        seed_jsonl_records_loaded += seed_hashes_from_jsonl(
            jsonl_path=seed_path,
            conn=seed_conn,
            batch_size=max(1, args.batch_size),
        )
        print(f"[lrclib] seed_jsonl_completed cumulative={seed_jsonl_records_loaded}", flush=True)

    source_spec = SourceSpec(
        source_id="lrclib_db_dump",
        source_family="lrclib",
        path=sqlite_path,
        format="sqlite",
        lyric_columns=("plainLyrics", "syncedLyrics"),
        title_columns=("trackName", "name", "title"),
        artist_columns=("artistName", "artist"),
    )
    stats: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    pending: list[dict[str, Any]] = []
    duplicate_examples = duplicate_path.open("w", encoding="utf-8", newline="")
    duplicate_writer = csv.DictWriter(
        duplicate_examples,
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
    duplicate_examples_written = 0
    admitted_tokens = 0
    table_info: dict[str, Any]

    try:
        lrclib_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        table_info = discover_lrclib_table(lrclib_conn)
        print(json.dumps({"event": "table_discovered", "table_info": table_info}, ensure_ascii=False), flush=True)
        for row in iter_lrclib_rows(lrclib_conn, table_info, batch_size=max(1, args.batch_size)):
            if args.max_rows and stats["scanned"] >= args.max_rows:
                break
            stats["scanned"] += 1
            if table_info.get("instrumental") and is_truthy(row.get(str(table_info["instrumental"]))):
                stats["instrumental"] += 1
                continue
            lyrics = clean_text_for_output(row.get(str(table_info.get("plain"))) if table_info.get("plain") else "")
            if not lyrics and table_info.get("synced"):
                lyrics = clean_text_for_output(normalize_synced_lyrics(str(row.get(str(table_info["synced"])) or "")))
            fingerprint = fingerprint_text(lyrics)
            if len(fingerprint) < args.min_fingerprint_chars:
                stats["empty_or_short"] += 1
                continue
            if len([line for line in lyrics.splitlines() if line.strip()]) < args.min_lines:
                stats["too_few_lines"] += 1
                continue
            normalized_hash = stable_hash(fingerprint)
            existing = seed_conn.execute(
                "SELECT record_id, source_id FROM lyric_hashes WHERE normalized_hash = ?",
                (normalized_hash,),
            ).fetchone()
            if existing:
                stats["duplicates"] += 1
                if duplicate_examples_written < 100_000:
                    duplicate_writer.writerow(
                        {
                            "duplicate_hash": normalized_hash,
                            "duplicate_source_id": source_spec.source_id,
                            "duplicate_source_path": str(sqlite_path),
                            "duplicate_source_row": int(row["__rowid__"]),
                            "canonical_record_id": existing[0],
                            "canonical_source_id": existing[1],
                        }
                    )
                    duplicate_examples_written += 1
                continue
            raw_row = {
                "title": row.get(str(table_info.get("title"))) if table_info.get("title") else "",
                "artist": row.get(str(table_info.get("artist"))) if table_info.get("artist") else "",
                "genre": "",
                "language": "",
                "year": "",
                "lyrics": lyrics,
            }
            admitted = make_admitted_record(
                spec=source_spec,
                source_path=str(sqlite_path),
                source_row=int(row["__rowid__"]),
                row=raw_row,
                normalized_hash=normalized_hash,
                lyrics=lyrics,
            )
            seed_conn.execute(
                "INSERT INTO lyric_hashes VALUES (?, ?, ?, ?, ?)",
                (normalized_hash, admitted["record_id"], source_spec.source_id, str(sqlite_path), int(row["__rowid__"])),
            )
            stats["unique"] += 1
            admitted_tokens += int(admitted["approx_tokens"])
            pending.append(admitted)
            if len(pending) >= args.write_batch_size:
                writer = write_batch(writer, admitted_path, pending)
                pending.clear()
                seed_conn.commit()
                duplicate_examples.flush()
            if stats["scanned"] and stats["scanned"] % args.progress_every == 0:
                print(
                    f"[lrclib] progress scanned={stats['scanned']} unique={stats['unique']} "
                    f"duplicates={stats['duplicates']} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        if pending:
            writer = write_batch(writer, admitted_path, pending)
            pending.clear()
    finally:
        if writer is not None:
            writer.close()
        duplicate_examples.close()
        seed_conn.commit()
        seed_conn.close()

    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "complete",
        "snapshot_id": args.snapshot_id,
        "source_id": "lrclib_db_dump",
        "compressed_dump": str(args.compressed_dump),
        "sqlite_path": str(sqlite_path),
        "table_info": table_info,
        "admitted_parquet": str(admitted_path),
        "duplicate_examples_csv": str(duplicate_path),
        "dedupe_index": str(index_path),
        "seed_records_loaded": int(seed_records_loaded),
        "seed_jsonl_records_loaded": int(seed_jsonl_records_loaded),
        "input_rows_scanned": int(stats["scanned"]),
        "unique_records": int(stats["unique"]),
        "duplicate_rows": int(stats["duplicates"]),
        "empty_or_short": int(stats["empty_or_short"]),
        "too_few_lines": int(stats["too_few_lines"]),
        "instrumental": int(stats["instrumental"]),
        "admitted_approx_tokens": int(admitted_tokens),
        "rights_partition": "private_unknown_rights",
        "rights_note": "Private/personal-use only. LRCLIB availability does not prove underlying lyric rights.",
        "c_free_gib_at_end": round(drive_free_gib(Path("C:/")), 3),
        "d_free_gib_at_end": round(drive_free_gib(Path("D:/")), 3),
        "wall_seconds": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
