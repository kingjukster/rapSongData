"""Materialize a private lyric-lake scratch profile from admitted parquet sources.

This is intentionally streaming and exact-dedupe-only. The upstream source
normalizers already performed source-specific cleaning; this script composes
their admitted parquet outputs into train/validation/test JSONL files suitable
for `rap_scratch.py train-tokenizer` without duplicating a huge `all.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow.parquet as pq

from rap_song_data.scratch.common import PRIVATE_RESEARCH_POLICY, command_record, hash_text, utc_now, write_json
from rap_song_data.scratch.corpus import (
    content_flags,
    normalize_artist,
    normalized_lyrics_key,
    split_for_artist,
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    title_columns: tuple[str, ...] = ("title", "song", "name", "track_name")
    artist_columns: tuple[str, ...] = ("artist", "artist_name", "artists")
    lyrics_columns: tuple[str, ...] = ("lyrics", "text")
    year_columns: tuple[str, ...] = ("year", "release_year", "date")
    hash_columns: tuple[str, ...] = ("normalized_hash", "normalized_text_sha256")
    source_family: str = "private_lyrics_lake"
    rights_partition: str = "private_unknown_rights"


DEFAULT_SOURCES = (
    SourceSpec(
        "private_exact_genius_family",
        Path("data/corpus_lake/normalized/private_lyrics_exact_dedupe/20260715_genius_family_v1/admitted_unique.parquet"),
    ),
    SourceSpec(
        "private_exact_remaining_local",
        Path("data/corpus_lake/normalized/private_lyrics_exact_dedupe/20260715_remaining_local_v1/admitted_unique.parquet"),
    ),
    SourceSpec(
        "common_crawl_lyrics_initial",
        Path("data/corpus_lake/raw/common_crawl_lyrics/20260715_commoncrawl_lyrics_v1/admitted_web_unique.parquet"),
        source_family="common_crawl",
    ),
    SourceSpec(
        "common_crawl_lyrics_serious",
        Path("data/corpus_lake/raw/common_crawl_lyrics/20260715_commoncrawl_lyrics_serious_20260715_190814/admitted_web_unique.parquet"),
        source_family="common_crawl",
    ),
    SourceSpec(
        "common_crawl_songmeanings",
        Path("data/corpus_lake/raw/common_crawl_lyrics/20260715_commoncrawl_songmeanings_20260715_192922/admitted_web_unique.parquet"),
        source_family="common_crawl",
    ),
    SourceSpec(
        "common_crawl_lyrics_backfill",
        Path("data/corpus_lake/raw/common_crawl_lyrics/20260715_commoncrawl_lyrics_backfill_20260715_174002/admitted_web_unique.parquet"),
        source_family="common_crawl",
    ),
    SourceSpec(
        "common_crawl_wet_probe",
        Path("data/corpus_lake/raw/common_crawl_wet_lyrics/20260716_commoncrawl_wet_lyrics_v1_20260715_223722/admitted_wet_unique.parquet"),
        source_family="common_crawl_wet",
    ),
    SourceSpec(
        "hf_lyrics_midi_extracted",
        Path("data/corpus_lake/normalized/hf_lyrics_midi_extracted/20260716_lyrics_midi_full_20260715_223056/admitted_unique.parquet"),
        source_family="huggingface",
    ),
    SourceSpec(
        "hf_asigalov61_clean_songs_overflow",
        Path(
            "C:/Users/kingj/rapSongData_overflow_review/huggingface_lyrics/"
            "asigalov61__clean-songs-lyrics-dataset/"
            "20260716_hf_clean_songs_overflow_v1_20260716_0000/data"
        ),
        title_columns=("song", "title"),
        source_family="huggingface",
    ),
)


def iter_parquet_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.parquet"))


def choose(row: dict[str, Any], columns: Iterable[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return value
    return ""


def init_seen_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_hashes("
        "normalized_hash TEXT PRIMARY KEY, record_id TEXT NOT NULL, source_id TEXT NOT NULL)"
    )
    return conn


def write_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def output_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def row_to_training_record(
    row: dict[str, Any],
    *,
    spec: SourceSpec,
    record_index: int,
    seed: int,
    min_chars: int,
    min_lines: int,
) -> tuple[dict[str, Any] | None, str]:
    lyrics = str(choose(row, spec.lyrics_columns) or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(lyrics) < min_chars:
        return None, "too_short"
    line_count = len([line for line in lyrics.splitlines() if line.strip()])
    if line_count < min_lines:
        return None, "too_few_lines"
    title = str(choose(row, spec.title_columns) or "").strip() or "<UNTITLED>"
    artist_raw = str(choose(row, spec.artist_columns) or "").strip() or "unknown-artist"
    artist_clean = normalize_artist(artist_raw)
    source_id = str(row.get("source_id") or spec.source_id)
    source_item_id = str(row.get("source_item_id") or row.get("source_row") or row.get("id") or record_index)
    normalized_hash = str(choose(row, spec.hash_columns) or "")
    if not normalized_hash:
        normalized_hash = hash_text(normalized_lyrics_key(lyrics), digest_size=32)
    year_value = choose(row, spec.year_columns)
    year = str(year_value).strip() if year_value is not None and str(year_value).strip() else "<YEAR_UNKNOWN>"
    record_id = str(row.get("record_id") or hash_text(f"{source_id}:{source_item_id}:{normalized_hash}", digest_size=24))
    flags = content_flags(lyrics)
    split = str(row.get("split") or split_for_artist(artist_clean, seed=seed))
    return (
        {
            "schema_version": 2,
            "profile": "scratch-private-lyric-lake-v1",
            "split": split,
            "record_id": record_id,
            "title": title,
            "year": year,
            "lyrics": lyrics,
            "content_flags": flags,
            "line_count": line_count,
            "artist_clean": artist_clean,
            "artist_hash": hash_text(artist_clean),
            "source_id": source_id,
            "source_family": str(row.get("source_family") or spec.source_family),
            "source_item_id": source_item_id,
            "source_path": str(row.get("source_path") or spec.path),
            "rights_partition": str(row.get("rights_partition") or spec.rights_partition),
            "normalized_hash": normalized_hash,
            "removal_key": f"{source_id}:{record_id}",
            **PRIVATE_RESEARCH_POLICY,
        },
        "",
    )


def parse_source(value: str) -> SourceSpec:
    parts = value.split("|")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--source must be source_id|path[|title_cols|artist_cols|lyrics_cols]")
    title_cols = tuple(parts[2].split(",")) if len(parts) > 2 and parts[2] else SourceSpec("", Path()).title_columns
    artist_cols = tuple(parts[3].split(",")) if len(parts) > 3 and parts[3] else SourceSpec("", Path()).artist_columns
    lyrics_cols = tuple(parts[4].split(",")) if len(parts) > 4 and parts[4] else SourceSpec("", Path()).lyrics_columns
    return SourceSpec(parts[0], Path(parts[1]), title_columns=title_cols, artist_columns=artist_cols, lyrics_columns=lyrics_cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=parse_source, action="append", default=[])
    parser.add_argument(
        "--extra-source",
        type=parse_source,
        action="append",
        default=[],
        help="Append source_id|path[|title_cols|artist_cols|lyrics_cols] to the default source set.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "corpus_manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"), flush=True)
        return 0
    if args.force:
        for pattern in ("*.jsonl", "*.partial", "*.sqlite*", "corpus_manifest.json"):
            for path in output_dir.glob(pattern):
                path.unlink()

    started_at = utc_now()
    started = time.monotonic()
    sources: tuple[SourceSpec, ...] = (
        tuple(args.source) if args.source else DEFAULT_SOURCES + tuple(args.extra_source)
    )
    db_path = output_dir / "exact_dedupe.sqlite3"
    conn = init_seen_db(db_path)
    handles = {
        "train": (output_dir / "train.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
        "validation": (output_dir / "validation.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
        "test": (output_dir / "test.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
        "rejected": (output_dir / "rejected.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
        "duplicates": (output_dir / "duplicates.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
    }
    counters: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    record_index = 0
    try:
        for spec in sources:
            paths = list(iter_parquet_paths(spec.path))
            if not paths:
                counters[f"missing_source:{spec.source_id}"] += 1
                print(f"[materialize] missing source_id={spec.source_id} path={spec.path}", flush=True)
                continue
            for parquet_path in paths:
                parquet = pq.ParquetFile(parquet_path)
                print(
                    f"[materialize] source_started source_id={spec.source_id} path={parquet_path} rows={parquet.metadata.num_rows}",
                    flush=True,
                )
                for batch in parquet.iter_batches(batch_size=args.batch_size):
                    for row in batch.to_pylist():
                        if args.limit is not None and counters["input_records"] >= args.limit:
                            break
                        counters["input_records"] += 1
                        record_index += 1
                        materialized, reason = row_to_training_record(
                            row,
                            spec=spec,
                            record_index=record_index,
                            seed=args.seed,
                            min_chars=args.min_chars,
                            min_lines=args.min_lines,
                        )
                        if materialized is None:
                            counters[f"rejected_{reason}"] += 1
                            write_jsonl(
                                handles["rejected"],
                                {"source_id": spec.source_id, "source_path": str(parquet_path), "reason": reason},
                            )
                            continue
                        normalized_hash = materialized["normalized_hash"]
                        existing = conn.execute(
                            "SELECT record_id, source_id FROM seen_hashes WHERE normalized_hash = ?",
                            (normalized_hash,),
                        ).fetchone()
                        if existing:
                            counters["duplicate_exact"] += 1
                            write_jsonl(
                                handles["duplicates"],
                                {
                                    "record_id": materialized["record_id"],
                                    "source_id": materialized["source_id"],
                                    "representative_record_id": existing[0],
                                    "representative_source_id": existing[1],
                                    "duplicate_type": "exact",
                                },
                            )
                            continue
                        conn.execute(
                            "INSERT INTO seen_hashes VALUES (?, ?, ?)",
                            (normalized_hash, materialized["record_id"], materialized["source_id"]),
                        )
                        split = materialized["split"]
                        if split not in ("train", "validation", "test"):
                            split = "train"
                            materialized["split"] = split
                        write_jsonl(handles[split], materialized)
                        counters["retained"] += 1
                        counters[f"split_{split}"] += 1
                        by_source[materialized["source_id"]] += 1
                        if materialized.get("content_flags"):
                            counters["explicit_flagged"] += 1
                        if counters["input_records"] % args.progress_every == 0:
                            conn.commit()
                            for handle in handles.values():
                                handle.flush()
                            print(
                                f"[materialize] progress input={counters['input_records']} retained={counters['retained']} "
                                f"duplicates={counters['duplicate_exact']} rejected={sum(v for k, v in counters.items() if k.startswith('rejected_'))}",
                                flush=True,
                            )
                    if args.limit is not None and counters["input_records"] >= args.limit:
                        break
                conn.commit()
                if args.limit is not None and counters["input_records"] >= args.limit:
                    break
            if args.limit is not None and counters["input_records"] >= args.limit:
                break
    finally:
        conn.commit()
        conn.close()
        for handle in handles.values():
            handle.close()

    outputs = {}
    for name in ("train", "validation", "test", "rejected", "duplicates"):
        partial = output_dir / f"{name}.jsonl.partial"
        final = output_dir / f"{name}.jsonl"
        if partial.exists():
            partial.replace(final)
        outputs[name] = output_info(final)

    manifest = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "complete",
        "operation": "materialize-private-lyric-lake-profile",
        "profile": "scratch-private-lyric-lake-v1",
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": command_record(),
        "settings": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "min_chars": args.min_chars,
            "min_lines": args.min_lines,
            "limit": args.limit,
            "dedupe": "sqlite exact normalized_hash",
            "write_all_jsonl": False,
        },
        "sources": [{"source_id": spec.source_id, "path": str(spec.path)} for spec in sources],
        "counts": dict(counters),
        "records_by_source": dict(sorted(by_source.items())),
        "outputs": outputs,
        "acceptance": {
            "minimum_retained_songs": 1,
            "retained_songs_passed": counters["retained"] >= 1,
            "token_gate_pending": True,
        },
        "note": "Private research-only profile composed from admitted lyric-lake parquet sources; raw sources were preserved.",
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
