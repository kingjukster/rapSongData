"""Normalize Lyrics-MIDI artifacts into the private lyric dedupe lane.

The raw Hugging Face artifact directory keeps the downloaded ZIP/pickle files as
the durable source of truth. This script streams lyric text from those artifacts,
dedupes exact normalized lyric fingerprints against prior admitted parquets, and
writes one private-use admitted parquet plus audit files.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from build_private_lyrics_dedupe_index import (
    ADMITTED_SCHEMA,
    SourceSpec,
    fingerprint_text,
    make_admitted_record,
    seed_hashes_from_parquet,
    stable_hash,
)


DEFAULT_ARTIFACT_DIR = Path(
    "data/corpus_lake/raw/huggingface_lyrics/"
    "asigalov61__Lyrics-MIDI-Dataset/20260715_hf_lyrics_midi_artifacts"
)
DEFAULT_OUTPUT_ROOT = Path("data/corpus_lake/normalized/hf_lyrics_midi_extracted")
DEFAULT_SNAPSHOT_ID = "20260716_lyrics_midi_streamed_v1"
SOURCE_URL = "https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset"


@dataclass(frozen=True)
class LyricCandidate:
    source_id: str
    source_path: str
    source_row: int
    title: str
    artist: str
    lyrics: str
    extra: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_title_artist_from_name(name: str) -> tuple[str, str]:
    stem = PurePosixPath(name).stem
    parts = stem.split(" --- ")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return stem.strip(), ""


def normalize_lyrics(text: Any) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def iter_pickle_candidates(path: Path, *, max_rows: int) -> Iterator[LyricCandidate]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return
    for index, (key, value) in enumerate(payload.items()):
        if max_rows and index >= max_rows:
            break
        if not isinstance(value, dict):
            continue
        lyrics = normalize_lyrics(value.get("lyrics"))
        if not lyrics:
            continue
        title, artist = parse_title_artist_from_name(str(key))
        yield LyricCandidate(
            source_id="hf_lyrics_midi_processed_pickle",
            source_path=str(path),
            source_row=index,
            title=title,
            artist=artist,
            lyrics=lyrics,
            extra={
                "artifact_kind": "processed_pickle",
                "key": str(key),
                "keywords": value.get("keywords") if isinstance(value.get("keywords"), list) else None,
            },
        )


def iter_zip_text_candidates(path: Path, *, max_text_files: int) -> Iterator[LyricCandidate]:
    count = 0
    with zipfile.ZipFile(path) as archive:
        for index, info in enumerate(archive.infolist()):
            name = info.filename
            lower = name.lower()
            if info.is_dir() or not lower.endswith(".txt"):
                continue
            if not (
                lower.startswith("midis and lyrics/deduped/")
                or lower.startswith("midis and lyrics/raw extras/")
                or lower.startswith("seeds/")
            ):
                continue
            if max_text_files and count >= max_text_files:
                break
            try:
                raw = archive.read(info).decode("utf-8", errors="replace")
            except Exception:
                raw = archive.read(info).decode("latin-1", errors="replace")
            lyrics = normalize_lyrics(raw)
            if not lyrics:
                continue
            title, artist = parse_title_artist_from_name(name)
            count += 1
            yield LyricCandidate(
                source_id="hf_lyrics_midi_zip_text",
                source_path=f"{path}::{name}",
                source_row=index,
                title=title,
                artist=artist,
                lyrics=lyrics,
                extra={
                    "artifact_kind": "zip_text",
                    "zip_file": path.name,
                    "zip_member": name,
                    "zip_member_bytes": info.file_size,
                },
            )


def iter_cleaned_subset_jsonl(path: Path, *, max_rows: int) -> Iterator[LyricCandidate]:
    member_name = "Lyrics/genius_lyrics_cleaned_matches.jsonl"
    with zipfile.ZipFile(path) as archive:
        if member_name not in archive.namelist():
            return
        with archive.open(member_name) as handle:
            for index, raw_line in enumerate(handle):
                if max_rows and index >= max_rows:
                    break
                try:
                    row = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(row, list):
                    row = next((item for item in row if isinstance(item, dict)), {})
                if not isinstance(row, dict):
                    continue
                lyrics = normalize_lyrics(
                    row.get("lyrics")
                    or row.get("lyric")
                    or row.get("text")
                    or row.get("lyrics_clean")
                    or row.get("lyrics_cleaned")
                )
                if not lyrics:
                    continue
                title = str(row.get("title") or row.get("song") or row.get("song_title") or "").strip()
                artist = str(row.get("artist") or row.get("artist_name") or "").strip()
                if not title:
                    title, parsed_artist = parse_title_artist_from_name(str(row.get("file_name") or row.get("midi") or index))
                    artist = artist or parsed_artist
                yield LyricCandidate(
                    source_id="hf_lyrics_midi_cleaned_subset_jsonl",
                    source_path=f"{path}::{member_name}",
                    source_row=index,
                    title=title,
                    artist=artist,
                    lyrics=lyrics,
                    extra={"artifact_kind": "cleaned_subset_jsonl"},
                )


def candidate_to_admitted(candidate: LyricCandidate, normalized_hash: str) -> dict[str, Any]:
    spec = SourceSpec(
        source_id=candidate.source_id,
        source_family="huggingface",
        path=Path(candidate.source_path.split("::", 1)[0]),
        format="artifact_stream",
        lyric_columns=("lyrics",),
    )
    return make_admitted_record(
        spec=spec,
        source_path=candidate.source_path,
        source_row=candidate.source_row,
        row={"title": candidate.title, "artist": candidate.artist, "lyrics": candidate.lyrics},
        normalized_hash=normalized_hash,
        lyrics=candidate.lyrics,
    )


def write_batch(writer: pq.ParquetWriter | None, parquet_path: Path, rows: list[dict[str, Any]]) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=ADMITTED_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(parquet_path, ADMITTED_SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--seed-admitted-parquet", action="append", type=Path, default=[])
    parser.add_argument("--include-pickle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-main-zip-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-cleaned-subset-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-main-zip-text-files", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max-pickle-rows", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max-cleaned-subset-rows", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--seed-batch-size", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=25_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "admitted_unique.parquet"
    rejected_path = output_dir / "rejected.jsonl"
    duplicate_csv_path = output_dir / "duplicate_examples.csv"
    summary_path = output_dir / "extraction_summary.json"
    command_path = output_dir / "command.json"
    for stale in (parquet_path, rejected_path, duplicate_csv_path, summary_path):
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

    started = time.time()
    seen_hashes: dict[str, tuple[str, str]] = {}
    seed_records_loaded = 0
    for seed in args.seed_admitted_parquet:
        print(f"[lyrics-midi] seed_started path={seed}", flush=True)
        seed_records_loaded += seed_hashes_from_parquet(
            parquet_path=seed,
            conn=None,
            seen_hashes=seen_hashes,
            batch_size=max(1, args.seed_batch_size),
        )
        print(f"[lyrics-midi] seed_completed cumulative={seed_records_loaded}", flush=True)

    streams: list[Iterator[LyricCandidate]] = []
    pickle_path = args.artifact_dir / "Lyrics_MIDI_Dataset_Processed_Corpus_CC_BY_NC_SA.pickle"
    main_zip_path = args.artifact_dir / "Lyrics-MIDI-Dataset-CC-BY-NC-SA.zip"
    subset_zip_path = args.artifact_dir / "Lyrics-MIDI-Genius-Cleaned-Subset-CC-BY-NC-SA.zip"
    if args.include_pickle and pickle_path.exists():
        streams.append(iter_pickle_candidates(pickle_path, max_rows=args.max_pickle_rows))
    if args.include_cleaned_subset_jsonl and subset_zip_path.exists():
        streams.append(iter_cleaned_subset_jsonl(subset_zip_path, max_rows=args.max_cleaned_subset_rows))
    if args.include_main_zip_text and main_zip_path.exists():
        streams.append(iter_zip_text_candidates(main_zip_path, max_text_files=args.max_main_zip_text_files))

    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    pending: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    duplicate_examples = 0
    admitted_tokens = 0
    samples: list[dict[str, Any]] = []
    duplicate_csv = duplicate_csv_path.open("w", encoding="utf-8", newline="")
    duplicate_writer = csv.DictWriter(
        duplicate_csv,
        ["duplicate_hash", "duplicate_source_id", "duplicate_source_path", "duplicate_source_row", "canonical_record_id", "canonical_source_id"],
    )
    duplicate_writer.writeheader()

    try:
        for stream in streams:
            for candidate in stream:
                counts["scanned"] += 1
                stats = by_source.setdefault(candidate.source_id, Counter())
                stats["scanned"] += 1
                lyrics = normalize_lyrics(candidate.lyrics)
                fingerprint = fingerprint_text(lyrics)
                if len(fingerprint) < 80:
                    counts["rejected_too_short"] += 1
                    stats["rejected_too_short"] += 1
                    with rejected_path.open("a", encoding="utf-8") as rejected:
                        rejected.write(
                            json.dumps(
                                {
                                    "reason": "too_short",
                                    "source_id": candidate.source_id,
                                    "source_path": candidate.source_path,
                                    "source_row": candidate.source_row,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue
                normalized_hash = stable_hash(fingerprint)
                existing = seen_hashes.get(normalized_hash)
                if existing:
                    counts["duplicates"] += 1
                    stats["duplicates"] += 1
                    if duplicate_examples < 100_000:
                        duplicate_writer.writerow(
                            {
                                "duplicate_hash": normalized_hash,
                                "duplicate_source_id": candidate.source_id,
                                "duplicate_source_path": candidate.source_path,
                                "duplicate_source_row": candidate.source_row,
                                "canonical_record_id": existing[0],
                                "canonical_source_id": existing[1],
                            }
                        )
                        duplicate_examples += 1
                    continue
                admitted = candidate_to_admitted(candidate, normalized_hash)
                seen_hashes[normalized_hash] = (admitted["record_id"], candidate.source_id)
                pending.append(admitted)
                counts["admitted"] += 1
                stats["admitted"] += 1
                admitted_tokens += int(admitted["approx_tokens"])
                if len(samples) < 10:
                    samples.append(
                        {
                            "source_id": candidate.source_id,
                            "title": candidate.title,
                            "artist": candidate.artist,
                            "chars": len(lyrics),
                            "preview": lyrics[:240],
                        }
                    )
                if len(pending) >= args.write_batch_size:
                    writer = write_batch(writer, parquet_path, pending)
                    pending.clear()
                if counts["scanned"] % args.progress_every == 0:
                    print(
                        f"[lyrics-midi] progress scanned={counts['scanned']} admitted={counts['admitted']} "
                        f"duplicates={counts['duplicates']} tokens={admitted_tokens}",
                        flush=True,
                    )
                    duplicate_csv.flush()
        if pending:
            writer = write_batch(writer, parquet_path, pending)
            pending.clear()
    finally:
        if writer is not None:
            writer.close()
        duplicate_csv.close()

    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_id": args.snapshot_id,
        "source_url": SOURCE_URL,
        "rights_partition": "private_unknown_rights",
        "license_id": "cc-by-nc-sa-4.0_package_underlying_lyrics_unverified",
        "artifact_dir": str(args.artifact_dir),
        "output_dir": str(output_dir),
        "admitted_parquet": str(parquet_path) if parquet_path.exists() else None,
        "rejected_jsonl": str(rejected_path),
        "duplicate_examples_csv": str(duplicate_csv_path),
        "seed_admitted_parquet": [str(path) for path in args.seed_admitted_parquet],
        "seed_records_loaded": seed_records_loaded,
        "counts": dict(counts),
        "sources": {source_id: dict(stats) for source_id, stats in by_source.items()},
        "admitted_approx_tokens": admitted_tokens,
        "sample_records": samples,
        "wall_seconds": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
