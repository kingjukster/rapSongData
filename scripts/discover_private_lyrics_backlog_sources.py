"""Discover downloaded private lyric backlog sources for exact-dedupe admission.

The downloader layer preserves many Hugging Face and Kaggle snapshots. This
helper scans those raw folders, samples candidate tabular/text files, and writes
a source-config JSON consumable by ``build_private_lyrics_dedupe_index.py``.

It is intentionally conservative: it skips manifests/readmes, known already
normalized source families, huge JSON arrays that are not stream-friendly, and
files whose sampled columns do not look lyric-like.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_RAW_ROOTS = (
    Path("data/corpus_lake/raw/huggingface_lyrics"),
    Path("data/corpus_lake/raw/kaggle_private_lyrics"),
    Path("C:/Users/kingj/rapSongData_overflow_review/huggingface_lyrics"),
)

DEFAULT_OUTPUT_CONFIG = Path(
    "C:/Users/kingj/rapSongData_overflow_review/source_configs/private_lyrics_backlog_sources.json"
)

KNOWN_NORMALIZED_MARKERS = (
    "Dr3dre__Genius-song-lyrics-cleaned",
    "theelderemo__genius-lyrics-cleaned",
    "amishshah__song_lyrics",
    "PJMixers-Dev__bigdata-pw_Lyrics1M-en",
    "HowitzerDeBoullion__Multi-Lingual-Lyrics-for-Genre-Classification",
    "halaction__song-lyrics",
    "nateraw__rap-lyrics-v1",
    "nateraw__rap-lyrics-v2",
    "smgriffin__modern-pop-lyrics",
    "theelderemo__lyrics-database",
    "theodoredc__hiphop-lyrics",
    "vancenceho__spotify-lyrics",
    "vancenceho__spotify-lyrics-clean",
    "Cropinky__rap_lyrics_english",
    "bwandowando__spotify-songs-with-attributes-and-lyrics",
    "d3stron__english-music-lyrics-5-genres-500k",
    "deepshah16__song-lyrics-dataset",
    "devdope__900k-spotify",
    "edenbd__150k-lyrics-labeled-with-spotify-valence",
    "eitanbentora__chords-and-lyrics-dataset",
    "evabot__spotify-lyrics-dataset",
    "imuhammad__audio-features-and-lyrics-of-spotify-songs",
    "juicobowley__drake-lyrics",
    "neisse__scrapped-lyrics-from-6-genres",
    "notshrirang__spotify-million-song-dataset",
    "promptcloudhq__taylor-swift-song-lyrics-from-all-the-albums",
    "suraj520__music-dataset-song-information-and-lyrics",
    "asigalov61__Lyrics-MIDI-Dataset",
    "asigalov61__clean-songs-lyrics-dataset",
)

SKIP_NAME_RE = re.compile(
    r"(manifest|license|readme|citation|dataset_info|state|metadata|config|\.md$|\.html?$)",
    re.IGNORECASE,
)

LYRIC_NAME_HINTS = (
    "lyrics",
    "lyric",
    "song_lyrics",
    "plainlyrics",
    "plain_lyrics",
    "syncedlyrics",
    "synced_lyrics",
    "text",
    "seq",
    "completion",
)
TITLE_COLUMNS = ("title", "Title", "SName", "song", "name", "track_name", "trackName")
ARTIST_COLUMNS = ("artist", "Artist", "ALink", "artist_name", "artists", "track_artist", "artistName")
GENRE_COLUMNS = ("tag", "genre", "Genre", "genres", "playlist_genre")
LANGUAGE_COLUMNS = ("language", "Language", "lang", "language_ft", "language_cld3")
YEAR_COLUMNS = ("year", "Year", "release_year", "Release Date", "date")


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    source_family: str
    path: str
    format: str
    lyric_columns: tuple[str, ...]
    title_columns: tuple[str, ...] = TITLE_COLUMNS
    artist_columns: tuple[str, ...] = ARTIST_COLUMNS
    genre_columns: tuple[str, ...] = GENRE_COLUMNS
    language_columns: tuple[str, ...] = LANGUAGE_COLUMNS
    year_columns: tuple[str, ...] = YEAR_COLUMNS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", value)[:140] or "source"


def looks_known_normalized(path: Path) -> bool:
    text = str(path).replace("\\", "/")
    return any(marker in text for marker in KNOWN_NORMALIZED_MARKERS)


def source_family_for(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "kaggle_private_lyrics" in text:
        return "kaggle"
    if "huggingface_lyrics" in text:
        return "huggingface"
    return "private_lyrics_backlog"


def source_id_for(path: Path) -> str:
    parts = [part for part in path.parts if "__" in part]
    base = parts[-1] if parts else path.parent.name
    local_context = "_".join(path.parts[-4:])
    return f"{source_family_for(path)}_backlog_{slug(base)}_{slug(local_context)}"


def sample_values_from_frame(frame: pd.DataFrame, column: str, limit: int = 200) -> list[str]:
    values: list[str] = []
    for value in frame[column].head(limit).tolist():
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def score_column(name: str, values: Iterable[str], *, min_chars: int, min_hits: int) -> tuple[int, int, int]:
    lowered = name.lower().replace("_", "")
    name_score = 0
    if "lyric" in lowered:
        name_score = 5
    elif lowered in {"text", "seq", "completion"}:
        name_score = 2
    hits = 0
    multiline = 0
    total_chars = 0
    for value in values:
        cleaned = value.strip()
        if len(cleaned) >= min_chars:
            hits += 1
            total_chars += len(cleaned)
            if "\n" in cleaned or "[" in cleaned:
                multiline += 1
    if hits < min_hits and name_score < 5:
        return (0, hits, total_chars)
    return (name_score * 1_000_000 + hits * 1_000 + multiline * 100 + min(total_chars, 999_999), hits, total_chars)


def choose_lyric_columns_from_frame(frame: pd.DataFrame, *, min_chars: int, min_hits: int) -> tuple[str, ...]:
    scored: list[tuple[int, str]] = []
    for column in frame.columns:
        lowered = str(column).lower().replace("_", "")
        if not any(hint.replace("_", "") in lowered or lowered == hint for hint in LYRIC_NAME_HINTS):
            continue
        values = sample_values_from_frame(frame, str(column))
        score, _, _ = score_column(str(column), values, min_chars=min_chars, min_hits=min_hits)
        if score:
            scored.append((score, str(column)))
    scored.sort(reverse=True)
    return tuple(column for _, column in scored[:2])


def inspect_csv(path: Path, *, sample_rows: int, min_chars: int, min_hits: int) -> SourceConfig | None:
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, nrows=sample_rows, encoding_errors="replace", low_memory=False, sep=sep)
    except Exception:
        return None
    lyrics = choose_lyric_columns_from_frame(frame, min_chars=min_chars, min_hits=min_hits)
    if not lyrics:
        return None
    return SourceConfig(
        source_id=source_id_for(path),
        source_family=source_family_for(path),
        path=str(path),
        format="tsv" if path.suffix.lower() == ".tsv" else "csv",
        lyric_columns=lyrics,
    )


def inspect_parquet(path: Path, *, sample_rows: int, min_chars: int, min_hits: int) -> SourceConfig | None:
    try:
        parquet = pq.ParquetFile(path)
        batch = next(parquet.iter_batches(batch_size=sample_rows), None)
        if batch is None:
            return None
        frame = batch.to_pandas()
    except Exception:
        return None
    lyrics = choose_lyric_columns_from_frame(frame, min_chars=min_chars, min_hits=min_hits)
    if not lyrics:
        return None
    return SourceConfig(
        source_id=source_id_for(path),
        source_family=source_family_for(path),
        path=str(path),
        format="parquet",
        lyric_columns=lyrics,
    )


def inspect_jsonl(path: Path, *, sample_rows: int, min_chars: int, min_hits: int) -> SourceConfig | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if len(rows) >= sample_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return None
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    lyrics = choose_lyric_columns_from_frame(frame, min_chars=min_chars, min_hits=min_hits)
    if not lyrics:
        return None
    return SourceConfig(
        source_id=source_id_for(path),
        source_family=source_family_for(path),
        path=str(path),
        format="jsonl",
        lyric_columns=lyrics,
    )


def inspect_text_dir(path: Path, *, min_chars: int) -> SourceConfig | None:
    try:
        files = [item for item in path.glob("*.txt") if item.is_file() and not SKIP_NAME_RE.search(item.name)]
    except Exception:
        return None
    if not files or len(files) > 250_000:
        return None
    good = 0
    for item in files[:50]:
        try:
            text = item.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(text.strip()) >= min_chars:
            good += 1
    if good == 0:
        return None
    return SourceConfig(
        source_id=source_id_for(path),
        source_family=source_family_for(path),
        path=str(path),
        format="text_dir",
        lyric_columns=("lyrics",),
    )


def iter_candidate_files(raw_roots: Iterable[Path]) -> Iterable[Path]:
    for root in raw_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if looks_known_normalized(path):
                continue
            if "_bulk_runs" in path.parts or "__pycache__" in path.parts:
                continue
            if SKIP_NAME_RE.search(path.name):
                continue
            suffix = path.suffix.lower()
            if suffix in {".csv", ".tsv", ".parquet", ".jsonl", ".ndjson"}:
                yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", action="append", type=Path, default=[])
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--sample-rows", type=int, default=2_000)
    parser.add_argument("--min-sample-chars", type=int, default=80)
    parser.add_argument("--min-sample-hits", type=int, default=3)
    parser.add_argument("--include-known-normalized", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    raw_roots = tuple(args.raw_root) if args.raw_root else DEFAULT_RAW_ROOTS
    sources: list[SourceConfig] = []
    stats: Counter[str] = Counter()

    for index, path in enumerate(iter_candidate_files(raw_roots)):
        if args.limit_files and index >= args.limit_files:
            break
        suffix = path.suffix.lower()
        config: SourceConfig | None = None
        if suffix in {".csv", ".tsv"}:
            config = inspect_csv(
                path,
                sample_rows=args.sample_rows,
                min_chars=args.min_sample_chars,
                min_hits=args.min_sample_hits,
            )
        elif suffix == ".parquet":
            config = inspect_parquet(
                path,
                sample_rows=args.sample_rows,
                min_chars=args.min_sample_chars,
                min_hits=args.min_sample_hits,
            )
        elif suffix in {".jsonl", ".ndjson"}:
            config = inspect_jsonl(
                path,
                sample_rows=args.sample_rows,
                min_chars=args.min_sample_chars,
                min_hits=args.min_sample_hits,
            )
        stats[f"scanned_{suffix or 'none'}"] += 1
        if config is None:
            stats["rejected_no_lyric_column"] += 1
            continue
        sources.append(config)
        stats["accepted"] += 1

    # Text directories are grouped per snapshot/folder to avoid thousands of tiny
    # SourceSpec entries. Do this after file-level discovery.
    for root in raw_roots:
        if not root.exists():
            continue
        for directory in root.rglob("*"):
            if not directory.is_dir() or looks_known_normalized(directory):
                continue
            if "_bulk_runs" in directory.parts:
                continue
            config = inspect_text_dir(directory, min_chars=args.min_sample_chars)
            if config is None:
                continue
            if any(existing.path == config.path for existing in sources):
                continue
            sources.append(config)
            stats["accepted_text_dir"] += 1

    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "operation": "discover-private-lyrics-backlog-sources",
        "raw_roots": [str(path) for path in raw_roots],
        "policy": {
            "rights_partition": "private_unknown_rights",
            "training_eligibility": "private_only",
            "note": "Generated from local raw lyric snapshots; underlying lyric rights remain unverified.",
        },
        "stats": dict(stats),
        "sources": [asdict(source) for source in sources],
        "wall_seconds": round(time.time() - started, 3),
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_config": str(args.output_config), "source_count": len(sources), "stats": dict(stats)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
