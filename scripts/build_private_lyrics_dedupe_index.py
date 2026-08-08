"""Build a streaming exact-dedupe admission index for private lyric sources.

The corpus lake keeps raw snapshots untouched. This script reads selected raw
song-level sources, normalizes lyric text into a stable fingerprint, writes one
canonical admitted record per unique lyric, and records duplicate/source stats.

The first preset intentionally targets the large Genius-family mirrors because
they are high-value and highly overlapping. Follow-on presets can seed from a
previous admitted parquet so each run only admits genuinely new lyrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
import sys
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


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

REMAINING_LOCAL_SOURCES = (
    SourceSpec(
        source_id="hf_pjmixers_bigdata_pw_lyrics1m_en",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/PJMixers-Dev__bigdata-pw_Lyrics1M-en/20260715_hf_public_snapshot/train.json"),
        format="json_array",
        lyric_columns=("text", "lyrics"),
    ),
    SourceSpec(
        source_id="hf_howitzer_multilingual_lyrics_genre",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/HowitzerDeBoullion__Multi-Lingual-Lyrics-for-Genre-Classification/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("Lyrics", "lyrics"),
        title_columns=("Song", "title"),
        artist_columns=("Artist", "artist"),
        genre_columns=("Genre", "genre"),
        language_columns=("Language", "language"),
        include_glob="*.parquet",
    ),
    SourceSpec(
        source_id="hf_halaction_song_lyrics",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/halaction__song-lyrics/20260715_hf_public_snapshot"),
        format="csv_dir",
        lyric_columns=("lyrics",),
        include_glob="*.csv",
    ),
    SourceSpec(
        source_id="hf_nateraw_rap_lyrics_v1",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/nateraw__rap-lyrics-v1/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("lyrics",),
        include_glob="*.parquet",
    ),
    SourceSpec(
        source_id="hf_nateraw_rap_lyrics_v2",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/nateraw__rap-lyrics-v2/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("completion", "text"),
        include_glob="*.parquet",
    ),
    SourceSpec(
        source_id="hf_smgriffin_modern_pop_lyrics",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/smgriffin__modern-pop-lyrics/20260715_hf_public_snapshot/modern_pop_lyrics.csv"),
        format="csv",
        lyric_columns=("lyrics",),
    ),
    SourceSpec(
        source_id="hf_theelderemo_lyrics_database",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/theelderemo__lyrics-database/20260715_hf_public_snapshot/song-lyrics.csv"),
        format="csv",
        lyric_columns=("lyrics",),
        artist_columns=("artist_name", "artist"),
        genre_columns=("genres_list", "genre"),
    ),
    SourceSpec(
        source_id="hf_theodoredc_hiphop_lyrics",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/theodoredc__hiphop-lyrics/20260715_hf_public_snapshot/data"),
        format="parquet_dir",
        lyric_columns=("Lyrics", "lyrics"),
        title_columns=("Title", "title"),
        artist_columns=("Artist", "artist"),
        include_glob="*.parquet",
    ),
    SourceSpec(
        source_id="hf_vancenceho_spotify_lyrics",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/vancenceho__spotify-lyrics/20260715_hf_public_snapshot/spotify_millsongdata.csv"),
        format="csv",
        lyric_columns=("text", "lyrics"),
    ),
    SourceSpec(
        source_id="hf_vancenceho_spotify_lyrics_clean",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/vancenceho__spotify-lyrics-clean/20260715_hf_public_snapshot/lyrics_cleaned.csv"),
        format="csv",
        lyric_columns=("lyrics", "text"),
    ),
    SourceSpec(
        source_id="hf_cropinky_rap_lyrics_english_text_files",
        source_family="huggingface",
        path=Path("data/corpus_lake/raw/huggingface_lyrics/Cropinky__rap_lyrics_english/20260715_hf_public_snapshot/songs"),
        format="text_dir",
        lyric_columns=("lyrics",),
        include_glob="*.txt",
    ),
    SourceSpec(
        source_id="kaggle_bwandowando_spotify_attributes_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/bwandowando__spotify-songs-with-attributes-and-lyrics/20260715_credentialed_download/songs_with_attributes_and_lyrics.csv"),
        format="csv",
        lyric_columns=("lyrics",),
        title_columns=("name", "title"),
        artist_columns=("artists", "artist"),
    ),
    SourceSpec(
        source_id="kaggle_d3stron_english_5_genres",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/d3stron__english-music-lyrics-5-genres-500k/20260715_credentialed_download"),
        format="csv_dir",
        lyric_columns=("Lyric", "lyrics"),
        genre_columns=("genre", "Genre"),
        include_glob="cleaned_*_lyrics.csv",
    ),
    SourceSpec(
        source_id="kaggle_deepshah_artist_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/deepshah16__song-lyrics-dataset/20260715_credentialed_download/csv"),
        format="csv_dir",
        lyric_columns=("Lyric", "lyrics"),
        title_columns=("Title", "title"),
        artist_columns=("Artist", "artist"),
        include_glob="*.csv",
    ),
    SourceSpec(
        source_id="kaggle_devdope_900k_spotify",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/devdope__900k-spotify/20260715_credentialed_download/spotify_dataset.csv"),
        format="csv",
        lyric_columns=("text", "lyrics"),
        title_columns=("song", "title"),
        artist_columns=("Artist(s)", "artist"),
        genre_columns=("Genre", "genre"),
        year_columns=("Release Date", "year"),
    ),
    SourceSpec(
        source_id="kaggle_edenbd_valence_labeled_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/edenbd__150k-lyrics-labeled-with-spotify-valence/20260715_credentialed_download/labeled_lyrics_cleaned.csv"),
        format="csv",
        lyric_columns=("seq", "lyrics"),
    ),
    SourceSpec(
        source_id="kaggle_eitanbentora_chords_and_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/eitanbentora__chords-and-lyrics-dataset/20260715_credentialed_download/chords_and_lyrics.csv"),
        format="csv",
        lyric_columns=("lyrics", "chords&lyrics"),
        title_columns=("song_name", "title"),
        artist_columns=("artist_name", "artist"),
        genre_columns=("genres", "genre"),
        language_columns=("lang", "language"),
    ),
    SourceSpec(
        source_id="kaggle_evabot_spotify_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/evabot__spotify-lyrics-dataset/20260715_credentialed_download/lyrics_10k.csv"),
        format="csv",
        lyric_columns=("lyrics",),
        artist_columns=("artists", "artist"),
        genre_columns=("genres", "genre"),
    ),
    SourceSpec(
        source_id="kaggle_imuhammad_spotify_songs",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/imuhammad__audio-features-and-lyrics-of-spotify-songs/20260715_credentialed_download/spotify_songs.csv"),
        format="csv",
        lyric_columns=("lyrics",),
        title_columns=("track_name", "title"),
        artist_columns=("track_artist", "artist"),
        genre_columns=("playlist_genre", "playlist_subgenre", "genre"),
    ),
    SourceSpec(
        source_id="kaggle_juicobowley_drake_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/juicobowley__drake-lyrics/20260715_credentialed_download/drake_data.csv"),
        format="csv",
        lyric_columns=("lyrics",),
        title_columns=("lyrics_title", "title"),
    ),
    SourceSpec(
        source_id="kaggle_neisse_6_genres_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/neisse__scrapped-lyrics-from-6-genres/20260715_credentialed_download/lyrics-data.csv"),
        format="csv",
        lyric_columns=("Lyric", "lyrics"),
        title_columns=("SName", "title"),
    ),
    SourceSpec(
        source_id="kaggle_notshrirang_spotify_million_song",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/notshrirang__spotify-million-song-dataset/20260715_credentialed_download/spotify_millsongdata.csv"),
        format="csv",
        lyric_columns=("text", "lyrics"),
    ),
    SourceSpec(
        source_id="kaggle_promptcloud_taylor_swift_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/promptcloudhq__taylor-swift-song-lyrics-from-all-the-albums/20260715_credentialed_download/taylor_swift_lyrics.csv"),
        format="csv",
        lyric_columns=("lyric", "lyrics", "line"),
        title_columns=("track_title", "song", "title"),
    ),
    SourceSpec(
        source_id="kaggle_suraj520_music_dataset_lyrics",
        source_family="kaggle",
        path=Path("data/corpus_lake/raw/kaggle_private_lyrics/suraj520__music-dataset-song-information-and-lyrics/20260715_credentialed_download/songs.csv"),
        format="csv",
        lyric_columns=("Lyrics", "lyrics"),
        title_columns=("Name", "title"),
        artist_columns=("Artist", "artist"),
    ),
)

PRESETS = {
    "genius_family": GENIUS_FAMILY_SOURCES,
    "remaining_local_downloaded": REMAINING_LOCAL_SOURCES,
    "all_local_downloaded": GENIUS_FAMILY_SOURCES + REMAINING_LOCAL_SOURCES,
}

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

    if spec.format == "tsv":
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        row_offset = 0
        for chunk in pd.read_csv(
            spec.path,
            chunksize=chunksize,
            encoding_errors="replace",
            low_memory=False,
            sep="\t",
        ):
            for row in chunk.to_dict("records"):
                yield str(spec.path), row_offset, row
                row_offset += 1
        return

    if spec.format == "csv_dir":
        files = sorted(spec.path.glob(spec.include_glob or "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {spec.path}")
        for file_path in files:
            row_offset = 0
            for chunk in pd.read_csv(file_path, chunksize=chunksize, encoding_errors="replace", low_memory=False):
                for row in chunk.to_dict("records"):
                    yield str(file_path), row_offset, row
                    row_offset += 1
        return

    if spec.format == "parquet":
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        parquet = pq.ParquetFile(spec.path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=chunksize):
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
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

    if spec.format == "json_array":
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        # The local JSON snapshots are ordinary arrays. Loading one at a time is
        # acceptable on this workstation and avoids adding another dependency.
        with spec.path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array for {spec.path}")
        for row_offset, row in enumerate(rows):
            if isinstance(row, dict):
                yield str(spec.path), row_offset, row
        return

    if spec.format == "jsonl":
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        with spec.path.open("r", encoding="utf-8", errors="replace") as handle:
            for row_offset, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield str(spec.path), row_offset, row
        return

    if spec.format == "jsonl_dir":
        files = sorted(spec.path.glob(spec.include_glob or "*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found under {spec.path}")
        for file_path in files:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for row_offset, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield str(file_path), row_offset, row
        return

    if spec.format == "text_dir":
        files = sorted(path for path in spec.path.rglob(spec.include_glob or "*.txt") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No text files found under {spec.path}")
        for row_offset, file_path in enumerate(files):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            yield str(file_path), row_offset, {
                "title": file_path.stem,
                "artist": file_path.stem,
                "lyrics": text,
            }
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


def seed_hashes_from_parquet(
    *,
    parquet_path: Path,
    conn: sqlite3.Connection | None,
    seen_hashes: dict[str, tuple[str, str]],
    batch_size: int,
) -> int:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    parquet = pq.ParquetFile(parquet_path)
    loaded = 0
    next_report = max(batch_size, 250_000)
    required_columns = ["normalized_hash", "record_id", "source_id"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=required_columns):
        table = pa.Table.from_batches([batch])
        rows = table.to_pylist()
        if conn is not None:
            conn.executemany(
                """
                INSERT OR IGNORE INTO lyric_hashes
                (normalized_hash, record_id, source_id, source_path, source_row)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        row["normalized_hash"],
                        row["record_id"],
                        row["source_id"],
                        str(parquet_path),
                        -1,
                    )
                    for row in rows
                    if row.get("normalized_hash")
                ),
            )
            conn.commit()
        else:
            for row in rows:
                normalized_hash = row.get("normalized_hash")
                if normalized_hash and normalized_hash not in seen_hashes:
                    seen_hashes[normalized_hash] = (row.get("record_id") or "", row.get("source_id") or "seed")
        loaded += len(rows)
        if loaded >= next_report:
            print(f"[dedupe] seed_progress path={parquet_path} loaded_rows={loaded}", flush=True)
            next_report += max(batch_size, 250_000)
    return loaded


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Private lyrics exact dedupe report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        f"Preset: `{summary['preset']}`",
        "",
        "## Totals",
        "",
        f"- Seed records loaded: {summary.get('seed_records_loaded', 0):,}",
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
    parser.add_argument("--preset", choices=sorted(PRESETS), default="genius_family")
    parser.add_argument(
        "--source-config",
        type=Path,
        help=(
            "JSON file containing a sources array of SourceSpec-compatible objects. "
            "When provided, these sources replace the built-in preset."
        ),
    )
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
        "--seed-admitted-parquet",
        action="append",
        type=Path,
        default=[],
        help="Existing admitted_unique.parquet to load into the dedupe index before scanning new sources; repeatable.",
    )
    parser.add_argument(
        "--index-backend",
        choices=["memory", "sqlite"],
        default="memory",
        help="memory is fastest and expected to fit this workstation; sqlite is safer but much slower.",
    )
    return parser.parse_args()


def source_spec_from_mapping(row: dict[str, Any]) -> SourceSpec:
    required = ("source_id", "source_family", "path", "format", "lyric_columns")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Source config row is missing required keys {missing}: {row}")
    return SourceSpec(
        source_id=str(row["source_id"]),
        source_family=str(row["source_family"]),
        path=Path(str(row["path"])),
        format=str(row["format"]),
        lyric_columns=tuple(str(value) for value in row["lyric_columns"]),
        title_columns=tuple(str(value) for value in row.get("title_columns", SourceSpec("", "", Path(), "", ()).title_columns)),
        artist_columns=tuple(str(value) for value in row.get("artist_columns", SourceSpec("", "", Path(), "", ()).artist_columns)),
        genre_columns=tuple(str(value) for value in row.get("genre_columns", SourceSpec("", "", Path(), "", ()).genre_columns)),
        language_columns=tuple(str(value) for value in row.get("language_columns", SourceSpec("", "", Path(), "", ()).language_columns)),
        year_columns=tuple(str(value) for value in row.get("year_columns", SourceSpec("", "", Path(), "", ()).year_columns)),
        include_glob=str(row["include_glob"]) if row.get("include_glob") else None,
    )


def load_source_config(path: Path) -> tuple[SourceSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array or object with a sources array: {path}")
    return tuple(source_spec_from_mapping(row) for row in rows)


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


def main() -> int:
    args = parse_args()
    excluded_sources = set(args.exclude_source or [])
    if args.source_config:
        sources = tuple(spec for spec in load_source_config(args.source_config) if spec.source_id not in excluded_sources)
    else:
        sources = tuple(spec for spec in PRESETS[args.preset] if spec.source_id not in excluded_sources)
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
    seed_records_loaded = 0
    for seed_path in args.seed_admitted_parquet or []:
        print(f"[dedupe] seed_started path={seed_path}", flush=True)
        seed_records_loaded += seed_hashes_from_parquet(
            parquet_path=seed_path,
            conn=conn,
            seen_hashes=seen_hashes,
            batch_size=max(1, args.chunksize),
        )
        print(f"[dedupe] seed_completed path={seed_path} cumulative_seed_records={seed_records_loaded}", flush=True)
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
        "seed_admitted_parquet": [str(path) for path in args.seed_admitted_parquet or []],
        "seed_records_loaded": int(seed_records_loaded),
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
