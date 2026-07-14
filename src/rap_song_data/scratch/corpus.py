from __future__ import annotations

import argparse
import html
import json
import os
import re
import sqlite3
import struct
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import (
    PRIVATE_RESEARCH_POLICY,
    command_record,
    hash_file,
    hash_text,
    path_manifest,
    read_json,
    utc_now,
    write_json,
)


SCHEMA_OVERRIDES = {
    "title": "String",
    "tag": "String",
    "artist": "String",
    "year": "Int32",
    "views": "Int64",
    "features": "String",
    "lyrics": "String",
    "id": "Int64",
    "language_cld3": "String",
    "language_ft": "String",
    "language": "String",
}
KEEP_COLUMNS = list(SCHEMA_OVERRIDES)
ENGLISH_LABELS = {"en", "eng", "english"}
WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.I)
SECTION_RE = re.compile(
    r"^\s*[\[(]\s*(intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus|refrain)(?:\s+\d+)?(?:\s*[:\-].*?)?\s*[\])]\s*$",
    re.I,
)
SCRAPE_LINE_RE = re.compile(
    r"(?i)^(?:\d+\s+contributors?.*lyrics|you might also like|embed|share url|copy link|translations?|romanizations?)$"
)
METADATA_TITLE_RE = re.compile(r"(?i)(translation|romanization|tracklist|discography|album credits|annotations?)")
EXPLICIT_TERMS = {
    "bitch",
    "fuck",
    "fucking",
    "motherfucker",
    "nigga",
    "nigger",
    "pussy",
    "shit",
    "slut",
    "whore",
}
SECTION_TOKEN = {
    "verse": "<|verse|>",
    "chorus": "<|chorus|>",
    "hook": "<|hook|>",
    "bridge": "<|bridge|>",
    "intro": "<|intro|>",
    "outro": "<|outro|>",
    "pre-chorus": "<|chorus|>",
    "pre chorus": "<|chorus|>",
    "refrain": "<|hook|>",
}


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def normalize_artist(value: Any) -> str:
    text = unicodedata.normalize("NFKD", normalize_scalar(value).lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text).strip() or "unknown-artist"


def language_decision(row: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    labels = [
        normalize_scalar(row.get(field)).lower()
        for field in ("language", "language_cld3", "language_ft")
        if normalize_scalar(row.get(field))
    ]
    english = any(label in ENGLISH_LABELS for label in labels)
    disagreement = english and any(label not in ENGLISH_LABELS for label in labels)
    return english, disagreement, labels


def canonicalize_lyrics(value: Any) -> tuple[str, list[str], int]:
    raw = html.unescape(str(value or "")).replace("\r\n", "\n").replace("\r", "\n")
    raw = unicodedata.normalize("NFKC", raw).replace("\u200b", "").replace("\ufeff", "")
    lines: list[str] = []
    sections: list[str] = []
    lyric_line_count = 0
    for raw_line in raw.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line or SCRAPE_LINE_RE.match(line):
            continue
        match = SECTION_RE.match(line)
        if match:
            label = match.group(1).lower().replace(" ", "-")
            token = SECTION_TOKEN.get(label, SECTION_TOKEN.get(label.replace("-", " "), "<|verse|>"))
            if not lines or lines[-1] != token:
                lines.append(token)
                sections.append(token)
            continue
        line = re.sub(r"(?i)\s*\d*embed\s*$", "", line).strip()
        if line:
            lines.append(line)
            lyric_line_count += 1
    return "\n".join(lines).strip(), sections, lyric_line_count


def content_flags(text: str) -> list[str]:
    words = {word.lower() for word in WORD_RE.findall(text)}
    flags: list[str] = []
    if words & EXPLICIT_TERMS:
        flags.append("explicit_language")
    return flags


def normalized_lyrics_key(text: str) -> str:
    lines = [" ".join(WORD_RE.findall(line.lower())) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def shingle_hashes(text: str, size: int = 3) -> set[int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        features = lines + [f"{left}\n{right}" for left, right in zip(lines, lines[1:])]
        return {int(hash_text(feature, digest_size=8), 16) for feature in features}
    words = WORD_RE.findall(text.lower())
    if len(words) < size:
        return {int(hash_text(" ".join(words), digest_size=8), 16)} if words else set()
    return {
        int(hash_text(" ".join(words[index : index + size]), digest_size=8), 16)
        for index in range(len(words) - size + 1)
    }


def simhash64(features: Iterable[int]) -> int:
    weights = [0] * 64
    for feature in features:
        for bit in range(64):
            weights[bit] += 1 if feature & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def band_keys(simhash: int, shingles: set[int] | None = None) -> list[str]:
    keys = [f"h:{band}:{(simhash >> (band * 16)) & 0xFFFF:04x}" for band in range(4)]
    # A few stable shingle anchors recover high-Jaccard candidates whose short
    # texts happen to cross every coarse SimHash band. Jaccard remains the
    # deciding test, so these only broaden candidate retrieval.
    if shingles:
        keys.extend(f"s:{value:016x}" for value in sorted(shingles)[:4])
    return keys


def pack_hashes(values: set[int]) -> bytes:
    ordered = sorted(values)
    return zlib.compress(struct.pack(f"<{len(ordered)}Q", *ordered), level=1)


def unpack_hashes(value: bytes) -> set[int]:
    payload = zlib.decompress(value)
    return set(struct.unpack(f"<{len(payload) // 8}Q", payload))


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def split_for_artist(artist: str, seed: int = 20260713) -> str:
    bucket = int(hash_text(f"{seed}:{artist}", digest_size=8), 16) % 100
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def base_document(title: str, year: Any, lyrics: str) -> str:
    year_text = normalize_scalar(year)
    year_token = year_text if year_text else "<|year_unknown|>"
    return f"<|bos|><|title|>{title}\n<|year|>{year_token}\n<|lyrics|>\n{lyrics}<|eos|>"


def first_section(lyrics: str) -> tuple[str, list[str]]:
    section = "verse"
    lines: list[str] = []
    for line in lyrics.splitlines():
        stripped = line.strip()
        if stripped in SECTION_TOKEN.values():
            if lines:
                break
            section = stripped.removeprefix("<|").removesuffix("|>")
            continue
        if stripped:
            lines.append(stripped)
        if len(lines) >= 32:
            break
    return section, lines


def sft_document(title: str, year: Any, lyrics: str, flags: list[str]) -> str:
    section, lines = first_section(lyrics)
    content_token = "<|content_explicit|>" if flags else "<|content_clean|>"
    year_text = normalize_scalar(year) or "<|year_unknown|>"
    return (
        f"<|bos|><|task_generate|><|title|>{title}\n<|year|>{year_text}\n"
        f"<|section|>{section}\n<|target_lines|>{len(lines)}\n{content_token}<|lyrics|>\n"
        + "\n".join(lines)
        + "<|eos|>"
    )


class DedupeStore:
    def __init__(self, path: Path, *, threshold: float = 0.9):
        self.path = path
        self.threshold = threshold
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS exact_hashes(hash TEXT PRIMARY KEY, record_id TEXT NOT NULL, batch_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS representatives(record_id TEXT PRIMARY KEY, simhash TEXT NOT NULL, shingles BLOB NOT NULL, batch_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS bands(band_key TEXT NOT NULL, record_id TEXT NOT NULL, batch_id INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_bands_key ON bands(band_key);
            """
        )

    def rollback_from_batch(self, batch_id: int) -> None:
        for table in ("bands", "representatives", "exact_hashes"):
            self.connection.execute(f"DELETE FROM {table} WHERE batch_id >= ?", (batch_id,))
        self.connection.commit()

    def classify(self, record_id: str, key: str, *, batch_id: int) -> tuple[str, str | None, float]:
        exact = hash_text(key)
        existing = self.connection.execute("SELECT record_id FROM exact_hashes WHERE hash = ?", (exact,)).fetchone()
        if existing:
            return "exact", str(existing[0]), 1.0
        shingles = shingle_hashes(key)
        signature = simhash64(shingles)
        candidates: set[str] = set()
        for band in band_keys(signature, shingles):
            candidates.update(
                row[0]
                for row in self.connection.execute(
                    "SELECT record_id FROM bands WHERE band_key = ? LIMIT 64", (band,)
                )
            )
        best_id: str | None = None
        best_score = 0.0
        for candidate in candidates:
            row = self.connection.execute(
                "SELECT shingles FROM representatives WHERE record_id = ?", (candidate,)
            ).fetchone()
            if not row:
                continue
            score = jaccard(shingles, unpack_hashes(row[0]))
            if score > best_score:
                best_id, best_score = candidate, score
        if best_id is not None and best_score >= self.threshold:
            return "near", best_id, best_score
        self.connection.execute(
            "INSERT INTO exact_hashes(hash, record_id, batch_id) VALUES (?, ?, ?)",
            (exact, record_id, batch_id),
        )
        self.connection.execute(
            "INSERT INTO representatives(record_id, simhash, shingles, batch_id) VALUES (?, ?, ?, ?)",
            (record_id, f"{signature:016x}", pack_hashes(shingles), batch_id),
        )
        self.connection.executemany(
            "INSERT INTO bands(band_key, record_id, batch_id) VALUES (?, ?, ?)",
            [(band, record_id, batch_id) for band in band_keys(signature, shingles)],
        )
        return "unique", None, 0.0

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def build_rap_cache(source: Path, cache: Path, source_hash: str, *, force: bool = False) -> None:
    import polars as pl

    manifest_path = cache.with_suffix(".manifest.json")
    if cache.exists() and manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        if manifest.get("source_sha256") == source_hash:
            return
    cache.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        name: getattr(pl, dtype)
        for name, dtype in SCHEMA_OVERRIDES.items()
    }
    lazy = pl.scan_csv(
        source,
        schema_overrides=schema,
        infer_schema_length=1_000,
        null_values=["", "NA", "N/A", "null", "None"],
    )
    columns = [column for column in KEEP_COLUMNS if column in lazy.collect_schema().names()]
    temporary = cache.with_suffix(".partial.parquet")
    lazy.filter(pl.col("tag").str.to_lowercase() == "rap").select(columns).sink_parquet(
        temporary, compression="zstd"
    )
    os.replace(temporary, cache)
    write_json(
        manifest_path,
        {
            **PRIVATE_RESEARCH_POLICY,
            "generated_at": utc_now(),
            "source": str(source),
            "source_sha256": source_hash,
            "cache": path_manifest(cache),
        },
    )


def _open_outputs(output_dir: Path, offsets: dict[str, int]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name in ("all", "train", "validation", "test", "rejected", "duplicates"):
        path = output_dir / f"{name}.jsonl.partial"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        handle.truncate(int(offsets.get(name, 0)))
        handle.seek(0, os.SEEK_END)
        outputs[name] = handle
    return outputs


def _write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def _finalize_outputs(output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("all", "train", "validation", "test", "rejected", "duplicates"):
        partial = output_dir / f"{name}.jsonl.partial"
        final = output_dir / f"{name}.jsonl"
        os.replace(partial, final)
        paths[name] = final
    import polars as pl

    parquet = output_dir / "full_songs.parquet"
    pl.scan_ndjson(paths["all"]).sink_parquet(parquet.with_suffix(".partial.parquet"), compression="zstd")
    os.replace(parquet.with_suffix(".partial.parquet"), parquet)
    paths["parquet"] = parquet
    return paths


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the data extra to build the scratch corpus.") from exc

    started = time.monotonic()
    started_at = utc_now()
    source = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hash_file(source)
    completed_manifest = output_dir / "corpus_manifest.json"
    if completed_manifest.exists() and not args.force:
        manifest = read_json(completed_manifest)
        if manifest.get("source_sha256") == source_hash and manifest.get("status") == "complete":
            return manifest

    cache = Path(args.rap_cache)
    build_rap_cache(source, cache, source_hash, force=args.force_cache)
    state_path = output_dir / "build_state.json"
    state = read_json(state_path) if state_path.exists() and not args.force else {
        "next_batch": 0,
        "source_rows": 0,
        "counts": {},
        "offsets": {},
    }
    if args.force:
        for path in output_dir.glob("*.partial"):
            path.unlink()
        for path in output_dir.glob("*.jsonl.partial"):
            path.unlink()
        database = output_dir / "dedupe.sqlite3"
        if database.exists():
            database.unlink()
        state = {"next_batch": 0, "source_rows": 0, "counts": {}, "offsets": {}}

    counters = Counter(state.get("counts", {}))
    next_batch = int(state.get("next_batch", 0))
    store = DedupeStore(output_dir / "dedupe.sqlite3", threshold=args.near_duplicate_threshold)
    store.rollback_from_batch(next_batch)
    outputs = _open_outputs(output_dir, state.get("offsets", {}))
    parquet_file = pq.ParquetFile(cache)
    try:
        for batch_index, batch in enumerate(parquet_file.iter_batches(batch_size=args.batch_size)):
            if batch_index < next_batch:
                continue
            rows = batch.to_pylist()
            for row_index, row in enumerate(rows):
                if args.limit is not None and counters["input_rows"] >= args.limit:
                    break
                counters["input_rows"] += 1
                record_id = normalize_scalar(row.get("id")) or hash_text(
                    f"{row.get('artist')}:{row.get('title')}:{batch_index}:{row_index}"
                )
                english, disagreement, labels = language_decision(row)
                reason = None
                if not english:
                    reason = "not_english"
                title = normalize_scalar(row.get("title"))
                artist = normalize_artist(row.get("artist"))
                lyrics, sections, lyric_lines = canonicalize_lyrics(row.get("lyrics"))
                if reason is None and (not title or METADATA_TITLE_RE.search(title)):
                    reason = "metadata_or_missing_title"
                if reason is None and (len(lyrics) < args.min_chars or len(lyrics) > args.max_chars):
                    reason = "length_outlier"
                if reason is None and lyric_lines < args.min_lines:
                    reason = "too_few_lines"
                if reason is not None:
                    counters[f"rejected_{reason}"] += 1
                    _write_row(outputs["rejected"], {"record_id": record_id, "reason": reason})
                    continue
                key = normalized_lyrics_key(lyrics)
                duplicate_type, representative, similarity = store.classify(
                    record_id, key, batch_id=batch_index
                )
                if duplicate_type != "unique":
                    counters[f"duplicate_{duplicate_type}"] += 1
                    _write_row(
                        outputs["duplicates"],
                        {
                            "record_id": record_id,
                            "representative_record_id": representative,
                            "duplicate_type": duplicate_type,
                            "similarity": round(similarity, 6),
                        },
                    )
                    continue
                flags = content_flags(lyrics)
                split = split_for_artist(artist, seed=args.seed)
                output = {
                    "record_id": record_id,
                    "split": split,
                    "title": title,
                    "year": row.get("year"),
                    "artist_clean": artist,
                    "artist_hash": hash_text(artist),
                    "language_labels": labels,
                    "language_disagreement": disagreement,
                    "content_flags": flags,
                    "section_tokens": sections,
                    "line_count": lyric_lines,
                    "lyrics": lyrics,
                    **PRIVATE_RESEARCH_POLICY,
                }
                _write_row(outputs["all"], output)
                _write_row(outputs[split], output)
                counters["retained"] += 1
                counters[f"split_{split}"] += 1
                if disagreement:
                    counters["language_disagreements"] += 1
                if flags:
                    counters["explicit_flagged"] += 1
            store.commit()
            for handle in outputs.values():
                handle.flush()
                os.fsync(handle.fileno())
            state = {
                "next_batch": batch_index + 1,
                "source_rows": counters["input_rows"],
                "counts": dict(counters),
                "offsets": {name: handle.tell() for name, handle in outputs.items()},
                "updated_at": utc_now(),
            }
            write_json(state_path, state)
            if args.limit is not None and counters["input_rows"] >= args.limit:
                break
    finally:
        for handle in outputs.values():
            handle.close()
        store.close()

    paths = _finalize_outputs(output_dir)
    state_path.unlink(missing_ok=True)
    ended_at = utc_now()
    manifest = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "complete",
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": command_record(),
        "source": str(source),
        "source_sha256": source_hash,
        "rap_cache": str(cache),
        "settings": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_lines": args.min_lines,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "limit": args.limit,
        },
        "counts": dict(counters),
        "acceptance": {
            "minimum_retained_songs": args.min_retained,
            "retained_songs_passed": counters["retained"] >= args.min_retained,
            "token_gate_pending": True,
        },
        "outputs": {name: path_manifest(path) for name, path in paths.items()},
    }
    write_json(completed_manifest, manifest)
    return manifest


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=Path("data/song_lyrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--rap-cache", type=Path, default=Path("data/scratch/cache/rap_songs.parquet"))
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--min-lines", type=int, default=8)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--min-retained", type=int, default=500_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
