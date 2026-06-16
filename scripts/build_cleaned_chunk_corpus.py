"""Build shorter cleaned lyric chunks for faster local QLoRA experiments.

The input is the quality-labeled cleaned corpus from ``data/cleaned``. Raw
corpus files are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TRAIN_SOURCE = Path("data/cleaned/categorized_rap_corpus_train.jsonl")
DEFAULT_VALIDATION_SOURCE = Path("data/cleaned/categorized_rap_corpus_validation.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/cleaned/chunked")
SECTION_TAG_RE = re.compile(r"^\[(VERSE|HOOK|BRIDGE|INTRO|OUTRO)\]$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--validation-source", type=Path, default=DEFAULT_VALIDATION_SOURCE)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-lines", type=int, default=16)
    parser.add_argument("--min-lines", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=360)
    parser.add_argument("--max-chunks-per-record", type=int, default=4)
    parser.add_argument("--include-hooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def stable_hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def split_train_validation(rows: list[dict[str, Any]], validation_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    ordered = sorted(rows, key=lambda record: stable_hash(str(record.get("record_id") or "")))
    if len(ordered) == 1:
        return ordered, ordered
    validation_count = max(1, int(round(len(ordered) * validation_ratio)))
    validation_count = min(validation_count, len(ordered) - 1)
    return ordered[validation_count:], ordered[:validation_count]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalized_line(line: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", line.lower()).strip()


def repeated_line_ratio(lines: list[str]) -> float:
    keys = [normalized_line(line) for line in lines if normalized_line(line)]
    if not keys:
        return 0.0
    counts = Counter(keys)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(keys)


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_label = "VERSE"
    current_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        tag_match = SECTION_TAG_RE.match(line)
        if tag_match:
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = tag_match.group(1)
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_label, current_lines))
    return sections


def chunk_lines(
    lines: list[str],
    *,
    target_lines: int,
    min_lines: int,
    max_words: int,
) -> Iterable[list[str]]:
    if not lines:
        return
    stride = target_lines
    for start in range(0, len(lines), stride):
        chunk = lines[start : start + target_lines]
        if len(chunk) < min_lines:
            continue
        while chunk and word_count("\n".join(chunk)) > max_words and len(chunk) > min_lines:
            chunk = chunk[:-1]
        if len(chunk) >= min_lines and word_count("\n".join(chunk)) <= max_words:
            yield chunk


def training_text(record: dict[str, Any], chunk: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<|task|>generate_cleaned_chunk",
            f"<|title|>{record.get('title') or ''}",
            f"<|artist|>{record.get('artist_clean') or record.get('artist') or ''}",
            f"<|rap_family|>{record.get('rap_family') or ''}",
            f"<|rap_category|>{record.get('rap_category') or ''}",
            f"<|quality_tier|>{record.get('quality_tier') or ''}",
            f"<|quality_score|>{record.get('quality_score') or 0}",
            f"<|source_record_id|>{record.get('record_id') or ''}",
            f"<|chunk_type|>{chunk['chunk_type']}",
            f"<|target_lines|>{chunk['line_count']}",
            "<|lyrics|>",
            chunk["text"],
            "<|end|>",
        ]
    )


def make_chunks(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section_label, section_lines in split_sections(str(record.get("lyrics_cleaned") or "")):
        if section_label == "HOOK" and not args.include_hooks:
            continue
        for lines in chunk_lines(
            section_lines,
            target_lines=args.target_lines,
            min_lines=args.min_lines,
            max_words=args.max_words,
        ):
            ratio = repeated_line_ratio(lines)
            if ratio > 0.45:
                continue
            text = "\n".join(lines).strip()
            chunks.append(
                {
                    "source_record_id": record.get("record_id"),
                    "title": record.get("title"),
                    "artist_clean": record.get("artist_clean"),
                    "rap_family": record.get("rap_family"),
                    "rap_category": record.get("rap_category"),
                    "quality_tier": record.get("quality_tier"),
                    "quality_score": record.get("quality_score"),
                    "chunk_index": len(chunks) + 1,
                    "chunk_type": section_label,
                    "line_count": len(lines),
                    "word_count": word_count(text),
                    "repeated_line_ratio": round(ratio, 4),
                    "text": text,
                }
            )
            if len(chunks) >= args.max_chunks_per_record:
                return chunks
    return chunks


def write_text(path: Path, chunks: list[dict[str, Any]], source_by_id: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            record = source_by_id[str(chunk["source_record_id"])]
            file.write(training_text(record, chunk))
            file.write("\n\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def build_split(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            continue
        source_by_id[record_id] = row
        chunks.extend(make_chunks(row, args))
    return chunks, source_by_id


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    words = [int(chunk["word_count"]) for chunk in chunks]
    lines = [int(chunk["line_count"]) for chunk in chunks]
    types = Counter(str(chunk["chunk_type"]) for chunk in chunks)
    tiers = Counter(str(chunk["quality_tier"]) for chunk in chunks)

    def percentile(values: list[int], p: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
        return ordered[index]

    return {
        "chunk_count": len(chunks),
        "chunk_types": dict(types),
        "quality_tiers": dict(tiers),
        "word_count": {
            "median": percentile(words, 0.50),
            "p75": percentile(words, 0.75),
            "p95": percentile(words, 0.95),
            "max": max(words, default=0),
        },
        "line_count": {
            "median": percentile(lines, 0.50),
            "p75": percentile(lines, 0.75),
            "p95": percentile(lines, 0.95),
            "max": max(lines, default=0),
        },
        "estimated_tokens": int(sum(max(chunk["word_count"] * 1.3, len(chunk["text"]) / 4.0) for chunk in chunks)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.source, args.limit)
    if args.validation_source.exists():
        train_rows = rows
        validation_rows = read_jsonl(args.validation_source, args.limit)
        split_mode = "explicit_validation_source"
    else:
        train_rows, validation_rows = split_train_validation(rows, args.validation_ratio)
        split_mode = "deterministic_record_id_hash"
    train_chunks, train_source_by_id = build_split(train_rows, args)
    validation_chunks, validation_source_by_id = build_split(validation_rows, args)

    train_txt = args.output_dir / "categorized_rap_corpus_train_chunks.txt"
    validation_txt = args.output_dir / "categorized_rap_corpus_validation_chunks.txt"
    train_jsonl = args.output_dir / "categorized_rap_corpus_train_chunks.jsonl"
    validation_jsonl = args.output_dir / "categorized_rap_corpus_validation_chunks.jsonl"
    manifest_path = args.output_dir / "cleaned_chunk_manifest.json"

    write_text(train_txt, train_chunks, train_source_by_id)
    write_text(validation_txt, validation_chunks, validation_source_by_id)
    write_jsonl(train_jsonl, train_chunks)
    write_jsonl(validation_jsonl, validation_chunks)

    manifest = {
        "generated_at": utc_now(),
        "source": str(args.source),
        "validation_source": str(args.validation_source),
        "output_dir": str(args.output_dir),
        "settings": {
            "target_lines": args.target_lines,
            "min_lines": args.min_lines,
            "max_words": args.max_words,
            "max_chunks_per_record": args.max_chunks_per_record,
            "include_hooks": args.include_hooks,
            "validation_ratio": args.validation_ratio,
            "limit": args.limit,
        },
        "split_mode": split_mode,
        "source_record_counts": {
            "input": len(rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
        },
        "files": {
            "train_txt": str(train_txt),
            "validation_txt": str(validation_txt),
            "train_jsonl": str(train_jsonl),
            "validation_jsonl": str(validation_jsonl),
        },
        "train": summarize_chunks(train_chunks),
        "validation": summarize_chunks(validation_chunks),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
