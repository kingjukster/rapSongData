"""Build Qwen chat-format SFT files from the curated rap song dataset.

The input is the model-ready song-level corpus. The output is local-only JSONL
with a ``training_text`` field accepted by ``model/train_local_cuda.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_SOURCE = Path("data/model_ready/rap_lyrics_training_dataset.parquet")
DEFAULT_OUTPUT_DIR = Path("data/training/curated_rap_songs_qwen3_sft")
SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
ARTIFACT_RE = re.compile(
    r"(lyrics taken from|lyrics from|you might also like|genius\.com|https?://|\bembed\b)",
    re.IGNORECASE,
)
BRACKET_LABEL_RE = re.compile(
    r"^\s*\[(intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus).*?\]\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--target-lines", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--max-examples", type=int, default=30000)
    parser.add_argument("--max-examples-per-family", type=int, default=3500)
    parser.add_argument("--max-examples-per-artist", type=int, default=350)
    parser.add_argument("--min-lines", type=int, default=10)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-repeated-line-ratio", type=float, default=0.35)
    return parser.parse_args()


def stable_id(*parts: Any) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def stable_int(*parts: Any) -> int:
    return int(stable_id(*parts)[:12], 16)


def read_source(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path, infer_schema_length=10_000)
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}")

    required = [
        "lyrics_model_text",
        "title",
        "artist_clean",
        "year",
        "rap_category",
        "rap_family",
        "word_count",
        "log_views",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Source dataset is missing required column(s): {missing}")
    return df


def clean_control(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split())


def lyric_lines(text: Any) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if BRACKET_LABEL_RE.match(line):
            continue
        if ARTIFACT_RE.search(line):
            return []
        lines.append(line)
    return lines


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def repeated_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines]
    normalized = [line for line in normalized if line]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(normalized)


def chunk_for_target(lines: list[str], *, target_lines: int, row_key: str, seed: int) -> list[str]:
    if len(lines) < target_lines:
        return []
    max_start = len(lines) - target_lines
    start = 0 if max_start <= 0 else stable_int(row_key, target_lines, seed) % (max_start + 1)
    return lines[start : start + target_lines]


def prompt_for(row: dict[str, Any], *, target_lines: int) -> str:
    category = clean_control(row.get("rap_category")) or "rap"
    family = clean_control(row.get("rap_family")) or "hip-hop"
    title = clean_control(row.get("title")) or "Untitled"
    year = clean_control(row.get("year")) or "unknown year"
    return (
        f"Write a {target_lines}-bar rap verse with a {family} feel and {category} style.\n"
        f"Reference title: {title}.\n"
        f"Reference era: {year}.\n"
        "Keep line breaks. Do not explain. Do not include bracket labels, URLs, or source notes."
    )


def chatml(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks) + "\n"


def build_example(
    row: dict[str, Any],
    *,
    target_lines: int,
    chunk_lines: list[str],
    row_key: str,
) -> dict[str, Any]:
    lyrics = "\n".join(chunk_lines).strip()
    prompt = prompt_for(row, target_lines=target_lines)
    source_row_id = clean_control(row.get("id")) or row_key
    example_id = stable_id("curated-qwen-sft", source_row_id, target_lines, lyrics)
    return {
        "id": example_id,
        "training_text": chatml(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": lyrics},
            ]
        ),
        "metadata": {
            "source": "curated_rap_song_dataset",
            "source_row_id": source_row_id,
            "title": clean_control(row.get("title")),
            "artist_clean": clean_control(row.get("artist_clean")),
            "year": row.get("year"),
            "rap_category": clean_control(row.get("rap_category")),
            "rap_family": clean_control(row.get("rap_family")),
            "target_lines": target_lines,
            "line_count": len(chunk_lines),
            "word_count": len(words(lyrics)),
            "format": "qwen_chatml_training_text",
        },
    }


def build_examples(df: pl.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sort_columns = [column for column in ["log_views", "word_count"] if column in df.columns]
    if sort_columns:
        df = df.sort(sort_columns, descending=[True] * len(sort_columns))

    examples: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    family_counts: defaultdict[str, int] = defaultdict(int)
    artist_counts: defaultdict[str, int] = defaultdict(int)

    for row_index, row in enumerate(df.to_dicts()):
        row_key = clean_control(row.get("id")) or stable_id(row_index, row.get("title"), row.get("artist_clean"))
        lines = lyric_lines(row.get("lyrics_model_text"))
        if len(lines) < args.min_lines:
            rejection_counts["too_few_lines"] += 1
            continue

        family = clean_control(row.get("rap_family")) or "unknown"
        artist = clean_control(row.get("artist_clean")) or "unknown"
        if family_counts[family] >= args.max_examples_per_family:
            rejection_counts["family_cap"] += 1
            continue
        if artist_counts[artist] >= args.max_examples_per_artist:
            rejection_counts["artist_cap"] += 1
            continue

        row_examples: list[dict[str, Any]] = []
        for target_lines in args.target_lines:
            chunk = chunk_for_target(lines, target_lines=target_lines, row_key=row_key, seed=args.seed)
            if not chunk:
                rejection_counts[f"too_few_lines_for_{target_lines}"] += 1
                continue
            lyric_text = "\n".join(chunk)
            if len(words(lyric_text)) < args.min_words:
                rejection_counts["too_few_words"] += 1
                continue
            if repeated_line_ratio(chunk) > args.max_repeated_line_ratio:
                rejection_counts["too_repetitive"] += 1
                continue
            row_examples.append(
                build_example(row, target_lines=target_lines, chunk_lines=chunk, row_key=row_key)
            )

        for example in row_examples:
            if len(examples) >= args.max_examples:
                rejection_counts["max_examples"] += 1
                break
            examples.append(example)
            family_counts[family] += 1
            artist_counts[artist] += 1
        if len(examples) >= args.max_examples:
            break

    stats = {
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "family_example_counts": dict(sorted(family_counts.items())),
        "artist_count": len(artist_counts),
    }
    return examples, stats


def split_train_validation(
    examples: list[dict[str, Any]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], []
    ratio = min(max(validation_ratio, 0.001), 0.5)
    validation_count = max(1, round(len(examples) * ratio))
    ordered = sorted(examples, key=lambda row: stable_int(row["id"]))
    validation = ordered[:validation_count]
    train = ordered[validation_count:]
    if not train:
        train = ordered[1:]
        validation = ordered[:1]
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def main() -> None:
    args = parse_args()
    df = read_source(args.source)
    examples, stats = build_examples(df, args)
    if not examples:
        raise SystemExit("No training examples were built from the curated dataset.")

    train, validation = split_train_validation(examples, validation_ratio=args.validation_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    task_counts = Counter(str(row["metadata"]["target_lines"]) for row in examples)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model_scope": "Qwen/Qwen3-4B",
        "local_only": True,
        "source": str(args.source),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "training_text_format": "qwen_chatml",
        "args": json_safe_args(args),
        "counts": {
            "source_rows": df.height,
            "total_examples": len(examples),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "target_line_counts": dict(sorted(task_counts.items())),
            **stats,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
