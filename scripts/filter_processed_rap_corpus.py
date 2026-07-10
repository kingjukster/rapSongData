#!/usr/bin/env python3
"""Filter processed rap rows before building SFT/preference datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SHORT_ALLOWED_SECTION_TYPES = {"intro", "adlib", "spoken"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--min-quality-score", type=float, default=0.70)
    parser.add_argument("--min-word-count", type=int, default=5)
    parser.add_argument("--dedupe-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe-keys", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_words(value: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(value or "")))


def sort_for_quality(frame):
    return frame.assign(
        _quality_sort=frame["quality_score"].fillna(0.0).astype(float),
        _word_sort=frame.get("word_count", frame["clean_bar_text"].map(count_words)).fillna(0).astype(int),
    ).sort_values(
        ["_quality_sort", "_word_sort"],
        ascending=[False, False],
        kind="mergesort",
    )


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - local dependency check
        raise SystemExit(f"pandas is required: {exc}") from exc

    frame = pd.read_parquet(args.input)
    input_rows = len(frame)
    required = {"quality_score", "clean_bar_text", "section_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["_word_count_check"] = frame["clean_bar_text"].map(count_words)
    frame["_normalized_text"] = frame["clean_bar_text"].map(normalize_text)
    frame = frame[frame["quality_score"].fillna(0.0).astype(float) >= args.min_quality_score]
    after_quality = len(frame)
    allowed_short = frame["section_type"].fillna("").str.lower().isin(SHORT_ALLOWED_SECTION_TYPES)
    frame = frame[(frame["_word_count_check"] >= args.min_word_count) | allowed_short]
    after_length = len(frame)

    duplicate_key_dropped = 0
    if args.dedupe_keys and {"song_id", "section_id", "bar_index"}.issubset(frame.columns):
        before = len(frame)
        frame = sort_for_quality(frame).drop_duplicates(["song_id", "section_id", "bar_index"], keep="first")
        duplicate_key_dropped = before - len(frame)

    duplicate_text_dropped = 0
    if args.dedupe_text:
        before = len(frame)
        frame = sort_for_quality(frame).drop_duplicates(["_normalized_text"], keep="first")
        duplicate_text_dropped = before - len(frame)

    frame = frame.drop(columns=[col for col in ["_word_count_check", "_normalized_text", "_quality_sort", "_word_sort"] if col in frame.columns])
    frame = frame.sort_values(
        [col for col in ["split", "song_id", "section_id", "bar_index"] if col in frame.columns],
        kind="mergesort",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "input_rows": input_rows,
        "after_quality_rows": after_quality,
        "after_length_rows": after_length,
        "duplicate_key_dropped": duplicate_key_dropped,
        "duplicate_text_dropped": duplicate_text_dropped,
        "output_rows": len(frame),
        "min_quality_score": args.min_quality_score,
        "min_word_count": args.min_word_count,
        "dedupe_text": args.dedupe_text,
        "dedupe_keys": args.dedupe_keys,
    }
    summary_out = args.summary_out or args.out.with_suffix(".filter_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
