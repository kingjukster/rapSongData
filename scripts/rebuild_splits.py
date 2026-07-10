#!/usr/bin/env python3
"""Assign grouped train/validation/test splits for the processed rap corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-key", default="song_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def stable_score(value: Any, seed: int) -> float:
    payload = f"{seed}|{value}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def choose_split(score: float, train_ratio: float, validation_ratio: float) -> str:
    if score < train_ratio:
        return "train"
    if score < train_ratio + validation_ratio:
        return "validation"
    return "test"


def main() -> int:
    args = parse_args()
    total = args.train_ratio + args.validation_ratio + args.test_ratio
    if total <= 0:
        raise SystemExit("Split ratios must sum to a positive value.")
    train_ratio = args.train_ratio / total
    validation_ratio = args.validation_ratio / total

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - local dependency check
        raise SystemExit(f"pandas is required: {exc}") from exc

    frame = pd.read_parquet(args.input)
    if args.group_key not in frame.columns:
        raise SystemExit(f"Missing group key column: {args.group_key}")
    groups = sorted(str(value) for value in frame[args.group_key].fillna("").unique())
    split_by_group = {
        group: choose_split(stable_score(group, args.seed), train_ratio, validation_ratio)
        for group in groups
    }
    frame[args.split_column] = frame[args.group_key].map(lambda value: split_by_group[str(value)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    group_counts = frame[[args.group_key, args.split_column]].drop_duplicates()[args.split_column].value_counts().to_dict()
    row_counts = frame[args.split_column].value_counts().to_dict()
    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "group_key": args.group_key,
        "split_column": args.split_column,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "group_count": len(groups),
        "group_counts": {str(key): int(value) for key, value in group_counts.items()},
        "row_count": int(len(frame)),
        "row_counts": {str(key): int(value) for key, value in row_counts.items()},
    }
    summary_out = args.summary_out or args.out.with_suffix(".split_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
