#!/usr/bin/env python3
"""Fail evaluation prompt banks that overlap training prompts or themes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CHAT_USER_RE = re.compile(r"<\|im_start\|>user\n(?P<content>.*?)<\|im_end\|>", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, action="append", required=True)
    parser.add_argument("--training-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-development", type=int, default=48)
    parser.add_argument("--expected-confirmation", type=int, default=24)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", str(value or "").lower())).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_schema_errors(row: dict[str, Any], *, source_path: str, index: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    required_text = ("prompt_key", "prompt", "theme_id", "theme", "instruction_family", "prompt_family", "evaluation_split")
    for key in required_text:
        if not str(row.get(key) or "").strip():
            errors.append({"path": source_path, "row": index, "field": key, "error": "missing"})
    if row.get("target_line_count") != 12:
        errors.append({"path": source_path, "row": index, "field": "target_line_count", "error": "must_equal_12"})
    if row.get("samples_per_model") != 2:
        errors.append({"path": source_path, "row": index, "field": "samples_per_model", "error": "must_equal_2"})
    expected_key = hashlib.sha256(str(row.get("prompt") or "").encode("utf-8")).hexdigest()[:16]
    if row.get("prompt_key") != expected_key:
        errors.append({"path": source_path, "row": index, "field": "prompt_key", "error": "hash_mismatch"})
    return errors


def read_prompts(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected prompt array at {path}")
    return [row if isinstance(row, dict) else {"prompt": str(row)} for row in value]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def training_prompt(row: dict[str, Any]) -> str:
    match = CHAT_USER_RE.search(str(row.get("training_text") or ""))
    if match:
        return match.group("content")
    return str(row.get("prompt") or "")


def main() -> int:
    args = parse_args()
    evaluation: list[dict[str, Any]] = []
    prompt_input_hashes: list[dict[str, str]] = []
    schema_errors: list[dict[str, Any]] = []
    for path in args.prompt_file:
        prompt_input_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        for index, row in enumerate(read_prompts(path), start=1):
            schema_errors.extend(prompt_schema_errors(row, source_path=str(path), index=index))
            evaluation.append({**row, "source_path": str(path)})
    training: list[dict[str, Any]] = []
    training_input_hashes: list[dict[str, str]] = []
    for path in args.training_jsonl:
        training_input_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        for row in read_jsonl(path):
            training.append({**row, "source_path": str(path)})

    train_prompts = {normalize(training_prompt(row)): row for row in training}
    train_themes = {
        normalize((row.get("metadata") or {}).get("theme"))
        for row in training
        if isinstance(row.get("metadata"), dict) and (row.get("metadata") or {}).get("theme")
    }
    prompt_overlap: list[dict[str, Any]] = []
    theme_overlap: list[dict[str, Any]] = []
    eval_prompt_seen: dict[str, str] = {}
    duplicate_evaluation_prompts: list[dict[str, Any]] = []
    prompt_key_seen: dict[str, str] = {}
    duplicate_prompt_keys: list[dict[str, Any]] = []
    for row in evaluation:
        prompt = normalize(row.get("prompt"))
        theme = normalize(row.get("theme"))
        if prompt in train_prompts:
            prompt_overlap.append(
                {
                    "prompt": row.get("prompt"),
                    "evaluation_path": row["source_path"],
                    "training_path": train_prompts[prompt]["source_path"],
                }
            )
        if theme and theme in train_themes:
            theme_overlap.append({"theme": row.get("theme"), "evaluation_path": row["source_path"]})
        if prompt in eval_prompt_seen:
            duplicate_evaluation_prompts.append(
                {"prompt": row.get("prompt"), "left": eval_prompt_seen[prompt], "right": row["source_path"]}
            )
        else:
            eval_prompt_seen[prompt] = row["source_path"]
        key = str(row.get("prompt_key") or "")
        if key in prompt_key_seen:
            duplicate_prompt_keys.append({"prompt_key": key, "left": prompt_key_seen[key], "right": row["source_path"]})
        else:
            prompt_key_seen[key] = row["source_path"]

    split_counts: dict[str, int] = {}
    for row in evaluation:
        split = str(row.get("evaluation_split") or "missing")
        split_counts[split] = split_counts.get(split, 0) + 1
    expected_counts = {
        "development": args.expected_development,
        "confirmation": args.expected_confirmation,
    }
    count_mismatches = {
        split: {"expected": expected, "actual": split_counts.get(split, 0)}
        for split, expected in expected_counts.items()
        if expected >= 0 and split_counts.get(split, 0) != expected
    }

    failed = bool(
        prompt_overlap
        or theme_overlap
        or duplicate_evaluation_prompts
        or duplicate_prompt_keys
        or schema_errors
        or count_mismatches
    )
    report = {
        "status": "fail" if failed else "pass",
        "prompt_inputs": prompt_input_hashes,
        "training_inputs": training_input_hashes,
        "evaluation_prompt_count": len(evaluation),
        "evaluation_split_counts": split_counts,
        "expected_split_counts": expected_counts,
        "count_mismatches": count_mismatches,
        "training_row_count": len(training),
        "normalized_prompt_overlap_count": len(prompt_overlap),
        "theme_overlap_count": len(theme_overlap),
        "duplicate_evaluation_prompt_count": len(duplicate_evaluation_prompts),
        "duplicate_prompt_key_count": len(duplicate_prompt_keys),
        "schema_error_count": len(schema_errors),
        "prompt_overlap": prompt_overlap[:25],
        "theme_overlap": theme_overlap[:25],
        "duplicate_evaluation_prompts": duplicate_evaluation_prompts[:25],
        "duplicate_prompt_keys": duplicate_prompt_keys[:25],
        "schema_errors": schema_errors[:25],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "fail" and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
