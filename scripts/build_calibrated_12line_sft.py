#!/usr/bin/env python3
"""Build a Qwen3 12-line SFT dataset from calibrated quality candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")

DEFAULT_SOURCE_DIR = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge/calibrated_quality_sets")
DEFAULT_OUTPUT_DIR = Path("data/training/qwen3_4b_12line_calibrated_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=0.12)
    parser.add_argument("--usable-repeat", type=int, default=2)
    parser.add_argument("--edit-repeat", type=int, default=1)
    parser.add_argument("--min-edit-score", type=float, default=3.70)
    parser.add_argument("--target-lines", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def stable_int(*parts: str) -> int:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16)


def clean_content(content: Any) -> str:
    cleaned = str(content or "")
    for token in CHATML_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def lyric_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text or "")))


def chatml(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = clean_content(message.get("role"))
        content = clean_content(message.get("content"))
        if role and content:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks) + "\n"


def score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("calibrated_review_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def row_to_record(row: dict[str, Any], *, source_bucket: str, repeat_index: int, target_lines: int) -> dict[str, Any] | None:
    prompt = clean_content(row.get("prompt"))
    lyrics = clean_content(row.get("lyrics"))
    candidate_id = str(row.get("candidate_id") or stable_id(prompt, lyrics))
    lines = lyric_lines(lyrics)
    if not prompt or not lyrics or len(lines) != target_lines:
        return None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": lyrics},
    ]
    prompt_key = stable_id(prompt)
    return {
        "id": f"{source_bucket}-{candidate_id}-{repeat_index}",
        "training_text": chatml(messages),
        "metadata": {
            "source": "qwen3_4b_12line_calibrated_quality_sets",
            "source_bucket": source_bucket,
            "candidate_id": candidate_id,
            "prompt_key": prompt_key,
            "repeat_index": repeat_index,
            "target_line_count": target_lines,
            "actual_line_count": len(lines),
            "assistant_word_count": word_count(lyrics),
            "calibrated_review_score": row.get("calibrated_review_score"),
            "combined_quality_score": row.get("combined_quality_score"),
            "confidence_bucket": row.get("confidence_bucket"),
            "judge_quality": row.get("judge_quality"),
            "judge_issue": row.get("judge_issue"),
            "selection_bucket": row.get("selection_bucket"),
            "format": "qwen_chatml_training_text",
            "license_scope": "synthetic_model_generated_local_audit",
        },
    }


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    usable_rows = read_jsonl(args.source_dir / "usable_candidates.jsonl")
    edit_rows = [
        row
        for row in read_jsonl(args.source_dir / "edit_candidates.jsonl")
        if score(row) >= args.min_edit_score
    ]

    records: list[dict[str, Any]] = []
    counts = {
        "usable_input_rows": len(usable_rows),
        "edit_input_rows": len(edit_rows),
        "usable_training_rows": 0,
        "edit_training_rows": 0,
        "skipped_rows": 0,
    }
    for row in usable_rows:
        for repeat_index in range(max(1, args.usable_repeat)):
            record = row_to_record(
                row,
                source_bucket="calibrated_usable",
                repeat_index=repeat_index,
                target_lines=args.target_lines,
            )
            if record:
                records.append(record)
                counts["usable_training_rows"] += 1
            else:
                counts["skipped_rows"] += 1
    for row in edit_rows:
        for repeat_index in range(max(1, args.edit_repeat)):
            record = row_to_record(
                row,
                source_bucket="calibrated_edit_review",
                repeat_index=repeat_index,
                target_lines=args.target_lines,
            )
            if record:
                records.append(record)
                counts["edit_training_rows"] += 1
            else:
                counts["skipped_rows"] += 1
    return records, counts


def split_by_prompt(
    records: list[dict[str, Any]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        prompt_key = str(record.get("metadata", {}).get("prompt_key") or stable_id(record["training_text"]))
        by_prompt[prompt_key].append(record)

    groups = sorted(by_prompt.items(), key=lambda item: stable_int(item[0]))
    if len(groups) == 1:
        return records, list(records)

    ratio = min(max(validation_ratio, 0.01), 0.5)
    target = max(1, round(len(records) * ratio))
    validation_groups: set[str] = set()
    validation_count = 0
    for prompt_key, group_rows in groups:
        validation_groups.add(prompt_key)
        validation_count += len(group_rows)
        if validation_count >= target:
            break

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for prompt_key, group_rows in by_prompt.items():
        if prompt_key in validation_groups:
            validation.extend(group_rows)
        else:
            train.extend(group_rows)
    if not train:
        prompt_key, moved = groups[-1]
        validation = [row for row in validation if row.get("metadata", {}).get("prompt_key") != prompt_key]
        train.extend(moved)
    return (
        sorted(train, key=lambda row: stable_int(row["id"])),
        sorted(validation, key=lambda row: stable_int(row["id"])),
    )


def validate_records(train: list[dict[str, Any]], validation: list[dict[str, Any]], target_lines: int) -> dict[str, Any]:
    all_rows = train + validation
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    line_count_errors = 0
    control_token_hits = 0
    source_counts: Counter[str] = Counter()
    for split, rows in [("train", train), ("validation", validation)]:
        for row in rows:
            metadata = row.get("metadata", {})
            prompt_splits[str(metadata.get("prompt_key"))].add(split)
            source_counts[str(metadata.get("source_bucket"))] += 1
            if int(metadata.get("actual_line_count") or 0) != target_lines:
                line_count_errors += 1
            assistant = row["training_text"].rsplit("<|im_start|>assistant", 1)[-1]
            if assistant.count("<|im_start|>") or assistant.count("<|im_end|>") != 1:
                control_token_hits += 1
    leaked_prompts = sorted(prompt for prompt, splits in prompt_splits.items() if len(splits) > 1)
    return {
        "total_rows": len(all_rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "source_bucket_counts": dict(source_counts),
        "target_line_count": target_lines,
        "line_count_errors": line_count_errors,
        "prompt_split_leak_count": len(leaked_prompts),
        "prompt_split_leak_examples": leaked_prompts[:5],
        "assistant_control_token_anomalies": control_token_hits,
    }


def write_preview(path: Path, records: list[dict[str, Any]], count: int = 20) -> None:
    lines = ["# Calibrated 12-Line SFT Preview", ""]
    for index, row in enumerate(records[:count], start=1):
        metadata = row["metadata"]
        assistant = row["training_text"].rsplit("<|im_start|>assistant\n", 1)[-1].rsplit("<|im_end|>", 1)[0]
        lines.extend(
            [
                f"## {index}. {metadata['candidate_id']}",
                "",
                f"- source_bucket: `{metadata['source_bucket']}`",
                f"- calibrated_review_score: `{metadata['calibrated_review_score']}`",
                f"- judge_issue: `{metadata['judge_issue']}`",
                "",
                "```text",
                assistant.strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    records, counts = build_records(args)
    if not records:
        raise SystemExit("No calibrated 12-line SFT records were built.")
    train, validation = split_by_prompt(records, validation_ratio=args.validation_ratio)
    audit = validate_records(train, validation, args.target_lines)
    if audit["line_count_errors"] or audit["prompt_split_leak_count"] or audit["assistant_control_token_anomalies"]:
        raise SystemExit(f"Calibrated SFT validation failed: {json.dumps(audit, indent=2)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    preview_path = args.output_dir / "preview.md"

    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    write_preview(preview_path, train + validation)

    manifest = {
        "name": "qwen3_4b_12line_calibrated_v1",
        "base_model_scope": "Qwen/Qwen3-4B",
        "source_dir": str(args.source_dir),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "preview_path": str(preview_path),
        "training_text_format": "qwen_chatml",
        "system_prompt": SYSTEM_PROMPT,
        "validation_ratio": args.validation_ratio,
        "usable_repeat": args.usable_repeat,
        "edit_repeat": args.edit_repeat,
        "min_edit_score": args.min_edit_score,
        "counts": {**counts, **audit},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
