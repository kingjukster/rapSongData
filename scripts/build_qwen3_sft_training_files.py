"""Build Qwen3-4B SFT train/validation files from packaged sweep datasets.

Inputs are the local-only outputs from ``package_qwen3_sweep_datasets.py``.
Outputs are JSONL files with a ``training_text`` column, which is the format
expected by ``model/train_local_cuda.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
REPAIR_SYSTEM_PROMPT = (
    "Repair rap generations. Preserve the prompt intent and usable lines, "
    "but remove artifacts, weak endings, prompt drift, and formatting problems."
)
CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packaged-dir", type=Path, default=Path("data/packaged/qwen3_4b_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/qwen3_4b_sweep_sft"))
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--positive-repeat", type=int, default=1)
    parser.add_argument("--repair-repeat", type=int, default=1)
    parser.add_argument("--manual-repeat", type=int, default=2)
    parser.add_argument(
        "--include-manual-repairs",
        action="store_true",
        help="Include rows from manual_repair_queue.jsonl where manual_repaired_text is filled in.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSONL not found: {path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_int(*parts: str) -> int:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16)


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def chatml(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            continue
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks) + "\n"


def clean_message_content(content: str) -> str:
    cleaned = content
    for token in CHATML_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, "")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def row_to_training_record(
    row: dict[str, Any],
    *,
    source_bucket: str,
    repeat_index: int,
) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = clean_message_content(str(message.get("content") or ""))
        if role and content:
            normalized_messages.append({"role": role, "content": content})
    if len(normalized_messages) < 2:
        return None
    training_text = chatml(normalized_messages)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    row_id = str(metadata.get("row_id") or row.get("row_id") or stable_id(training_text))
    return {
        "id": f"{source_bucket}-{row_id}-{repeat_index}",
        "training_text": training_text,
        "metadata": {
            **metadata,
            "source_bucket": source_bucket,
            "repeat_index": repeat_index,
            "format": "qwen_chatml_training_text",
        },
    }


def manual_repair_to_training_record(row: dict[str, Any], *, repeat_index: int) -> dict[str, Any] | None:
    repaired = str(row.get("manual_repaired_text") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    raw = str(row.get("raw_generated_text") or "").strip()
    if not repaired or not prompt:
        return None
    user = (
        "Original prompt:\n"
        f"{prompt}\n\n"
        "Raw model output:\n"
        f"{raw}\n\n"
        "Return only the repaired rap lyrics."
    )
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": repaired},
    ]
    row_id = str(row.get("row_id") or stable_id(prompt, raw, repaired))
    return {
        "id": f"manual-repair-{row_id}-{repeat_index}",
        "training_text": chatml(messages),
        "metadata": {
            "source_bucket": "manual_repair",
            "row_id": row_id,
            "prompt_key": row.get("prompt_key"),
            "score": row.get("score"),
            "failure_tags": row.get("failure_tags"),
            "repeat_index": repeat_index,
            "format": "qwen_chatml_training_text",
        },
    }


def repeat_records(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    if count <= 1:
        return rows
    repeated: list[dict[str, Any]] = []
    for row in rows:
        for repeat_index in range(count):
            copy = json.loads(json.dumps(row, ensure_ascii=False))
            copy["id"] = f"{row['id']}-r{repeat_index}"
            copy["metadata"]["repeat_index"] = repeat_index
            repeated.append(copy)
    return repeated


def split_train_validation(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    if len(rows) == 1:
        return rows, list(rows)
    ratio = min(max(validation_ratio, 0.01), 0.5)
    validation_target = max(1, round(len(rows) * ratio))
    sorted_rows = sorted(rows, key=lambda row: stable_int(row["id"]))
    validation = sorted_rows[:validation_target]
    train = sorted_rows[validation_target:]
    if not train:
        train = sorted_rows[1:]
        validation = sorted_rows[:1]
    return train, validation


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    positive_rows = read_jsonl(args.packaged_dir / "positive_sft.jsonl", required=False)
    repair_rows = read_jsonl(args.packaged_dir / "repair_pairs_sft.jsonl", required=False)
    manual_rows = (
        read_jsonl(args.packaged_dir / "manual_repair_queue.jsonl", required=False)
        if args.include_manual_repairs
        else []
    )

    records: list[dict[str, Any]] = []
    counts = {
        "positive_input_rows": len(positive_rows),
        "repair_input_rows": len(repair_rows),
        "manual_input_rows": len(manual_rows),
        "positive_training_rows": 0,
        "repair_training_rows": 0,
        "manual_training_rows": 0,
    }

    positive_records: list[dict[str, Any]] = []
    for row in positive_rows:
        for repeat_index in range(max(1, args.positive_repeat)):
            record = row_to_training_record(row, source_bucket="positive_sft", repeat_index=repeat_index)
            if record:
                positive_records.append(record)
    counts["positive_training_rows"] = len(positive_records)
    records.extend(positive_records)

    repair_records: list[dict[str, Any]] = []
    for row in repair_rows:
        for repeat_index in range(max(1, args.repair_repeat)):
            record = row_to_training_record(row, source_bucket="repair_pair_sft", repeat_index=repeat_index)
            if record:
                repair_records.append(record)
    counts["repair_training_rows"] = len(repair_records)
    records.extend(repair_records)

    manual_records: list[dict[str, Any]] = []
    for row in manual_rows:
        for repeat_index in range(max(1, args.manual_repeat)):
            record = manual_repair_to_training_record(row, repeat_index=repeat_index)
            if record:
                manual_records.append(record)
    counts["manual_training_rows"] = len(manual_records)
    records.extend(manual_records)
    return records, counts


def main() -> None:
    args = parse_args()
    records, counts = build_records(args)
    if not records:
        raise SystemExit(
            "No SFT training rows were built. Run package_qwen3_sweep_datasets.py first, "
            "then check positive_sft.jsonl and repair_pairs_sft.jsonl."
        )

    train, validation = split_train_validation(records, validation_ratio=args.validation_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    manifest = {
        "base_model_scope": "Qwen/Qwen3-4B",
        "local_only": True,
        "packaged_dir": str(args.packaged_dir),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "training_text_format": "qwen_chatml",
        "validation_ratio": args.validation_ratio,
        "counts": {
            **counts,
            "total_training_rows": len(records),
            "train_rows": len(train),
            "validation_rows": len(validation),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
