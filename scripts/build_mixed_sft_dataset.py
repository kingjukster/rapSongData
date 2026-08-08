"""Build mixed generation/mutation SFT JSONL files for local QLoRA runs."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


SYSTEM_GENERATION = "Generate original rap lyrics. Do not copy existing songs."
SYSTEM_MUTATION = "Rewrite and extend rap bars according to the requested controls. Do not copy existing songs."

DRIFT_PATTERNS = {
    "laughter": re.compile(r"\b(?:ha(?:ha)+|ahaha|hahaha|hehe|heehee|lmao|lol)\b", re.IGNORECASE),
    "dialogue": re.compile(
        r"[\"“][^\"”]{1,80}[\"”]|\b(?:said|says|told them|you ready\?|come here|okay then|alrighty|okie-dokie)\b",
        re.IGNORECASE,
    ),
    "stage_media": re.compile(
        r"\b(?:episode|sitcom|screen|godfather|debut single|album was|hour show)\b",
        re.IGNORECASE,
    ),
    "threat": re.compile(
        r"\b(?:kill|take your life|dead by morning|blast|shoot|murder|gun|pistol)\b",
        re.IGNORECASE,
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def assistant_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    chunks = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    return "\n".join(chunks)


def drift_flags(text: str) -> list[str]:
    return [name for name, pattern in DRIFT_PATTERNS.items() if pattern.search(text)]


def normalize_generation(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("generation record missing messages")
    metadata = dict(record.get("metadata") or {})
    metadata["sft_source"] = "generation"
    return {"messages": messages, "metadata": metadata}


def normalize_mutation(record: dict[str, Any]) -> dict[str, Any]:
    input_bars = [str(item) for item in record.get("input_bars") or []]
    output_bars = [str(item) for item in record.get("output_bars") or []]
    controls = dict(record.get("controls") or {})
    plan = ", ".join(str(item) for item in controls.get("mutation_plan") or [])
    user_lines = [
        "Rewrite and extend these rap bars.",
        f"Section type: {controls.get('section_type', 'unknown')}",
        f"Target output bars: {controls.get('output_bars', len(output_bars))}",
        f"Target emotion: {controls.get('target_emotion', 'unknown')}",
        f"Preserve meaning: {bool(controls.get('preserve_meaning', False))}",
    ]
    if plan:
        user_lines.append(f"Mutation plan: {plan}")
    user_lines.extend(["", "Input bars:", *input_bars])
    metadata = dict(record.get("metadata") or {})
    metadata["sft_source"] = "mutation"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MUTATION},
            {"role": "user", "content": "\n".join(user_lines).strip()},
            {"role": "assistant", "content": "\n".join(output_bars).strip()},
        ],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=Path("data/sft/rap_generation_sft.jsonl"))
    parser.add_argument("--mutation", type=Path, default=Path("data/sft/rap_mutation_sft.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/sft/rap_mixed_sft_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/sft/rap_mixed_sft_validation.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/sft/rap_mixed_sft_summary.json"))
    parser.add_argument("--validation-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--filter-drift",
        action="store_true",
        help="Drop records whose assistant lyric payload matches dialogue/laughter/stage/threat drift patterns.",
    )
    args = parser.parse_args()

    records = [normalize_generation(item) for item in read_jsonl(args.generation)]
    records.extend(normalize_mutation(item) for item in read_jsonl(args.mutation))
    unfiltered_count = len(records)
    filtered_counts: dict[str, int] = {}
    if args.filter_drift:
        kept: list[dict[str, Any]] = []
        for record in records:
            flags = drift_flags(assistant_text(record))
            if flags:
                for flag in flags:
                    filtered_counts[flag] = filtered_counts.get(flag, 0) + 1
                continue
            kept.append(record)
        records = kept

    rng = random.Random(args.seed)
    rng.shuffle(records)
    validation_count = min(max(args.validation_records, 0), len(records) // 5)
    validation = records[:validation_count]
    train = records[validation_count:]

    for split, split_records in [("validation", validation), ("train", train)]:
        for record in split_records:
            record.setdefault("metadata", {})["split"] = split

    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)

    summary = {
        "source_files": {
            "generation": str(args.generation),
            "mutation": str(args.mutation),
        },
        "outputs": {
            "train": str(args.train_output),
            "validation": str(args.validation_output),
        },
        "seed": args.seed,
        "filter_drift": bool(args.filter_drift),
        "filtering": {
            "input_records": unfiltered_count,
            "kept_records": len(records),
            "dropped_records": unfiltered_count - len(records),
            "dropped_by_flag": filtered_counts,
        },
        "records": {
            "total": len(records),
            "train": len(train),
            "validation": len(validation),
        },
        "sources": {},
    }
    for record in records:
        source = record.get("metadata", {}).get("sft_source", "unknown")
        summary["sources"][source] = summary["sources"].get(source, 0) + 1
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
