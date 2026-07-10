"""Split annotated Qwen3-4B sweep curation into focused local datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_kind(row: dict[str, Any]) -> str:
    return str(row.get("analysis", {}).get("output_kind") or "unknown")


def no_failure_tags(row: dict[str, Any]) -> bool:
    return not (row.get("failure_tags") or [])


def is_strict_verse_positive(row: dict[str, Any]) -> bool:
    return (
        row.get("decision_label") == "keeper"
        and output_kind(row) == "verse"
        and no_failure_tags(row)
        and not row.get("hit_token_cap")
        and row.get("finish_reason") == "eos"
    )


def is_hook_positive(row: dict[str, Any]) -> bool:
    return (
        row.get("decision_label") == "keeper"
        and output_kind(row) == "hook"
        and not row.get("hit_token_cap")
    )


def is_instruction_leakage(row: dict[str, Any]) -> bool:
    tags = set(row.get("failure_tags") or [])
    return bool(tags & {"instruction_echo", "prompt_leakage", "meta_commentary"})


def has_repair_signal(row: dict[str, Any]) -> bool:
    if row.get("decision_label") == "fixable":
        return True
    if row.get("postprocess_applied") and row.get("raw_generated_text") != row.get("generated_text"):
        return True
    return bool(set(row.get("failure_tags") or []) & {"incomplete_ending", "terminal_fragment", "line_count_miss"})


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    hook_positive = [row for row in rows if is_hook_positive(row)]
    verse_positive = [row for row in rows if is_strict_verse_positive(row)]
    repair_pairs = [row for row in rows if has_repair_signal(row)]
    hard_negative = [row for row in rows if row.get("decision_label") == "reject"]
    instruction_leakage = [row for row in rows if is_instruction_leakage(row)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "hook_positive": args.output_dir / "hook_positive.jsonl",
        "verse_positive_strict": args.output_dir / "verse_positive_strict.jsonl",
        "repair_pairs": args.output_dir / "repair_pairs.jsonl",
        "hard_negative": args.output_dir / "hard_negative.jsonl",
        "instruction_leakage": args.output_dir / "instruction_leakage.jsonl",
    }
    write_jsonl(outputs["hook_positive"], hook_positive)
    write_jsonl(outputs["verse_positive_strict"], verse_positive)
    write_jsonl(outputs["repair_pairs"], repair_pairs)
    write_jsonl(outputs["hard_negative"], hard_negative)
    write_jsonl(outputs["instruction_leakage"], instruction_leakage)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "local_only": True,
        "base_model_scope": "Qwen/Qwen3-4B",
        "counts": {
            "input_rows": len(rows),
            "hook_positive": len(hook_positive),
            "verse_positive_strict": len(verse_positive),
            "repair_pairs": len(repair_pairs),
            "hard_negative": len(hard_negative),
            "instruction_leakage": len(instruction_leakage),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (args.output_dir / "bucket_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
