#!/usr/bin/env python3
"""Create a deterministic family subset from a structured prompt bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_balanced_quality_gate import normalize_family


def filter_prompts(
    rows: list[dict[str, object]], families: tuple[str, ...], max_prompts: int | None = None
) -> list[dict[str, object]]:
    wanted = set(families)
    if len(wanted) != len(families):
        raise ValueError("Families must be unique.")
    selected = [row for row in rows if normalize_family(row.get("prompt_family")) in wanted]
    observed = {normalize_family(row.get("prompt_family")) for row in selected}
    missing = wanted - observed
    if missing:
        raise ValueError(f"Prompt bank is missing requested families: {sorted(missing)}")
    if max_prompts is not None:
        if max_prompts < 1:
            raise ValueError("max_prompts must be at least 1.")
        selected = selected[:max_prompts]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input prompt bank must be a JSON list.")
    selected = filter_prompts(rows, tuple(args.families), args.max_prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prompts": len(selected), "families": args.families}))


if __name__ == "__main__":
    main()
