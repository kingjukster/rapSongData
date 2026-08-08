"""Summarize fixed generation-eval JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(path: Path) -> dict:
    rows = read_rows(path)
    analyses = [row.get("analysis", {}) for row in rows]
    generations = [row.get("timing") or row.get("generation", {}) for row in rows]
    hook_total = sum(1 for item in analyses if item.get("hook_line_cap_ok") is not None)
    return {
        "path": str(path),
        "records": len(rows),
        "line_counts": [item.get("line_count") for item in analyses],
        "word_counts": [item.get("word_count") for item in analyses],
        "slur_counts": [item.get("slur_count") for item in analyses],
        "exact_line_match_count": sum(1 for item in analyses if item.get("exact_line_match") is True),
        "hook_line_cap_pass_count": sum(1 for item in analyses if item.get("hook_line_cap_ok") is True),
        "hook_line_cap_total": hook_total,
        "avg_repeated_line_ratio": round(
            sum(float(item.get("repeated_line_ratio") or 0.0) for item in analyses) / max(1, len(analyses)),
            4,
        ),
        "avg_tokens_per_second": round(
            sum(float(item.get("tokens_per_second") or 0.0) for item in generations) / max(1, len(generations)),
            2,
        ),
        "max_memory_allocated_gb": max(
            [float(item.get("max_memory_allocated_gb") or 0.0) for item in generations] or [0.0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize(path), indent=2))


if __name__ == "__main__":
    main()
