#!/usr/bin/env python3
"""Analyze strict family gates from a materialized song-quality calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


SCORED_DIMENSIONS = [
    "overall", "technical_rhyme", "flow_cadence", "coherence", "thematic_depth",
    "imagery", "ending_strength", "family_compliance", "cleanliness", "genericness",
]
BASE_GATE_DIMENSIONS = [
    "overall", "technical_rhyme", "flow_cadence", "coherence", "thematic_depth", "ending_strength",
]


def has_critical_failure(row: dict[str, Any]) -> bool:
    flags = set(row.get("critical_failure_flags") or [])
    return bool(flags - {"none"})


def computed_strict_pass(row: dict[str, Any]) -> bool:
    passed = (
        all(int(row.get(field) or 0) >= 4 for field in BASE_GATE_DIMENSIONS)
        and int(row.get("family_compliance") or 0) == 5
        and not has_critical_failure(row)
    )
    if row.get("judge_family") == "clean":
        passed = passed and int(row.get("cleanliness") or 0) == 5 and int(row.get("genericness") or 6) <= 2
    return passed


def family_relevant_strict_pass(row: dict[str, Any]) -> bool:
    """Apply the same gate without treating clean-only safety as a technical failure."""
    flags = set(row.get("critical_failure_flags") or []) - {"none"}
    if row.get("judge_family") == "technical":
        flags.discard("unsafe_for_clean")
    passed = (
        all(int(row.get(field) or 0) >= 4 for field in BASE_GATE_DIMENSIONS)
        and int(row.get("family_compliance") or 0) == 5
        and not flags
    )
    if row.get("judge_family") == "clean":
        passed = passed and int(row.get("cleanliness") or 0) == 5 and int(row.get("genericness") or 6) <= 2
    return passed


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"rows": len(rows), "families": {}}
    for family in ("technical", "clean"):
        group = [row for row in rows if row.get("judge_family") == family]
        model_pass = sum(bool(row.get("strict_pass")) for row in group)
        computed_pass = sum(computed_strict_pass(row) for row in group)
        mismatches = sum(bool(row.get("strict_pass")) != computed_strict_pass(row) for row in group)
        relevant_pass = sum(family_relevant_strict_pass(row) for row in group)
        dimension_stats = {}
        for field in SCORED_DIMENSIONS:
            values = [int(row[field]) for row in group]
            dimension_stats[field] = {
                "mean": round(mean(values), 3),
                "median": median(values),
                "below_4": sum(value < 4 for value in values),
                "score_counts": dict(sorted(Counter(values).items())),
            }
        gate_failures = Counter()
        for row in group:
            for field in BASE_GATE_DIMENSIONS:
                if int(row.get(field) or 0) < 4:
                    gate_failures[field] += 1
            if int(row.get("family_compliance") or 0) != 5:
                gate_failures["family_compliance_not_5"] += 1
            if has_critical_failure(row):
                gate_failures["critical_failure"] += 1
            if family == "clean":
                if int(row.get("cleanliness") or 0) != 5:
                    gate_failures["cleanliness_not_5"] += 1
                if int(row.get("genericness") or 6) > 2:
                    gate_failures["genericness_above_2"] += 1
        flags = Counter(
            flag for row in group for flag in (row.get("critical_failure_flags") or []) if flag != "none"
        )
        report["families"][family] = {
            "rows": len(group),
            "model_strict_pass": model_pass,
            "model_strict_pass_rate": round(model_pass / len(group), 4) if group else 0,
            "computed_strict_pass": computed_pass,
            "computed_strict_pass_rate": round(computed_pass / len(group), 4) if group else 0,
            "model_computed_mismatches": mismatches,
            "family_relevant_strict_pass": relevant_pass,
            "family_relevant_strict_pass_rate": round(relevant_pass / len(group), 4) if group else 0,
            "truncated_rows": sum(bool(row.get("truncated")) for row in group),
            "gate_failure_counts": dict(gate_failures.most_common()),
            "critical_failure_flags": dict(flags.most_common()),
            "dimensions": dimension_stats,
            "structural_metrics": {
                "internal_rhyme_count_mean": round(mean(float(row["internal_rhyme_count"]) for row in group), 3),
                "multisyllabic_rhyme_count_mean": round(mean(float(row["multisyllabic_rhyme_count"]) for row in group), 3),
                "repeated_ending_ratio_mean": round(mean(float(row["repeated_ending_ratio"]) for row in group), 4),
                "syllable_variance_mean": round(mean(float(row["syllable_variance"]) for row in group), 3),
                "unresolved_final_fragment_count": sum(bool(row["unresolved_final_fragment"]) for row in group),
            },
        }
    return report


def markdown(report: dict[str, Any], input_path: Path) -> str:
    lines = [
        "# GPT-5.5 song-quality calibration", "", f"Input: `{input_path}`", "",
        "## Decision", "",
        "Do not scale the strict judge to the full 50,000-song pool. Both families are far below the 20-30% strict-pass yield required for a targeted v4 dataset campaign.", "",
        "## Family results", "",
        "| Family | Rows | Model pass | Recomputed pass | Family-relevant pass | Truncated |", "|---|---:|---:|---:|---:|---:|",
    ]
    for family, stats in report["families"].items():
        lines.append(
            f"| {family} | {stats['rows']} | {stats['model_strict_pass']} ({stats['model_strict_pass_rate']:.1%}) | "
            f"{stats['computed_strict_pass']} ({stats['computed_strict_pass_rate']:.1%}) | "
            f"{stats['family_relevant_strict_pass']} ({stats['family_relevant_strict_pass_rate']:.1%}) | {stats['truncated_rows']} |"
        )
    lines.extend(["", "## Dominant gate failures", ""])
    for family, stats in report["families"].items():
        top = list(stats["gate_failure_counts"].items())[:8]
        lines.append(f"### {family.title()}")
        lines.append("")
        lines.extend(f"- {name}: {count}/{stats['rows']}" for name, count in top)
        lines.append("")
    lines.extend([
        "## Next experiment", "",
        "Manually inspect a stratified packet containing strict passes, near-passes, and clear failures from each family. This will determine whether the bottleneck is source quality, full-song scoring versus verse-level extraction, or an over-constrained rubric before spending on more judging.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reviews/song_quality_judge_gpt55_calibration_1k.jsonl"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/song_quality_judge_gpt55_calibration_1k_analysis.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/song_quality_judge_gpt55_calibration_1k_analysis.md"))
    parser.add_argument("--usage-summary", type=Path, default=Path("data/reviews/song_quality_judge_gpt55_calibration_1k_summary.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = analyze(rows)
    if args.usage_summary.exists():
        usage = json.loads(args.usage_summary.read_text(encoding="utf-8")).get("usage") or {}
        standard_cost = float(usage.get("input_tokens") or 0) / 1_000_000 * 5.0 + float(usage.get("output_tokens") or 0) / 1_000_000 * 30.0
        report["usage"] = usage
        report["estimated_standard_cost_usd"] = round(standard_cost, 4)
        report["estimated_batch_cost_usd"] = round(standard_cost * 0.5, 4)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report, args.input), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
