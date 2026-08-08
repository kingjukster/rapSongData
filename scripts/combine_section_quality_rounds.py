#!/usr/bin/env python3
"""Combine section-judge rounds and apply the acceptance cap globally."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .analyze_section_quality_batch import cap_accepted, near_duplicate_rows, write_jsonl
except ImportError:
    from analyze_section_quality_batch import cap_accepted, near_duplicate_rows, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration.jsonl"))
    parser.add_argument("--round2", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_round2.jsonl"))
    parser.add_argument("--round1-summary", type=Path, default=Path("reports/section_quality_judge_gpt55_calibration_analysis.json"))
    parser.add_argument("--round2-summary", type=Path, default=Path("reports/section_quality_judge_gpt55_round2_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_combined_accepted_capped.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/section_quality_judge_gpt55_combined_analysis.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/section_quality_judge_gpt55_combined_analysis.md"))
    parser.add_argument("--accepted-per-song-cap", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.round1) + read_jsonl(args.round2)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in (args.round1_summary, args.round2_summary)]
    strict = [row for row in rows if row.get("computed_strict_pass")]
    accepted = cap_accepted(strict, args.accepted_per_song_cap)
    write_jsonl(args.output, accepted)
    near_duplicates = near_duplicate_rows(rows, 0.84)
    expected = sum(int(summary["expected_responses"]) for summary in summaries)
    costs = sum(float(summary.get("estimated_batch_cost_usd") or 0) for summary in summaries)
    family_stats: dict[str, Any] = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        group = [row for row in rows if row["family"] == family]
        family_expected = sum(int(summary["families"][family]["expected"]) for summary in summaries)
        passed = [row for row in group if row["computed_strict_pass"]]
        capped = [row for row in accepted if row["family"] == family]
        generic_failures = sum(int(row["genericness"]) > 2 for row in group)
        stats = {
            "expected": family_expected, "parsed": len(group), "strict_pass": len(passed),
            "strict_yield": round(len(passed) / family_expected, 4),
            "accepted_after_global_cap": len(capped),
            "accepted_unique_songs": len({row["song_key"] for row in capped}),
            "genericness_failure_count": generic_failures,
            "genericness_failure_rate": round(generic_failures / len(group), 4),
        }
        if family == "clean":
            safety = sum(int(row["safety"]) == 5 and "unsafe" not in row["critical_failure_flags"] for row in group)
            stats["safety_pass_count"] = safety
            stats["safety_pass_rate"] = round(safety / len(group), 4)
        family_stats[family] = stats
    mismatches = sum(row["model_strict_pass"] != row["computed_strict_pass"] for row in rows)
    recovered_truncations = sum(int(summary.get("recovered_truncated_notes") or 0) for summary in summaries)
    truncation_rate = recovered_truncations / expected if expected else 0
    gates = {
        "gate_agreement_100pct": mismatches == 0,
        "response_coverage_100pct": len(rows) == expected,
        "near_duplicate_rate_below_5pct": len(near_duplicates) / len(rows) < 0.05,
        "truncation_rate_below_1pct": truncation_rate < 0.01,
        "at_least_75_accepted_per_family": all(family_stats[family]["accepted_after_global_cap"] >= 75 for family in families),
    }
    if "technical" in family_stats:
        gates["technical_yield_at_least_15pct"] = family_stats["technical"]["strict_yield"] >= 0.15
    if "clean" in family_stats:
        gates.update({
            "clean_yield_at_least_10pct": family_stats["clean"]["strict_yield"] >= 0.10,
            "clean_safety_at_least_80pct": family_stats["clean"]["safety_pass_rate"] >= 0.80,
            "clean_genericness_failure_below_35pct": family_stats["clean"]["genericness_failure_rate"] < 0.35,
        })
    if "story" in family_stats:
        gates.update({
            "story_yield_at_least_15pct": family_stats["story"]["strict_yield"] >= 0.15,
            "story_genericness_failure_below_35pct": family_stats["story"]["genericness_failure_rate"] < 0.35,
        })
    summary = {
        "expected_responses": expected, "parsed_responses": len(rows),
        "response_coverage_rate": round(len(rows) / expected, 4),
        "estimated_total_batch_cost_usd": round(costs, 4),
        "gate_mismatches": mismatches,
        "near_duplicate_rows": len(near_duplicates),
        "near_duplicate_rate": round(len(near_duplicates) / len(rows), 4),
        "recovered_truncated_notes": recovered_truncations,
        "truncation_rate": round(truncation_rate, 4),
        "exact_duplicate_rows": len(rows) - len({row["text_hash"] for row in rows}),
        "families": family_stats, "experiment_gates": gates,
        "all_gates_pass": all(gates.values()), "accepted_output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Combined section-quality calibration", "",
        "## Decision", "", "PASS" if all(gates.values()) else "HOLD", "",
        "| Family | Parsed/expected | Strict pass | Yield | Accepted after global 2/song cap | Safety pass | Genericness fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, stats in family_stats.items():
        safety = f"{stats['safety_pass_rate']:.1%}" if "safety_pass_rate" in stats else "n/a"
        lines.append(f"| {family} | {stats['parsed']}/{stats['expected']} | {stats['strict_pass']} | {stats['strict_yield']:.1%} | {stats['accepted_after_global_cap']} | {safety} | {stats['genericness_failure_rate']:.1%} |")
    lines.extend([
        "", "## Audit", "", f"- Response coverage: {summary['response_coverage_rate']:.1%}",
        f"- Gate mismatches: {mismatches}", f"- Near-duplicate rate: {summary['near_duplicate_rate']:.1%}",
        f"- Exact duplicate rows: {summary['exact_duplicate_rows']}",
        f"- Total estimated Batch cost: ${summary['estimated_total_batch_cost_usd']:.2f}", "", "## Gates", "",
    ])
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in gates.items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
