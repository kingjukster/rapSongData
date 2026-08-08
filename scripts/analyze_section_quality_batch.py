#!/usr/bin/env python3
"""Materialize and audit the section-level GPT-5.5 quality calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .build_section_quality_judge_batch import computed_strict_pass
    from .mine_judged_song_sections import minhash_signature, minhash_similarity
except ImportError:
    from build_section_quality_judge_batch import computed_strict_pass
    from mine_judged_song_sections import minhash_signature, minhash_similarity


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def response_json(body: dict[str, Any]) -> dict[str, Any]:
    def parse(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            prefix, separator, _ = text.partition(',"notes":')
            if not separator:
                raise
            recovered = json.loads(prefix + "}")
            required_gate_fields = {
                "section_id", "overall", "coherence", "thematic_specificity",
                "ending_strength", "natural_phrasing_cadence", "technical_rhyme",
                "genericness", "safety", "self_contained_excerpt",
                "critical_failure_flags", "strict_pass",
            }
            if not required_gate_fields.issubset(recovered):
                raise
            recovered["notes"] = "Recovered complete gate fields; model notes were truncated."
            recovered["parse_recovered_truncated_notes"] = True
            return recovered

    if isinstance(body.get("output_text"), str):
        return parse(body["output_text"])
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return parse(content["text"])
    raise ValueError("response has no JSON text")


def near_duplicate_rows(rows: list[dict[str, Any]], threshold: float) -> set[int]:
    signatures = [minhash_signature(row["text"]) for row in rows]
    duplicate_indexes: set[int] = set()
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if minhash_similarity(signatures[left], signatures[right]) >= threshold:
                duplicate_indexes.update((left, right))
    return duplicate_indexes


def cap_accepted(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    scored = sorted(rows, key=lambda row: (
        -int(row["overall"]), -int(row["coherence"]), -int(row["ending_strength"]),
        -int(row["technical_rhyme"]), -float(row.get("local_score") or 0), row["section_id"],
    ))
    for row in scored:
        key = (str(row["song_key"]), str(row["family"]))
        if counts[key] >= cap:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-output-dir", type=Path, default=Path("data/openai_batch/section_quality_judge_gpt55_calibration/outputs"))
    parser.add_argument("--source-sections", type=Path, default=Path("data/section_mining/section_mining_calibration_selected_800.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration.jsonl"))
    parser.add_argument("--accepted-output", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration_accepted_capped.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/section_quality_judge_gpt55_calibration_analysis.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/section_quality_judge_gpt55_calibration_analysis.md"))
    parser.add_argument("--errors-output", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration_errors.jsonl"))
    parser.add_argument("--accepted-per-song-cap", type=int, default=2)
    parser.add_argument("--minhash-threshold", type=float, default=0.84)
    args = parser.parse_args()

    source = {row["section_id"]: row for row in iter_jsonl([args.source_sections])}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    for batch_row in iter_jsonl(sorted(args.batch_output_dir.glob("*_output.jsonl"))):
        custom_id = str(batch_row.get("custom_id") or "")
        response = batch_row.get("response") or {}
        body = response.get("body") if isinstance(response, dict) else None
        if not isinstance(body, dict) or int(response.get("status_code") or 0) != 200:
            errors.append({"custom_id": custom_id, "error": "response_error"})
            continue
        try:
            judgment = response_json(body)
            section_id = str(judgment["section_id"])
            base = source[section_id]
            merged = {**base, **judgment, "model_strict_pass": bool(judgment["strict_pass"]), "response_id": body.get("id")}
            merged["computed_strict_pass"] = computed_strict_pass(merged)
            rows.append(merged)
            for key, value in (body.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] += value
        except Exception as exc:  # noqa: BLE001
            errors.append({"custom_id": custom_id, "error": str(exc)})
    rows.sort(key=lambda row: (row["family"], row["section_id"]))
    write_jsonl(args.output, rows)
    write_jsonl(args.errors_output, errors)

    mismatches = [row for row in rows if row["model_strict_pass"] != row["computed_strict_pass"]]
    strict = [row for row in rows if row["computed_strict_pass"]]
    accepted = cap_accepted(strict, args.accepted_per_song_cap)
    write_jsonl(args.accepted_output, accepted)
    near_duplicates = near_duplicate_rows(rows, args.minhash_threshold)
    exact_duplicates = len(rows) - len({row["text_hash"] for row in rows})

    family_stats: dict[str, Any] = {}
    expected_family_counts = Counter(row["family"] for row in source.values())
    for family in sorted(expected_family_counts):
        group = [row for row in rows if row["family"] == family]
        passed = [row for row in group if row["computed_strict_pass"]]
        capped = [row for row in accepted if row["family"] == family]
        expected = expected_family_counts[family]
        stats: dict[str, Any] = {
            "expected": expected, "parsed": len(group), "strict_pass": len(passed),
            "strict_yield": round(len(passed) / expected, 4) if expected else 0,
            "accepted_after_source_cap": len(capped),
            "accepted_unique_songs": len({row["song_key"] for row in capped}),
            "genericness_failure_count": sum(int(row["genericness"]) > 2 for row in group),
            "genericness_failure_rate": round(sum(int(row["genericness"]) > 2 for row in group) / len(group), 4) if group else 0,
            "average_scores": {field: round(sum(int(row[field]) for row in group) / len(group), 3) for field in (
                "overall", "coherence", "thematic_specificity", "ending_strength",
                "natural_phrasing_cadence", "technical_rhyme", "genericness", "safety", "self_contained_excerpt",
            )},
            "critical_flags": dict(Counter(flag for row in group for flag in row["critical_failure_flags"] if flag != "none").most_common()),
        }
        if family == "clean":
            safety_pass = sum(int(row["safety"]) == 5 and "unsafe" not in row["critical_failure_flags"] for row in group)
            stats["safety_pass_count"] = safety_pass
            stats["safety_pass_rate"] = round(safety_pass / len(group), 4) if group else 0
        family_stats[family] = stats

    standard_cost = float(usage.get("input_tokens") or 0) / 1_000_000 * 5 + float(usage.get("output_tokens") or 0) / 1_000_000 * 30
    recovered_truncations = sum(bool(row.get("parse_recovered_truncated_notes")) for row in rows)
    truncation_rate = recovered_truncations / len(source) if source else 0
    summary = {
        "responses": len(rows), "errors": len(errors), "expected_responses": len(source),
        "response_coverage_rate": round(len(rows) / len(source), 4) if source else 0,
        "usage": dict(usage), "estimated_batch_cost_usd": round(standard_cost * 0.5, 4),
        "gate_mismatches": len(mismatches), "gate_agreement_rate": round(1 - len(mismatches) / len(rows), 4) if rows else 0,
        "recovered_truncated_notes": recovered_truncations,
        "exact_duplicate_rows": exact_duplicates, "near_duplicate_rows": len(near_duplicates),
        "near_duplicate_rate": round(len(near_duplicates) / len(rows), 4) if rows else 0,
        "truncation_rate": round(truncation_rate, 4), "families": family_stats,
        "experiment_gates": {
            "gate_agreement_100pct": len(mismatches) == 0,
            "response_coverage_100pct": len(rows) == len(source),
            "near_duplicate_rate_below_5pct": len(near_duplicates) / len(rows) < 0.05 if rows else False,
            "truncation_rate_below_1pct": truncation_rate < 0.01,
            "at_least_75_accepted_per_family": all(stats["accepted_after_source_cap"] >= 75 for stats in family_stats.values()),
        },
        "outputs": {"judgments": str(args.output), "accepted_capped": str(args.accepted_output), "errors": str(args.errors_output)},
    }
    if "technical" in family_stats:
        summary["experiment_gates"]["technical_yield_at_least_15pct"] = family_stats["technical"]["strict_yield"] >= 0.15
    if "clean" in family_stats:
        summary["experiment_gates"].update({
            "clean_yield_at_least_10pct": family_stats["clean"]["strict_yield"] >= 0.10,
            "clean_safety_at_least_80pct": family_stats["clean"].get("safety_pass_rate", 0) >= 0.80,
            "clean_genericness_failure_below_35pct": family_stats["clean"]["genericness_failure_rate"] < 0.35,
        })
    if "story" in family_stats:
        summary["experiment_gates"].update({
            "story_yield_at_least_15pct": family_stats["story"]["strict_yield"] >= 0.15,
            "story_genericness_failure_below_35pct": family_stats["story"]["genericness_failure_rate"] < 0.35,
        })
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    decision = "PASS: section mining is credible for a targeted campaign." if all(summary["experiment_gates"].values()) else "HOLD: one or more section-mining pilot gates failed."
    lines = [
        "# Section-quality GPT-5.5 calibration", "", f"## Decision", "", decision, "",
        "| Family | Parsed/expected | Strict pass | Yield | Accepted after 2/song cap | Safety pass | Genericness fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, stats in family_stats.items():
        safety = f"{stats['safety_pass_rate']:.1%}" if "safety_pass_rate" in stats else "n/a"
        lines.append(f"| {family} | {stats['parsed']}/{stats['expected']} | {stats['strict_pass']} | {stats['strict_yield']:.1%} | {stats['accepted_after_source_cap']} | {safety} | {stats['genericness_failure_rate']:.1%} |")
    lines.extend([
        "", "## Audit", "", f"- Gate agreement: {summary['gate_agreement_rate']:.1%} ({len(mismatches)} mismatches)",
        f"- Response coverage: {summary['response_coverage_rate']:.1%} ({len(rows)}/{len(source)} parsed)",
        f"- Near-duplicate rows: {len(near_duplicates)}/{len(rows)} ({summary['near_duplicate_rate']:.1%})",
        f"- Exact duplicate rows: {exact_duplicates}", f"- Truncation rate: {truncation_rate:.1%} ({recovered_truncations} recovered notes truncations)",
        f"- Estimated Batch cost: ${summary['estimated_batch_cost_usd']:.2f}", "",
        "## Gates", "",
    ])
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in summary["experiment_gates"].items())
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
