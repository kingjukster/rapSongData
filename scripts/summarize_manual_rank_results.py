#!/usr/bin/env python3
"""Summarize exported manual ranking labels for quality-judge calibration."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge")
DEFAULT_INPUT = DEFAULT_REPORT_DIR / "manual_rank_results.json"
DEFAULT_OUT_JSON = DEFAULT_REPORT_DIR / "manual_rank_summary.json"
DEFAULT_OUT_MD = DEFAULT_REPORT_DIR / "manual_rank_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at row {index} in {path}")
        rows.append(row)
    return rows


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rounded(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def ordered_counts(values: list[Any]) -> dict[str, int]:
    counts = Counter("(missing)" if value in (None, "") else str(value) for value in values)
    return {key: counts[key] for key in sorted(counts, key=lambda item: (item == "(missing)", item))}


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    clean_x = [x for x, _ in pairs]
    clean_y = [y for _, y in pairs]
    mean_x = sum(clean_x) / len(clean_x)
    mean_y = sum(clean_y) / len(clean_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in clean_x))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in clean_y))
    denominator = denom_x * denom_y
    if denominator == 0:
        return None
    return numerator / denominator


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "manual_rank": row.get("manual_rank"),
        "candidate_id": row.get("candidate_id"),
        "manual_rating": row.get("manual_rating"),
        "decision": row.get("decision"),
        "judge_quality": row.get("judge_quality"),
        "judge_issue": row.get("judge_issue"),
        "heuristic_score": row.get("heuristic_score"),
        "combined_score": row.get("combined_score"),
        "notes": row.get("notes") or "",
        "prompt": row.get("prompt"),
    }


def sort_ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(row: dict[str, Any]) -> tuple[float, str]:
        rank = number(row.get("manual_rank"))
        return (rank if rank is not None else math.inf, str(row.get("candidate_id") or ""))

    return sorted(rows, key=rank_key)


def summarize(rows: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    manual_ratings = [number(row.get("manual_rating")) for row in rows]
    judge_qualities = [number(row.get("judge_quality")) for row in rows]
    heuristic_scores = [number(row.get("heuristic_score")) for row in rows]
    combined_scores = [number(row.get("combined_score")) for row in rows]

    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_issue[str(row.get("judge_issue") or "(missing)")].append(row)

    issue_breakdown: dict[str, dict[str, Any]] = {}
    for issue, issue_rows in sorted(by_issue.items(), key=lambda item: (-len(item[1]), item[0])):
        issue_manual = [number(row.get("manual_rating")) for row in issue_rows]
        issue_judge = [number(row.get("judge_quality")) for row in issue_rows]
        issue_heuristic = [number(row.get("heuristic_score")) for row in issue_rows]
        decisions = [row.get("decision") for row in issue_rows]
        keep_count = sum(1 for decision in decisions if decision == "keep")
        issue_breakdown[issue] = {
            "count": len(issue_rows),
            "avg_manual_rating": rounded(mean([value for value in issue_manual if value is not None]), 3),
            "avg_judge_quality": rounded(mean([value for value in issue_judge if value is not None]), 3),
            "avg_heuristic_score": rounded(mean([value for value in issue_heuristic if value is not None]), 4),
            "decision_counts": ordered_counts(decisions),
            "keep_rate": rounded(keep_count / len(issue_rows), 3) if issue_rows else None,
        }

    by_judge_quality: dict[str, dict[str, Any]] = {}
    grouped_by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_judge[str(row.get("judge_quality") or "(missing)")].append(row)
    for quality, quality_rows in sorted(grouped_by_judge.items()):
        decisions = [row.get("decision") for row in quality_rows]
        keep_count = sum(1 for decision in decisions if decision == "keep")
        by_judge_quality[quality] = {
            "count": len(quality_rows),
            "decision_counts": ordered_counts(decisions),
            "keep_rate": rounded(keep_count / len(quality_rows), 3),
            "avg_manual_rating": rounded(
                mean([value for value in (number(row.get("manual_rating")) for row in quality_rows) if value is not None]),
                3,
            ),
        }

    judge_disagreements: list[dict[str, Any]] = []
    for row in sort_ranked(rows):
        manual = number(row.get("manual_rating"))
        judge = number(row.get("judge_quality"))
        if manual is None or judge is None:
            continue
        if abs(manual - judge) >= 2:
            item = row_brief(row)
            item["gap"] = rounded(judge - manual, 3)
            judge_disagreements.append(item)

    ranked_rows = sort_ranked(rows)
    keepers = [
        row_brief(row)
        for row in ranked_rows
        if row.get("decision") == "keep" and (number(row.get("manual_rating")) or 0) >= 4
    ]
    negatives = [
        row_brief(row)
        for row in ranked_rows
        if row.get("decision") == "drop" or (number(row.get("manual_rating")) or 0) <= 2
    ]

    avg_manual = mean([value for value in manual_ratings if value is not None])
    avg_judge = mean([value for value in judge_qualities if value is not None])
    return {
        "total_candidates": len(rows),
        "reviewed_candidates": sum(1 for row in rows if number(row.get("manual_rating")) is not None and row.get("decision")),
        "missing_manual_rating": sum(1 for value in manual_ratings if value is None),
        "missing_decision": sum(1 for row in rows if not row.get("decision")),
        "decision_counts": ordered_counts([row.get("decision") for row in rows]),
        "manual_rating_counts": ordered_counts([row.get("manual_rating") for row in rows]),
        "judge_quality_counts": ordered_counts([row.get("judge_quality") for row in rows]),
        "judge_issue_counts": ordered_counts([row.get("judge_issue") for row in rows]),
        "averages": {
            "manual_rating": rounded(avg_manual, 3),
            "judge_quality": rounded(avg_judge, 3),
            "judge_minus_manual": rounded(avg_judge - avg_manual, 3) if avg_manual is not None and avg_judge is not None else None,
            "heuristic_score": rounded(mean([value for value in heuristic_scores if value is not None]), 4),
            "combined_score": rounded(mean([value for value in combined_scores if value is not None]), 4),
        },
        "metric_correlations_to_manual_rating": {
            "judge_quality": rounded(pearson(manual_ratings, judge_qualities), 4),
            "heuristic_score": rounded(pearson(manual_ratings, heuristic_scores), 4),
            "combined_score": rounded(pearson(manual_ratings, combined_scores), 4),
        },
        "by_judge_issue": issue_breakdown,
        "by_judge_quality": by_judge_quality,
        "top_manual_candidates": [row_brief(row) for row in ranked_rows[:top_n]],
        "manual_positive_seeds": keepers,
        "manual_negative_seeds": negatives,
        "judge_disagreements": judge_disagreements,
    }


def cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def count_line(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def recommendation_lines(summary: dict[str, Any]) -> list[str]:
    total = int(summary["total_candidates"])
    decisions = summary["decision_counts"]
    keep_count = decisions.get("keep", 0)
    drop_count = decisions.get("drop", 0)
    edit_count = decisions.get("edit", 0)
    keep_rate = keep_count / total if total else 0.0
    avg_heuristic = summary["averages"]["heuristic_score"] or 0.0
    avg_manual = summary["averages"]["manual_rating"] or 0.0

    lines: list[str] = []
    if total and keep_rate == 1.0 and avg_heuristic >= 0.72 and avg_manual >= 3.5:
        lines.append(
            "- Keep the current auto-keep rule: high heuristic plus usable judge=4 performed well on this audit."
        )
        lines.append(
            "- Do not treat judge_quality=4 alone as auto-keep; the disagreement audit showed low-heuristic judge=4 cases still need review."
        )
        lines.append(
            "- Use these rows as positive calibration seeds for the next ranker or prompt-selection pass."
        )
    elif drop_count or edit_count > keep_count:
        lines.append(
            "- Keep these candidates in the review queue; the current scores are not reliable enough for automatic acceptance."
        )
        lines.append(
            "- Use keep rows as positive seeds and drop/low-rated rows as negative calibration seeds."
        )
        lines.append(
            "- Penalize issue types with low manual averages before changing generation prompts or decoding."
        )
    else:
        lines.append("- Use this audit as calibration data before changing thresholds.")
        lines.append("- Compare it with an auto-keep or disagreement sample to avoid tuning from one biased slice.")
    return lines


def table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def build_markdown(summary: dict[str, Any], source_path: Path) -> str:
    averages = summary["averages"]
    lines = [
        "# Manual Rank Summary",
        "",
        f"Source: `{source_path.as_posix()}`",
        "",
        "## Label Completeness",
        "",
        f"- Reviewed candidates: {summary['reviewed_candidates']} / {summary['total_candidates']}",
        f"- Missing manual ratings: {summary['missing_manual_rating']}",
        f"- Missing decisions: {summary['missing_decision']}",
        f"- Decisions: {count_line(summary['decision_counts'])}",
        f"- Manual ratings: {count_line(summary['manual_rating_counts'])}",
        "",
        "## Calibration Signal",
        "",
        f"- Average manual rating: {averages['manual_rating']}",
        f"- Average judge rating: {averages['judge_quality']}",
        f"- Judge minus manual average: {averages['judge_minus_manual']}",
        f"- Average heuristic score: {averages['heuristic_score']}",
        f"- Average combined score: {averages['combined_score']}",
        f"- Correlation to manual rating: judge={summary['metric_correlations_to_manual_rating']['judge_quality']}, "
        f"heuristic={summary['metric_correlations_to_manual_rating']['heuristic_score']}, "
        f"combined={summary['metric_correlations_to_manual_rating']['combined_score']}",
        "",
        "## Judge Quality Buckets",
        "",
        table(
            ["Judge", "Count", "Keep Rate", "Avg Manual", "Decisions"],
            [
                [
                    quality,
                    bucket["count"],
                    bucket["keep_rate"],
                    bucket["avg_manual_rating"],
                    count_line(bucket["decision_counts"]),
                ]
                for quality, bucket in summary["by_judge_quality"].items()
            ],
        ),
        "",
        "## Issue Breakdown",
        "",
        table(
            ["Issue", "Count", "Avg Manual", "Avg Judge", "Keep Rate", "Decisions"],
            [
                [
                    issue,
                    bucket["count"],
                    bucket["avg_manual_rating"],
                    bucket["avg_judge_quality"],
                    bucket["keep_rate"],
                    count_line(bucket["decision_counts"]),
                ]
                for issue, bucket in summary["by_judge_issue"].items()
            ],
        ),
        "",
        "## Top Manual Candidates",
        "",
        table(
            ["Rank", "Candidate", "Rating", "Decision", "Judge", "Issue", "Heuristic", "Combined", "Notes"],
            [
                [
                    row["manual_rank"],
                    row["candidate_id"],
                    row["manual_rating"],
                    row["decision"],
                    row["judge_quality"],
                    row["judge_issue"],
                    row["heuristic_score"],
                    row["combined_score"],
                    row["notes"],
                ]
                for row in summary["top_manual_candidates"]
            ],
        ),
        "",
        "## Judge Disagreements",
        "",
    ]
    if summary["judge_disagreements"]:
        lines.append(
            table(
                ["Rank", "Candidate", "Manual", "Judge", "Gap", "Decision", "Issue", "Notes"],
                [
                    [
                        row["manual_rank"],
                        row["candidate_id"],
                        row["manual_rating"],
                        row["judge_quality"],
                        row["gap"],
                        row["decision"],
                        row["judge_issue"],
                        row["notes"],
                    ]
                    for row in summary["judge_disagreements"]
                ],
            )
        )
    else:
        lines.append("No judge/manual gaps of 2 or more points.")

    lines.extend(
        [
            "",
            "## Recommended Next Change",
            "",
            *recommendation_lines(summary),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = load_results(args.input)
    summary = summarize(rows, top_n=args.top_n)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(build_markdown(summary, args.input), encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
