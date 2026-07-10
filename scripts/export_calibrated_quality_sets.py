#!/usr/bin/env python3
"""Export calibrated usable/edit/reject sets from auto-judged Qwen3 candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.judge_qwen3_quality_openai import apply_manual_calibration, ranking_score  # noqa: E402


DEFAULT_JUDGE_DIR = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    parser.add_argument("--judged-jsonl", type=Path, default=None)
    parser.add_argument("--manual-results", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--edit-threshold", type=float, default=3.70)
    parser.add_argument("--reject-threshold", type=float, default=3.25)
    parser.add_argument("--max-edit", type=int, default=120)
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


def read_manual_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        rows.extend(row for row in payload if isinstance(row, dict))
    return rows


def compact_row(row: dict[str, Any], bucket: str, rank: int) -> dict[str, Any]:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    calibration = row.get("manual_calibration") if isinstance(row.get("manual_calibration"), dict) else {}
    return {
        "rank": rank,
        "selection_bucket": bucket,
        "candidate_id": row.get("candidate_id"),
        "prompt": row.get("prompt"),
        "lyrics": row.get("lyrics") or row.get("generated_text"),
        "confidence_bucket": row.get("confidence_bucket"),
        "quality_score": row.get("quality_score"),
        "heuristic_5": row.get("heuristic_5"),
        "judge_quality": judge.get("overall_quality"),
        "judge_usable_as_is": judge.get("usable_as_is"),
        "judge_issue": judge.get("main_issue"),
        "judge_reason": judge.get("short_reason"),
        "combined_quality_score": row.get("combined_quality_score"),
        "calibrated_review_score": row.get("calibrated_review_score"),
        "calibration_penalty": calibration.get("penalty"),
        "calibration_signals": calibration.get("signals") or [],
        "quality_tags": row.get("quality_tags") or [],
    }


def select_sets(rows: list[dict[str, Any]], *, edit_threshold: float, reject_threshold: float, max_edit: int) -> dict[str, list[dict[str, Any]]]:
    calibrated = [apply_manual_calibration(dict(row)) for row in rows]
    calibrated.sort(key=ranking_score, reverse=True)

    usable_source = [
        row
        for row in calibrated
        if row.get("confidence_bucket") == "auto_keep"
        and row.get("judge", {}).get("usable_as_is") == "yes"
    ]
    edit_source = [
        row
        for row in calibrated
        if row.get("confidence_bucket") == "needs_review"
        and ranking_score(row) >= edit_threshold
        and row.get("judge", {}).get("overall_quality", 0) >= 4
        and row.get("judge", {}).get("usable_as_is") == "yes"
    ][:max_edit]
    reject_source = [
        row
        for row in calibrated
        if row.get("confidence_bucket") == "auto_reject" or ranking_score(row) < reject_threshold
    ]

    return {
        "usable": [compact_row(row, "usable", rank) for rank, row in enumerate(usable_source, start=1)],
        "edit": [compact_row(row, "edit", rank) for rank, row in enumerate(edit_source, start=1)],
        "reject": [compact_row(row, "reject", rank) for rank, row in enumerate(reject_source, start=1)],
    }


def manual_preference_pairs(manual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manual_rows:
        prompt = str(row.get("prompt") or "")
        if prompt:
            by_prompt[prompt].append(row)

    pairs: list[dict[str, Any]] = []
    for prompt, rows in sorted(by_prompt.items()):
        positives = [
            row
            for row in rows
            if row.get("decision") == "keep" and int(row.get("manual_rating") or 0) >= 4
        ]
        negatives = [
            row
            for row in rows
            if row.get("decision") == "drop" or int(row.get("manual_rating") or 0) <= 2
        ]
        for positive in positives:
            for negative in negatives:
                if positive.get("candidate_id") == negative.get("candidate_id"):
                    continue
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": positive.get("lyrics"),
                        "rejected": negative.get("lyrics"),
                        "metadata": {
                            "chosen_candidate_id": positive.get("candidate_id"),
                            "rejected_candidate_id": negative.get("candidate_id"),
                            "chosen_manual_rating": positive.get("manual_rating"),
                            "rejected_manual_rating": negative.get("manual_rating"),
                            "chosen_decision": positive.get("decision"),
                            "rejected_decision": negative.get("decision"),
                            "source": "manual_quality_audit_same_prompt",
                        },
                    }
                )
    return pairs


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(out)


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]], count: int = 50) -> None:
    lines = [f"# {title}", "", f"- rows: {len(rows)}", ""]
    for row in rows[:count]:
        lines.extend(
            [
                f"## {row['rank']}. {row['candidate_id']}",
                "",
                f"- bucket: `{row['selection_bucket']}` / `{row['confidence_bucket']}`",
                f"- heuristic: `{row['quality_score']}` | judge: `{row['judge_quality']}` | combined: `{row['combined_quality_score']}` | calibrated: `{row['calibrated_review_score']}`",
                f"- issue: `{row['judge_issue']}` | penalty: `{row['calibration_penalty']}` | signals: `{', '.join(row['calibration_signals']) or 'none'}`",
                f"- prompt: {row['prompt']}",
                "",
                "```text",
                str(row.get("lyrics") or "").strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_payload(sets: dict[str, list[dict[str, Any]]], pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def issue_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(Counter(str(row.get("judge_issue") or "other") for row in rows))

    return {
        "source": str(args.judged_jsonl or args.judge_dir / "judged_candidates.jsonl"),
        "edit_threshold": args.edit_threshold,
        "reject_threshold": args.reject_threshold,
        "usable_count": len(sets["usable"]),
        "edit_count": len(sets["edit"]),
        "reject_count": len(sets["reject"]),
        "manual_preference_pair_count": len(pairs),
        "issue_counts": {
            "usable": issue_counts(sets["usable"]),
            "edit": issue_counts(sets["edit"]),
            "reject": issue_counts(sets["reject"]),
        },
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Calibrated Quality Sets",
        "",
        f"- source: `{summary['source']}`",
        f"- edit_threshold: `{summary['edit_threshold']}`",
        f"- reject_threshold: `{summary['reject_threshold']}`",
        "",
        markdown_table(
            ["Set", "Rows"],
            [
                ["usable", summary["usable_count"]],
                ["edit", summary["edit_count"]],
                ["reject", summary["reject_count"]],
                ["manual preference pairs", summary["manual_preference_pair_count"]],
            ],
        ),
        "",
        "## Policy",
        "",
        "- `usable`: current `auto_keep` rows only.",
        "- `edit`: high calibrated `needs_review` rows that the judge marked usable.",
        "- `reject`: `auto_reject` rows or rows below the calibrated reject threshold.",
        "- preference pairs: same-prompt manual audit pairs only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    judged_path = args.judged_jsonl or args.judge_dir / "judged_candidates.jsonl"
    out_dir = args.out_dir or args.judge_dir / "calibrated_quality_sets"
    manual_paths = args.manual_results or [
        args.judge_dir / "manual_rank_results.json",
        args.judge_dir / "manual_rank_auto_keep_results.json",
    ]

    rows = read_jsonl(judged_path)
    sets = select_sets(
        rows,
        edit_threshold=args.edit_threshold,
        reject_threshold=args.reject_threshold,
        max_edit=args.max_edit,
    )
    pairs = manual_preference_pairs(read_manual_results(manual_paths))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in sets.items():
        write_jsonl(out_dir / f"{name}_candidates.jsonl", selected)
        write_markdown(out_dir / f"{name}_candidates.md", name.title() + " Candidates", selected)
    write_jsonl(out_dir / "manual_preference_pairs.jsonl", pairs)

    summary = summary_payload(sets, pairs, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_md(out_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
