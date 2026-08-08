"""Analyze how much a generation sweep depends on postprocessing.

This script compares local curation judgments on raw model output versus the
postprocessed output saved by ``model/run_fixed_generation_eval.py``. It also
extracts training-data planning buckets without starting any training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rank_generation_candidates import judge_candidate


BENIGN_ACTIONS = {"normalized_whitespace", "clean_model_artifacts"}
ENDING_REPAIR_ACTIONS = {
    "drop_dangling_final_line",
    "drop_unfinished_final_line",
    "trimmed_to_requested_line_count",
    "hook_trimmed_to_cap",
}
SEVERE_NEGATIVE_TAGS = {
    "no_slur_fail",
    "dialogue_drift",
    "absurd_drift",
    "violent_derailment",
    "name_or_brand_leak",
    "hook_too_long",
    "too_short",
    "long_line",
    "repeated_lines",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-lines", type=int, default=6)
    parser.add_argument("--max-line-words", type=int, default=34)
    parser.add_argument("--top-fixable", type=int, default=200)
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


def action_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for action in row.get("postprocess_actions") or []:
        if isinstance(action, dict):
            names.append(str(action.get("action") or "unknown"))
        else:
            names.append(str(action))
    return names


def recalc_judgment(row: dict[str, Any], text: str, *, min_lines: int, max_line_words: int) -> dict[str, Any]:
    candidate = dict(row)
    candidate.pop("analysis", None)
    candidate["generated_text"] = text
    candidate["text"] = text
    return judge_candidate(candidate, min_lines=min_lines, max_line_words=max_line_words)


def compact_source(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "index",
        "batch_id",
        "prompt_index",
        "sample_index",
        "prompt",
        "seed",
        "settings",
        "environment",
        "timing",
        "effective_max_new_tokens",
    }
    return {key: row[key] for key in keep if key in row}


def enriched_row(row: dict[str, Any], raw_judgment: dict[str, Any], post_judgment: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(row.get("raw_generated_text") or row.get("raw_text") or "")
    generated_text = str(row.get("generated_text") or row.get("text") or "")
    actions = row.get("postprocess_actions") or []
    names = action_names(row)
    return {
        **compact_source(row),
        "raw_generated_text": raw_text,
        "generated_text": generated_text,
        "raw_differs_from_generated": raw_text != generated_text,
        "postprocess_applied": bool(row.get("postprocess_applied")),
        "postprocess_actions": actions,
        "postprocess_action_names": names,
        "raw_line_count": row.get("raw_line_count"),
        "postprocessed_line_count": row.get("postprocessed_line_count"),
        "raw_score": raw_judgment["score"],
        "raw_decision_label": raw_judgment["decision_label"],
        "raw_failure_tags": raw_judgment["failure_tags"],
        "raw_strength_tags": raw_judgment["strength_tags"],
        "raw_slur_present": raw_judgment["slur_present"],
        "raw_hard_reject_slur": raw_judgment["hard_reject_slur"],
        "raw_analysis": raw_judgment["analysis"],
        "postprocessed_score": post_judgment["score"],
        "postprocessed_decision_label": post_judgment["decision_label"],
        "postprocessed_failure_tags": post_judgment["failure_tags"],
        "postprocessed_strength_tags": post_judgment["strength_tags"],
        "postprocessed_slur_present": post_judgment["slur_present"],
        "postprocessed_hard_reject_slur": post_judgment["hard_reject_slur"],
        "postprocessed_analysis": post_judgment["analysis"],
    }


def is_low_postprocess(row: dict[str, Any]) -> bool:
    names = set(row["postprocess_action_names"])
    return not names or names.issubset(BENIGN_ACTIONS)


def is_ending_only_repair(row: dict[str, Any]) -> bool:
    names = set(row["postprocess_action_names"])
    meaningful = names - BENIGN_ACTIONS
    return bool(meaningful) and meaningful.issubset(ENDING_REPAIR_ACTIONS)


def report_rows(rows: list[dict[str, Any]], *, source: Path, out_dir: Path) -> str:
    total = len(rows)
    post_decisions = Counter(row["postprocessed_decision_label"] for row in rows)
    raw_decisions = Counter(row["raw_decision_label"] for row in rows)
    raw_keeper = [row for row in rows if row["raw_decision_label"] == "keeper"]
    post_keeper = [row for row in rows if row["postprocessed_decision_label"] == "keeper"]
    only_after = [
        row
        for row in rows
        if row["postprocessed_decision_label"] == "keeper" and row["raw_decision_label"] != "keeper"
    ]
    still_failures = Counter(
        tag for row in rows for tag in row.get("postprocessed_failure_tags", []) if tag
    )
    raw_failures = Counter(tag for row in rows for tag in row.get("raw_failure_tags", []) if tag)
    actions = Counter(name for row in rows for name in row.get("postprocess_action_names", []) if name)
    prompt_action_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        prompt_action_counts[int(row.get("prompt_index") or 0)].update(row.get("postprocess_action_names") or [])

    no_slur_rows = [
        row
        for row in rows
        if row.get("raw_analysis", {}).get("no_slurs_requested")
        or row.get("postprocessed_analysis", {}).get("no_slurs_requested")
    ]
    no_slur_raw_fail = sum(1 for row in no_slur_rows if row.get("raw_hard_reject_slur"))
    no_slur_post_fail = sum(1 for row in no_slur_rows if row.get("postprocessed_hard_reject_slur"))

    md: list[str] = [
        "# Postprocess Dependency Analysis",
        "",
        f"- Source: `{source}`",
        f"- Output directory: `{out_dir}`",
        f"- Records: `{total}`",
        "",
        "## Decision Dependency",
        "",
        f"- Raw decisions: `{dict(raw_decisions)}`",
        f"- Postprocessed decisions: `{dict(post_decisions)}`",
        f"- Raw keepers: `{len(raw_keeper)}`",
        f"- Postprocessed keepers: `{len(post_keeper)}`",
        f"- Keepers only after postprocessing: `{len(only_after)}`",
        f"- Raw keepers that stayed keepers: `{sum(1 for row in rows if row['raw_decision_label'] == 'keeper' and row['postprocessed_decision_label'] == 'keeper')}`",
        "",
        "## Postprocessing Dependence",
        "",
        f"- Postprocess action counts: `{dict(actions.most_common())}`",
        "- Prompts most dependent on `drop_dangling_final_line`:",
    ]
    for prompt_index, counts in sorted(
        prompt_action_counts.items(),
        key=lambda item: item[1].get("drop_dangling_final_line", 0),
        reverse=True,
    )[:10]:
        md.append(f"  - prompt `{prompt_index}`: `{counts.get('drop_dangling_final_line', 0)}`")

    md.extend(
        [
            "",
            "## Remaining Failure Tags After Postprocessing",
            "",
            *[f"- `{tag}`: {count}" for tag, count in still_failures.most_common(20)],
            "",
            "## Raw Failure Tags",
            "",
            *[f"- `{tag}`: {count}" for tag, count in raw_failures.most_common(20)],
            "",
            "## No-Slur Failure Split",
            "",
            f"- No-slur prompt rows: `{len(no_slur_rows)}`",
            f"- Raw-model no-slur failures: `{no_slur_raw_fail}`",
            f"- Postprocessed/curated no-slur failures: `{no_slur_post_fail}`",
            "- Interpretation: postprocessing does not remove slurs; these are model/prompt-adherence failures, not cleanup-policy wins.",
            "",
            "## Dataset Bucket Meanings",
            "",
            "- `raw_keep_positive.jsonl`: raw output already judged keeper with no or only benign postprocess actions; strongest SFT-positive candidates.",
            "- `postprocess_repair_pairs.jsonl`: raw output differs from generated output and the postprocessed side is keeper/fixable; useful for repair or preference pairs.",
            "- `fixable_manual_repair_queue.jsonl`: high-score fixables to manually edit before training.",
            "- `hard_negative_rejects.jsonl`: rejected rows with severe tags; useful for DPO reject side and regression evals.",
            "",
            "## Training Implication",
            "",
            "Do not imitate cleaned text blindly. Raw keepers are positive imitation candidates. Postprocessed keepers are repair/pair candidates. Fixables need manual edits or chosen/rejected framing. Hard negatives should teach the model what to avoid.",
        ]
    )
    return "\n".join(md) + "\n"


def main() -> None:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.input):
        raw_text = str(row.get("raw_generated_text") or row.get("raw_text") or "")
        generated_text = str(row.get("generated_text") or row.get("text") or raw_text)
        raw_judgment = recalc_judgment(
            row, raw_text, min_lines=args.min_lines, max_line_words=args.max_line_words
        )
        post_judgment = recalc_judgment(
            row, generated_text, min_lines=args.min_lines, max_line_words=args.max_line_words
        )
        rows.append(enriched_row(row, raw_judgment, post_judgment))

    raw_keep_positive = [
        row
        for row in rows
        if row["raw_decision_label"] == "keeper"
        and row["postprocessed_decision_label"] == "keeper"
        and is_low_postprocess(row)
    ]
    postprocess_repair_pairs = [
        {
            **row,
            "repair_type": "ending_only" if is_ending_only_repair(row) else "mixed_postprocess",
            "chosen": row["generated_text"],
            "rejected": row["raw_generated_text"],
        }
        for row in rows
        if row["raw_differs_from_generated"]
        and row["postprocessed_decision_label"] in {"keeper", "fixable"}
    ]
    fixable_manual_repair_queue = sorted(
        [row for row in rows if row["postprocessed_decision_label"] == "fixable"],
        key=lambda item: -float(item.get("postprocessed_score") or 0),
    )[: args.top_fixable]
    hard_negative_rejects = [
        row
        for row in rows
        if row["postprocessed_decision_label"] == "reject"
        and (set(row.get("postprocessed_failure_tags") or []) & SEVERE_NEGATIVE_TAGS)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "analyzed_all.jsonl", rows)
    write_jsonl(args.out_dir / "raw_keep_positive.jsonl", raw_keep_positive)
    write_jsonl(args.out_dir / "postprocess_repair_pairs.jsonl", postprocess_repair_pairs)
    write_jsonl(args.out_dir / "fixable_manual_repair_queue.jsonl", fixable_manual_repair_queue)
    write_jsonl(args.out_dir / "hard_negative_rejects.jsonl", hard_negative_rejects)
    (args.out_dir / "postprocess_dependency_report.md").write_text(
        report_rows(rows, source=args.input, out_dir=args.out_dir), encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "records": len(rows),
        "raw_keep_positive": len(raw_keep_positive),
        "postprocess_repair_pairs": len(postprocess_repair_pairs),
        "fixable_manual_repair_queue": len(fixable_manual_repair_queue),
        "hard_negative_rejects": len(hard_negative_rejects),
        "raw_decisions": dict(Counter(row["raw_decision_label"] for row in rows)),
        "postprocessed_decisions": dict(Counter(row["postprocessed_decision_label"] for row in rows)),
        "postprocess_action_counts": dict(
            Counter(name for row in rows for name in row.get("postprocess_action_names", [])).most_common()
        ),
    }
    (args.out_dir / "postprocess_dependency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
