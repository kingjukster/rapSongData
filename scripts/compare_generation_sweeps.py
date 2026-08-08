"""Compare two local generation sweeps with local curation metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rank_generation_candidates import judge_candidate, text_for


ENDING_TAGS = {
    "weak_ending",
    "unfinished_punctuation",
    "unfinished_thought",
    "unfinished_fragment",
    "unclosed_phrase",
    "abrupt_short_ending",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judged_rows(path: Path, *, min_lines: int, max_line_words: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        record = dict(row)
        judgment = judge_candidate(record, min_lines=min_lines, max_line_words=max_line_words)
        record["score"] = judgment["score"]
        record["decision_label"] = judgment["decision_label"]
        record["failure_tags"] = judgment["failure_tags"]
        record["strength_tags"] = judgment["strength_tags"]
        record["slur_present"] = judgment["slur_present"]
        record["hard_reject_slur"] = judgment["hard_reject_slur"]
        record["analysis"] = judgment["analysis"]
        rows.append(record)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    analyses = [row.get("analysis", {}) for row in rows]
    exact_records = [item for item in analyses if item.get("exact_line_match") is not None]
    hook_records = [item for item in analyses if item.get("hook_line_cap_ok") is not None]
    no_slur_records = [item for item in analyses if item.get("no_slurs_requested")]
    failure_tags = Counter(tag for row in rows for tag in row.get("failure_tags", []))
    decisions = Counter(str(row.get("decision_label")) for row in rows)
    postprocessed = [row for row in rows if row.get("postprocess_applied")]
    return {
        "records": len(rows),
        "decision_counts": dict(decisions),
        "keeper_count": decisions.get("keeper", 0),
        "fixable_count": decisions.get("fixable", 0),
        "reject_count": decisions.get("reject", 0),
        "exact_line_match_rate": round(
            sum(1 for item in exact_records if item.get("exact_line_match") is True) / len(exact_records),
            4,
        )
        if exact_records
        else None,
        "hook_cap_pass_rate": round(
            sum(1 for item in hook_records if item.get("hook_line_cap_ok") is True) / len(hook_records),
            4,
        )
        if hook_records
        else None,
        "weak_ending_count": sum(failure_tags.get(tag, 0) for tag in ENDING_TAGS),
        "dangling_fragment_count": failure_tags.get("abrupt_short_ending", 0) + failure_tags.get("unfinished_fragment", 0),
        "unfinished_fragment_count": failure_tags.get("unfinished_fragment", 0),
        "no_slur_prompt_pass_rate": round(
            sum(1 for item in no_slur_records if item.get("no_slurs_passed") is True) / len(no_slur_records),
            4,
        )
        if no_slur_records
        else None,
        "avg_repeated_line_ratio": round(
            sum(float(item.get("repeated_line_ratio") or 0.0) for item in analyses) / max(1, len(analyses)),
            4,
        ),
        "postprocess_applied_count": len(postprocessed),
        "postprocess_action_counts": dict(
            Counter(
                action.get("action")
                for row in postprocessed
                for action in row.get("postprocess_actions", [])
                if isinstance(action, dict)
            )
        ),
        "failure_tag_counts": dict(failure_tags.most_common(20)),
    }


def top_by_prompt(rows: list[dict[str, Any]], *, limit: int) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row.get("prompt_index") or 0)].append(row)
    return {
        key: sorted(bucket, key=lambda item: -float(item.get("score") or 0))[:limit]
        for key, bucket in sorted(buckets.items())
    }


def delta(new: Any, old: Any) -> Any:
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return round(new - old, 4)
    return None


def write_report(path: Path, *, old_path: Path, new_path: Path, old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    old_summary = summarize(old_rows)
    new_summary = summarize(new_rows)
    metrics = [
        "records",
        "keeper_count",
        "fixable_count",
        "reject_count",
        "exact_line_match_rate",
        "hook_cap_pass_rate",
        "weak_ending_count",
        "dangling_fragment_count",
        "unfinished_fragment_count",
        "no_slur_prompt_pass_rate",
        "avg_repeated_line_ratio",
        "postprocess_applied_count",
    ]
    md: list[str] = [
        "# Local Generation Sweep Comparison",
        "",
        f"- Old: `{old_path}`",
        f"- New: `{new_path}`",
        "",
        "## Metric Snapshot",
        "",
        "| Metric | Old | New | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        old_value = old_summary.get(metric)
        new_value = new_summary.get(metric)
        md.append(f"| `{metric}` | {old_value} | {new_value} | {delta(new_value, old_value)} |")

    md.extend(
        [
            "",
            "## Improvement Attribution",
            "",
            "- Model behavior: compare raw structure metrics such as exact line match, hook cap pass, and repeated-line ratio.",
            "- Postprocessing cleanup: inspect `postprocess_applied_count` and `postprocess_action_counts`; these show where output was changed after generation.",
            "- Local curation: keeper/fixable/reject counts reflect the local judge, not model loss.",
            "- Prompt category differences: inspect the top outputs per prompt below; some prompts still need data or preference work even when aggregate metrics improve.",
            "",
            "## Postprocessing Actions",
            "",
            f"- Old: `{old_summary.get('postprocess_action_counts')}`",
            f"- New: `{new_summary.get('postprocess_action_counts')}`",
            "",
            "## Failure Tags",
            "",
            f"- Old: `{old_summary.get('failure_tag_counts')}`",
            f"- New: `{new_summary.get('failure_tag_counts')}`",
            "",
            "## Top 5 New Outputs Per Prompt",
            "",
        ]
    )
    for prompt_index, bucket in top_by_prompt(new_rows, limit=5).items():
        prompt = bucket[0].get("prompt", "") if bucket else ""
        md.extend([f"### Prompt {prompt_index}", "", str(prompt), ""])
        for rank, row in enumerate(bucket, start=1):
            analysis = row.get("analysis", {}) if isinstance(row.get("analysis"), dict) else {}
            md.extend(
                [
                    (
                        f"#### #{rank} {row.get('decision_label')} score={row.get('score')} "
                        f"sample={row.get('sample_index')}"
                    ),
                    "",
                    f"- failure_tags: `{', '.join(row.get('failure_tags') or []) or 'none'}`",
                    f"- strength_tags: `{', '.join(row.get('strength_tags') or []) or 'none'}`",
                    f"- lines: `{analysis.get('line_count')}` words: `{analysis.get('word_count')}` slurs: `{analysis.get('slur_count')}`",
                    "",
                    "```text",
                    text_for(row),
                    "```",
                    "",
                ]
            )

    md.extend(
        [
            "## Learning Targets",
            "",
            "- Examples to imitate: promote high-scoring keepers that have few or no postprocess actions.",
            "- Fixable examples to repair manually: prioritize high-scoring fixables with ending-only failures.",
            "- Repeated failure patterns to avoid: use increased failure tags as negative examples or DPO rejected candidates.",
            "- Prompt categories where postprocessing helps: prompts with many postprocess actions but improved keeper/fixable counts.",
            "- Prompt categories where the model itself fails: prompts with high reject counts despite postprocessing.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps({"old": old_summary, "new": new_summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-lines", type=int, default=6)
    parser.add_argument("--max-line-words", type=int, default=34)
    args = parser.parse_args()

    old_rows = judged_rows(args.old, min_lines=args.min_lines, max_line_words=args.max_line_words)
    new_rows = judged_rows(args.new, min_lines=args.min_lines, max_line_words=args.max_line_words)
    write_report(args.out, old_path=args.old, new_path=args.new, old_rows=old_rows, new_rows=new_rows)
    print(json.dumps({"status": "complete", "out": str(args.out), "old_records": len(old_rows), "new_records": len(new_rows)}, indent=2))


if __name__ == "__main__":
    main()
