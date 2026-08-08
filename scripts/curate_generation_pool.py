"""Curate a local generation sweep into keeper, fixable, and reject pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rank_generation_candidates import judge_candidate, text_for


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def curate(rows: list[dict[str, Any]], *, min_lines: int, max_line_words: int) -> list[dict[str, Any]]:
    curated: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        judgment = judge_candidate(record, min_lines=min_lines, max_line_words=max_line_words)
        record["score"] = judgment["score"]
        record["candidate_score"] = judgment["score"]
        record["decision_label"] = judgment["decision_label"]
        record["failure_tags"] = judgment["failure_tags"]
        record["candidate_flags"] = judgment["failure_tags"]
        record["strength_tags"] = judgment["strength_tags"]
        record["slur_present"] = judgment["slur_present"]
        record["hard_reject_slur"] = judgment["hard_reject_slur"]
        record["analysis"] = judgment["analysis"]
        curated.append(record)
    curated.sort(
        key=lambda item: (
            str(item.get("decision_label") or "reject") != "keeper",
            str(item.get("decision_label") or "reject") != "fixable",
            int(item.get("prompt_index") or 0),
            -float(item.get("score") or 0),
        )
    )
    return curated


def by_decision(rows: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("decision_label") == decision]


def count_tags(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(tag for row in rows for tag in row.get(field, []) if tag)


def top_by_prompt(rows: list[dict[str, Any]], *, limit: int) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row.get("prompt_index") or 0)].append(row)
    return {
        prompt_index: sorted(bucket, key=lambda item: -float(item.get("score") or 0))[:limit]
        for prompt_index, bucket in sorted(buckets.items())
    }


def write_report(path: Path, *, source: Path, rows: list[dict[str, Any]], top_per_prompt: int) -> None:
    decisions = Counter(str(row.get("decision_label") or "unknown") for row in rows)
    failure_tags = count_tags(rows, "failure_tags")
    strength_tags = count_tags(rows, "strength_tags")
    keepers = by_decision(rows, "keeper")
    fixable = by_decision(rows, "fixable")
    rejects = by_decision(rows, "reject")
    display_pool = keepers + fixable
    top_prompts = top_by_prompt(display_pool or rows, limit=top_per_prompt)
    strongest_fixable = sorted(fixable, key=lambda item: -float(item.get("score") or 0))[:10]

    md: list[str] = [
        "# Local Generation Curation Report",
        "",
        f"- Source: `{source}`",
        f"- Records: `{len(rows)}`",
        f"- Decisions: `{dict(decisions)}`",
        f"- Failure tags: `{dict(failure_tags.most_common(20))}`",
        f"- Strength tags: `{dict(strength_tags.most_common(20))}`",
        "",
        "## Top Candidates Per Prompt",
        "",
    ]
    for prompt_index, bucket in top_prompts.items():
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

    md.extend(["## Strongest Fixable Candidates", ""])
    for row in strongest_fixable:
        analysis = row.get("analysis", {}) if isinstance(row.get("analysis"), dict) else {}
        md.extend(
            [
                f"### prompt={row.get('prompt_index')} sample={row.get('sample_index')} score={row.get('score')}",
                "",
                f"- failure_tags: `{', '.join(row.get('failure_tags') or []) or 'none'}`",
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
            "## Common Reject Reasons",
            "",
            *[f"- `{tag}`: {count}" for tag, count in count_tags(rejects, "failure_tags").most_common(20)],
            "",
            "## Learning Targets",
            "",
            "- Examples to imitate: use `keepers.jsonl`, especially records with `exact_line_count`, `complete_verse_shape`, `compact_hook_shape`, and `no_repeated_lines`.",
            "- Fixable examples to repair manually: use `fixable.jsonl`; prioritize high-score rows with only ending or line-count failures.",
            "- Repeated failure patterns to avoid: use the reject failure-tag counts above as negative-data or preference-pair targets.",
            "- Prompt categories where postprocessing helps: inspect records with `postprocess_applied=true` and keeper/fixable labels.",
            "- Prompt categories where the model itself fails: prompts with high reject counts after postprocessing need data or preference work.",
        ]
    )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-per-prompt", type=int, default=5)
    parser.add_argument("--min-lines", type=int, default=6)
    parser.add_argument("--max-line-words", type=int, default=34)
    args = parser.parse_args()

    rows = curate(read_jsonl(args.input), min_lines=args.min_lines, max_line_words=args.max_line_words)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "curated_all.jsonl", rows)
    write_jsonl(args.out_dir / "keepers.jsonl", by_decision(rows, "keeper"))
    write_jsonl(args.out_dir / "fixable.jsonl", by_decision(rows, "fixable"))
    write_jsonl(args.out_dir / "rejects.jsonl", by_decision(rows, "reject"))
    write_report(args.out_dir / "curation_report.md", source=args.input, rows=rows, top_per_prompt=args.top_per_prompt)
    print(
        json.dumps(
            {
                "status": "complete",
                "input": str(args.input),
                "out_dir": str(args.out_dir),
                "records": len(rows),
                "decisions": dict(Counter(str(row.get("decision_label")) for row in rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
