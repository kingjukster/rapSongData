"""Export top ranked non-hook Qwen3-4B sweep verses.

The exporter reads an annotated local curation JSONL file and writes a readable
Markdown file plus a JSONL companion. It is intentionally local-only and uses
the curation metadata already present in the sweep rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
INCOMPLETE_END_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "cause",
    "could",
    "for",
    "from",
    "have",
    "here",
    "his",
    "hold",
    "how",
    "i",
    "if",
    "in",
    "inside",
    "is",
    "like",
    "make",
    "might",
    "my",
    "of",
    "on",
    "or",
    "our",
    "outside",
    "push",
    "same",
    "save",
    "serious",
    "should",
    "so",
    "sure",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "while",
    "who",
    "why",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
}
BAD_FAILURE_WEIGHTS = {
    "empty_output": 3.0,
    "source_artifact": 2.5,
    "no_slur_fail": 2.5,
    "question_drift": 2.0,
    "violent_derailment": 2.0,
    "hit_token_cap": 2.0,
    "incomplete_ending": 2.0,
    "profanity_fail": 1.7,
    "line_count_miss": 1.2,
    "weak_ending": 1.2,
    "too_short": 1.0,
    "question_ending": 0.8,
    "non_ascii": 0.4,
    "slur_present": 1.5,
    "dialogue_like": 0.8,
    "repeated_lines": 1.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--keepers-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-token-cap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-incomplete-ending", action=argparse.BooleanOptionalAction, default=False)
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


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def requested_line_count(prompt: str) -> int | None:
    match = re.search(r"\b(?:exactly\s+)?(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, re.I)
    return int(match.group(1)) if match else None


def is_hook(prompt: str) -> bool:
    return bool(re.search(r"\bhook\b", prompt, re.I))


def complete_ending_score(text: str) -> float:
    output_lines = lines(text)
    if not output_lines:
        return 0.0
    last = output_lines[-1].strip()
    last_words = words(last)
    if not last_words:
        return 0.0
    score = 1.0
    if not re.search(r"[.!?]$", last):
        score -= 0.45
    if last_words[-1] in INCOMPLETE_END_WORDS:
        score -= 0.45
    if last.count("(") > last.count(")") or last.count("[") > last.count("]"):
        score -= 0.45
    if len(last_words) <= 4:
        score -= 0.2
    if last.endswith((",", ":", ";", "-", "...")):
        score -= 0.45
    return max(0.0, score)


def line_shape_score(prompt: str, text: str) -> float:
    output_lines = lines(text)
    target = requested_line_count(prompt)
    if target is None:
        return 1.0 if 10 <= len(output_lines) <= 20 else 0.72
    delta = abs(len(output_lines) - target)
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.65
    return max(0.0, 0.65 - 0.15 * (delta - 1))


def repeated_line_penalty(text: str) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines(text)]
    normalized = [line for line in normalized if line]
    if not normalized:
        return 1.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    repeated = sum(count for count in counts.values() if count > 1)
    return min(1.0, repeated / len(normalized))


def rank_row(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    prompt = str(row.get("prompt") or "")
    text = str(row.get("generated_text") or "")
    output_lines = lines(text)
    base_score = float(row.get("score") or 0.0)
    failure_tags = row.get("failure_tags") or []
    failure_penalty = sum(BAD_FAILURE_WEIGHTS.get(tag, 0.6) for tag in failure_tags)
    completion = complete_ending_score(text)
    shape = line_shape_score(prompt, text)
    repeat_penalty = repeated_line_penalty(text)
    post_penalty = 0.05 * len(row.get("postprocess_actions") or [])
    length_penalty = 0.0
    if len(output_lines) < 8:
        length_penalty += 0.5
    avg_line_words = mean([len(words(line)) for line in output_lines]) if output_lines else 0.0
    if avg_line_words > 18:
        length_penalty += 0.12
    strict_score = (
        base_score
        + 0.5 * completion
        + 0.3 * shape
        - 0.25 * repeat_penalty
        - 0.2 * failure_penalty
        - post_penalty
        - length_penalty
    )
    return round(strict_score, 4), {
        "completion_score": round(completion, 3),
        "line_shape_score": round(shape, 3),
        "line_count": len(output_lines),
        "requested_line_count": requested_line_count(prompt),
        "avg_line_words": round(avg_line_words, 2),
    }


def eligible(
    row: dict[str, Any],
    *,
    keepers_only: bool,
    allow_token_cap: bool,
    allow_incomplete_ending: bool,
) -> bool:
    prompt = str(row.get("prompt") or "")
    if is_hook(prompt):
        return False
    if keepers_only and row.get("decision_label") != "keeper":
        return False
    if not allow_token_cap and row.get("hit_token_cap") and not row.get("hit_eos"):
        return False
    if not allow_incomplete_ending and "incomplete_ending" in (row.get("failure_tags") or []):
        return False
    return bool(str(row.get("generated_text") or "").strip())


def main() -> None:
    args = parse_args()
    ranked: list[dict[str, Any]] = []
    for row in read_jsonl(args.input):
        if not eligible(
            row,
            keepers_only=args.keepers_only,
            allow_token_cap=args.allow_token_cap,
            allow_incomplete_ending=args.allow_incomplete_ending,
        ):
            continue
        strict_score, metrics = rank_row(row)
        enriched = dict(row)
        enriched["strict_rank_score"] = strict_score
        enriched["strict_rank_metrics"] = metrics
        ranked.append(enriched)
    ranked.sort(
        key=lambda row: (
            float(row.get("strict_rank_score") or 0.0),
            float(row.get("score") or 0.0),
            row.get("hit_eos") is True,
            len(row.get("strength_tags") or []),
            -len(row.get("failure_tags") or []),
            not bool(row.get("postprocess_applied")),
        ),
        reverse=True,
    )
    selected = ranked[: args.count]

    records: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        records.append(
            {
                "rank": rank,
                "row_id": row.get("row_id"),
                "candidate_index": row.get("candidate_index"),
                "prompt": row.get("prompt"),
                "generated_text": row.get("generated_text"),
                "decision_label": row.get("decision_label"),
                "score": row.get("score"),
                "strict_rank_score": row.get("strict_rank_score"),
                "failure_tags": row.get("failure_tags"),
                "strength_tags": row.get("strength_tags"),
                "postprocess_applied": row.get("postprocess_applied"),
                "postprocess_actions": row.get("postprocess_actions"),
                "hit_eos": row.get("hit_eos"),
                "hit_token_cap": row.get("hit_token_cap"),
                "finish_reason": row.get("finish_reason"),
                "strict_rank_metrics": row.get("strict_rank_metrics"),
            }
        )

    output_jsonl = args.output_jsonl or args.output_md.with_suffix(".jsonl")
    write_jsonl(output_jsonl, records)

    markdown = [
        "# Top Generated Verses",
        "",
        f"Source: `{args.input}`",
        "",
        (
            "Ranking excludes hooks"
            + ("" if args.allow_token_cap else ", cap-truncated rows")
            + ("" if args.allow_incomplete_ending else ", incomplete terminal lines")
            + "."
        ),
        "",
    ]
    for record in records:
        metrics = record["strict_rank_metrics"]
        requested = metrics.get("requested_line_count")
        requested_text = requested if requested is not None else "n/a"
        tags = ", ".join(record.get("failure_tags") or []) or "none"
        actions = ", ".join(record.get("postprocess_actions") or []) or "none"
        markdown.extend(
            [
                f"## {record['rank']}. Candidate {record.get('candidate_index')}",
                "",
                f"- Prompt: {record.get('prompt')}",
                (
                    f"- Decision: `{record.get('decision_label')}` | Local score: `{record.get('score')}` | "
                    f"Strict rank score: `{record.get('strict_rank_score')}`"
                ),
                (
                    f"- Lines: `{metrics.get('line_count')}` | Requested lines: `{requested_text}` | "
                    f"Finish: `{record.get('finish_reason')}` | Postprocess: `{record.get('postprocess_applied')}`"
                ),
                f"- Failure tags: `{tags}` | Postprocess actions: `{actions}`",
                "",
                "```text",
                str(record.get("generated_text") or "").strip(),
                "```",
                "",
            ]
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(markdown), encoding="utf-8")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output_md": str(args.output_md),
                "output_jsonl": str(output_jsonl),
                "selected": len(records),
                "eligible": len(ranked),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
