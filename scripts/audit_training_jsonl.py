#!/usr/bin/env python3
"""Audit packaged training JSONL files before a fine-tune run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
CHAT_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(?P<content>.*?)<\|im_end\|>", re.S)
CHAT_USER_RE = re.compile(r"<\|im_start\|>user\n(?P<content>.*?)<\|im_end\|>", re.S)
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--preference-pairs", type=Path, default=None)
    parser.add_argument("--hard-negatives", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--target-line-count", type=int, default=12)
    parser.add_argument("--min-preference-score-delta", type=float, default=0.0)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lyric_lines(value: Any) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def word_count(value: Any) -> int:
    return len(WORD_RE.findall(str(value or "")))


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p25": round(pct(0.25), 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(pct(0.75), 4),
        "p95": round(pct(0.95), 4),
        "mean": round(statistics.mean(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [{"line": None, "error": "missing_file"}]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(payload, dict):
                errors.append({"line": line_number, "error": "row_is_not_object"})
                continue
            payload["_line_number"] = line_number
            rows.append(payload)
    return rows, errors


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def extract_assistant(row: dict[str, Any]) -> str:
    text = str(row.get("training_text") or "")
    match = CHAT_ASSISTANT_RE.search(text)
    if match:
        return match.group("content").strip()
    return str(row.get("assistant") or "")


def extract_user(row: dict[str, Any]) -> str:
    text = str(row.get("training_text") or "")
    match = CHAT_USER_RE.search(text)
    if match:
        return match.group("content").strip()
    return str(row.get("prompt") or "")


def add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    count: int,
    examples: list[dict[str, Any]] | None = None,
    severity: str = "hard",
) -> None:
    if count <= 0:
        return
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "count": count,
            "examples": examples or [],
        }
    )


def audit_split(path: Path, split: str, target_line_count: int) -> dict[str, Any]:
    rows, parse_errors = read_jsonl(path)
    issues: list[dict[str, Any]] = []
    add_issue(
        issues,
        code="jsonl_parse_errors",
        message=f"{split} JSONL contains parse/schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    empty_assistant: list[dict[str, Any]] = []
    line_errors: list[dict[str, Any]] = []
    control_token_hits: list[dict[str, Any]] = []
    ids: list[str] = []
    prompt_keys: list[str] = []
    line_counts: list[int] = []
    word_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    normalized_assistants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        meta = metadata(row)
        row_id = str(row.get("id") or meta.get("candidate_id") or f"{split}:{index}")
        assistant = extract_assistant(row)
        lines = lyric_lines(assistant)
        ids.append(row_id)
        prompt_keys.append(str(meta.get("prompt_key") or normalize_text(extract_user(row))[:80]))
        source_counts[str(meta.get("source_bucket") or meta.get("source") or "unknown")] += 1
        issue_counts[str(meta.get("judge_issue") or "none")] += 1
        for tag in meta.get("quality_tags") or []:
            tag_counts[str(tag)] += 1
        line_counts.append(len(lines))
        word_counts.append(word_count(assistant))
        normalized_assistants[normalize_text(assistant)].append(
            {"row": index, "id": row_id, "prompt_key": prompt_keys[-1]}
        )
        if not assistant.strip():
            empty_assistant.append({"row": index, "id": row_id})
        expected_lines = int(meta.get("target_line_count") or target_line_count)
        if len(lines) != expected_lines:
            line_errors.append(
                {
                    "row": index,
                    "id": row_id,
                    "expected": expected_lines,
                    "actual": len(lines),
                }
            )
        if any(token in assistant for token in CONTROL_TOKENS):
            control_token_hits.append({"row": index, "id": row_id})
    duplicate_ids = [
        {"id": row_id, "count": count}
        for row_id, count in Counter(ids).items()
        if count > 1
    ]
    duplicate_assistants = [
        {"normalized_text": text[:120], "count": len(items), "examples": items[:3]}
        for text, items in normalized_assistants.items()
        if text and len(items) > 1
    ]
    add_issue(
        issues,
        code="empty_assistant",
        message=f"{split} rows contain empty assistant text.",
        count=len(empty_assistant),
        examples=empty_assistant[:5],
    )
    add_issue(
        issues,
        code="line_count_mismatch",
        message=f"{split} assistant line count does not match target.",
        count=len(line_errors),
        examples=line_errors[:5],
    )
    add_issue(
        issues,
        code="assistant_control_tokens",
        message=f"{split} assistant content contains chat control tokens.",
        count=len(control_token_hits),
        examples=control_token_hits[:5],
    )
    add_issue(
        issues,
        code="duplicate_row_ids",
        message=f"{split} contains duplicate row ids.",
        count=len(duplicate_ids),
        examples=duplicate_ids[:5],
    )
    add_issue(
        issues,
        code="duplicate_assistant_text_within_split",
        message=f"{split} contains duplicate normalized assistant text.",
        count=len(duplicate_assistants),
        examples=duplicate_assistants[:5],
        severity="warning",
    )
    return {
        "path": str(path),
        "split": split,
        "rows": len(rows),
        "issues": issues,
        "ids": ids,
        "prompt_keys": prompt_keys,
        "normalized_assistants": normalized_assistants,
        "line_count_summary": numeric_summary(line_counts),
        "assistant_word_count_summary": numeric_summary(word_counts),
        "source_counts": dict(source_counts),
        "judge_issue_counts": dict(issue_counts),
        "quality_tag_counts": dict(tag_counts),
    }


def audit_cross_split(train: dict[str, Any], validation: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    train_ids = set(train["ids"])
    validation_ids = set(validation["ids"])
    id_overlap = sorted(train_ids & validation_ids)
    add_issue(
        issues,
        code="train_validation_id_overlap",
        message="Train and validation share row ids.",
        count=len(id_overlap),
        examples=[{"id": item} for item in id_overlap[:5]],
    )
    train_prompts = set(train["prompt_keys"])
    validation_prompts = set(validation["prompt_keys"])
    prompt_overlap = sorted(train_prompts & validation_prompts)
    add_issue(
        issues,
        code="train_validation_prompt_overlap",
        message="Train and validation share prompt keys.",
        count=len(prompt_overlap),
        examples=[{"prompt_key": item} for item in prompt_overlap[:5]],
    )
    train_texts = set(train["normalized_assistants"])
    validation_texts = set(validation["normalized_assistants"])
    text_overlap = sorted(text for text in train_texts & validation_texts if text)
    add_issue(
        issues,
        code="train_validation_assistant_text_overlap",
        message="Train and validation share normalized assistant text.",
        count=len(text_overlap),
        examples=[{"normalized_text": item[:120]} for item in text_overlap[:5]],
    )
    return issues


def audit_manifest(path: Path | None, train: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    issues: list[dict[str, Any]] = []
    if not path.exists():
        return {"path": str(path), "issues": [{"severity": "hard", "code": "missing_manifest", "message": "Manifest file missing.", "count": 1, "examples": []}]}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "issues": [{"severity": "hard", "code": "manifest_parse_error", "message": str(exc), "count": 1, "examples": []}]}
    counts = manifest.get("counts") if isinstance(manifest, dict) else {}
    if isinstance(counts, dict):
        expected_train = counts.get("train_rows")
        expected_validation = counts.get("validation_rows")
        expected_total = counts.get("total_rows") or counts.get("strict_sft_examples")
        mismatches: list[dict[str, Any]] = []
        if expected_train is not None and int(expected_train) != int(train["rows"]):
            mismatches.append({"field": "train_rows", "expected": expected_train, "actual": train["rows"]})
        if expected_validation is not None and int(expected_validation) != int(validation["rows"]):
            mismatches.append(
                {"field": "validation_rows", "expected": expected_validation, "actual": validation["rows"]}
            )
        if expected_total is not None and int(expected_total) != int(train["rows"] + validation["rows"]):
            mismatches.append(
                {
                    "field": "total_rows_or_strict_sft_examples",
                    "expected": expected_total,
                    "actual": train["rows"] + validation["rows"],
                }
            )
        add_issue(
            issues,
            code="manifest_count_mismatch",
            message="Manifest counts do not match JSONL row counts.",
            count=len(mismatches),
            examples=mismatches,
        )
    return {
        "path": str(path),
        "name": manifest.get("name") if isinstance(manifest, dict) else None,
        "status": manifest.get("status") if isinstance(manifest, dict) else None,
        "issues": issues,
    }


def audit_preference_pairs(path: Path | None, min_delta: float) -> dict[str, Any] | None:
    if path is None:
        return None
    rows, parse_errors = read_jsonl(path)
    issues: list[dict[str, Any]] = []
    add_issue(
        issues,
        code="preference_parse_errors",
        message="Preference JSONL contains parse/schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    identical: list[dict[str, Any]] = []
    below_delta: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    deltas: list[float] = []
    for index, row in enumerate(rows):
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        meta = metadata(row)
        split_counts[str(meta.get("split") or "unknown")] += 1
        if normalize_text(chosen) == normalize_text(rejected):
            identical.append({"row": index, "id": row.get("id")})
        value = meta.get("score_delta", meta.get("quality_delta"))
        try:
            delta = float(value)
            deltas.append(delta)
            if delta < min_delta:
                below_delta.append({"row": index, "id": row.get("id"), "score_delta": delta})
        except (TypeError, ValueError):
            if min_delta > 0:
                below_delta.append({"row": index, "id": row.get("id"), "score_delta": value})
    add_issue(
        issues,
        code="identical_preference_pair",
        message="Preference rows have identical chosen and rejected text.",
        count=len(identical),
        examples=identical[:5],
    )
    add_issue(
        issues,
        code="preference_score_delta_below_minimum",
        message=f"Preference rows have score delta below {min_delta}.",
        count=len(below_delta),
        examples=below_delta[:5],
    )
    return {
        "path": str(path),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "score_delta_summary": numeric_summary(deltas),
        "issues": issues,
    }


def audit_hard_negatives(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    rows, parse_errors = read_jsonl(path)
    issues: list[dict[str, Any]] = []
    add_issue(
        issues,
        code="hard_negative_parse_errors",
        message="Hard-negative JSONL contains parse/schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    issue_counts: Counter[str] = Counter()
    line_counts: list[int] = []
    for row in rows:
        issue_counts[str(metadata(row).get("judge_issue") or "unknown")] += 1
        line_counts.append(len(lyric_lines(row.get("rejected"))))
    return {
        "path": str(path),
        "rows": len(rows),
        "judge_issue_counts": dict(issue_counts),
        "line_count_summary": numeric_summary(line_counts),
        "issues": issues,
    }


def collect_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for value in report.get("datasets", {}).values():
        if isinstance(value, dict):
            issues.extend(value.get("issues") or [])
    issues.extend(report.get("cross_split_issues") or [])
    return issues


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    issues = collect_issues(report)
    hard = [issue for issue in issues if issue.get("severity") == "hard"]
    warnings = [issue for issue in issues if issue.get("severity") == "warning"]
    return {
        "status": "fail" if hard else "pass",
        "hard_issue_types": len(hard),
        "hard_affected_rows": sum(int(issue.get("count") or 0) for issue in hard),
        "warning_issue_types": len(warnings),
        "warning_affected_rows": sum(int(issue.get("count") or 0) for issue in warnings),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Training JSONL Audit",
        "",
        f"- status: `{report['summary']['status']}`",
        f"- hard issue types: `{report['summary']['hard_issue_types']}`",
        f"- hard affected rows: `{report['summary']['hard_affected_rows']}`",
        f"- warning issue types: `{report['summary']['warning_issue_types']}`",
        f"- warning affected rows: `{report['summary']['warning_affected_rows']}`",
        "",
    ]
    for name, dataset in report["datasets"].items():
        if dataset is None:
            continue
        lines.extend([f"## {name}", "", f"- path: `{dataset.get('path')}`", f"- rows: `{dataset.get('rows', 'n/a')}`"])
        for key in (
            "name",
            "status",
            "line_count_summary",
            "assistant_word_count_summary",
            "source_counts",
            "judge_issue_counts",
            "quality_tag_counts",
            "split_counts",
            "score_delta_summary",
        ):
            if key in dataset:
                lines.append(f"- {key}: `{json.dumps(dataset[key], ensure_ascii=False)}`")
        if not dataset.get("issues"):
            lines.extend(["", "- no issues", ""])
            continue
        lines.append("")
        for issue in dataset["issues"]:
            lines.append(f"- `{issue['severity']}` `{issue['code']}` count={issue['count']}: {issue['message']}")
            if issue.get("examples"):
                lines.append(f"  - examples: `{json.dumps(issue['examples'][:3], ensure_ascii=False)[:500]}`")
        lines.append("")
    if report.get("cross_split_issues"):
        lines.extend(["## cross_split", ""])
        for issue in report["cross_split_issues"]:
            lines.append(f"- `{issue['severity']}` `{issue['code']}` count={issue['count']}: {issue['message']}")
            if issue.get("examples"):
                lines.append(f"  - examples: `{json.dumps(issue['examples'][:3], ensure_ascii=False)[:500]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    train = audit_split(args.train, "train", args.target_line_count)
    validation = audit_split(args.validation, "validation", args.target_line_count)
    report: dict[str, Any] = {
        "config": {
            "target_line_count": args.target_line_count,
            "min_preference_score_delta": args.min_preference_score_delta,
        },
        "datasets": {
            "train": {key: value for key, value in train.items() if key not in {"ids", "prompt_keys", "normalized_assistants"}},
            "validation": {
                key: value
                for key, value in validation.items()
                if key not in {"ids", "prompt_keys", "normalized_assistants"}
            },
            "manifest": audit_manifest(args.manifest, train, validation),
            "preference_pairs": audit_preference_pairs(args.preference_pairs, args.min_preference_score_delta),
            "hard_negatives": audit_hard_negatives(args.hard_negatives),
        },
        "cross_split_issues": audit_cross_split(train, validation),
    }
    report["summary"] = summarize(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["status"] == "fail" and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
