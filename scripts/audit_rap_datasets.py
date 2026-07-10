#!/usr/bin/env python3
"""Audit rap training datasets for hard integrity violations.

The checks here are intentionally conservative. They are meant to run before
training and fail when the generated datasets contain rows that provide no
training signal, leak across splits, or contradict their own controls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SAFE_LICENSES = {
    "original_user_written",
    "licensed_or_public_domain",
    "synthetic_transformed",
}

SECTION_TYPES_ALLOWING_SHORT_LINES = {"intro", "adlib", "spoken"}
FILLER_MARKER_RE = re.compile(r"\[\s*filler\s*]", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rap SFT, mutation, preference, and processed datasets.")
    parser.add_argument("--generation", type=Path, default=Path("data/sft/rap_generation_sft.jsonl"))
    parser.add_argument("--mutation", type=Path, default=Path("data/sft/rap_mutation_sft.jsonl"))
    parser.add_argument("--preferences", type=Path, default=Path("data/preferences/rap_quality_pairs.jsonl"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/rap_sections_labeled.parquet"))
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_integrity_audit.json"))
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--min-assistant-tokens", type=int, default=5)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.05)
    parser.add_argument("--max-synthetic-preference-rate", type=float, default=0.05)
    parser.add_argument("--min-preference-quality-gap", type=float, default=0.0)
    parser.add_argument("--label-concentration-warning", type=float, default=0.70)
    parser.add_argument(
        "--allowed-license",
        action="append",
        default=[],
        help="Additional source_license value allowed without a hard violation.",
    )
    parser.add_argument("--no-fail", action="store_true", help="Write reports but always exit 0.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_words(value: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(value or "")))


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            rows.append(payload)
    return rows, errors


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    for key in ("assistant", "completion", "target_completion", "generated_text", "output", "training_text"):
        if row.get(key):
            return str(row[key])
    return ""


def split_value(row: dict[str, Any]) -> str:
    return str(metadata(row).get("split") or row.get("split") or "").strip()


def add_issue(
    dataset: dict[str, Any],
    *,
    severity: str,
    code: str,
    message: str,
    count: int,
    examples: list[Any] | None = None,
) -> None:
    if count <= 0:
        return
    dataset["issues"].append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "count": count,
            "examples": examples or [],
        }
    )


def duplicate_instances(values: list[str]) -> tuple[int, list[dict[str, Any]]]:
    counts = Counter(value for value in values if value)
    total = sum(count - 1 for count in counts.values() if count > 1)
    examples = [{"value": value, "count": count} for value, count in counts.most_common(5) if count > 1]
    return total, examples


def cross_split_values(rows: list[dict[str, Any]], key_fn) -> tuple[int, list[dict[str, Any]]]:
    splits_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = key_fn(row)
        split = split_value(row)
        if key and split:
            splits_by_key[key].add(split)
    leaked = [(key, sorted(splits)) for key, splits in splits_by_key.items() if len(splits) > 1]
    examples = [{"value": key, "splits": splits} for key, splits in leaked[:5]]
    return len(leaked), examples


def concentration_warning(values: list[str], threshold: float) -> tuple[int, list[dict[str, Any]]]:
    filtered = [value for value in values if value]
    if len(filtered) < 20:
        return 0, []
    counts = Counter(filtered)
    value, count = counts.most_common(1)[0]
    share = count / len(filtered)
    if share <= threshold:
        return 0, []
    return count, [{"value": value, "count": count, "share": round(share, 4), "total": len(filtered)}]


def display_value(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "<missing>"


def top_counts(values: list[Any], limit: int = 20) -> dict[str, int]:
    return {
        str(value): int(count)
        for value, count in Counter(display_value(value) for value in values).most_common(limit)
    }


def numeric_summary(values: list[Any]) -> dict[str, float | int | None]:
    numbers: list[float] = []
    for value in values:
        try:
            if value is not None:
                numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numbers:
        return {"count": 0, "min": None, "mean": None, "max": None}
    numbers.sort()
    count = len(numbers)
    return {
        "count": count,
        "min": round(numbers[0], 4),
        "p25": round(numbers[int((count - 1) * 0.25)], 4),
        "median": round(numbers[int((count - 1) * 0.50)], 4),
        "p75": round(numbers[int((count - 1) * 0.75)], 4),
        "p95": round(numbers[int((count - 1) * 0.95)], 4),
        "mean": round(sum(numbers) / count, 4),
        "max": round(numbers[-1], 4),
    }


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;|]", value) if item.strip()]
    return []


def audit_generation(path: Path, args: argparse.Namespace, allowed_licenses: set[str]) -> dict[str, Any]:
    rows, parse_errors = load_jsonl(path)
    result: dict[str, Any] = {"path": str(path), "rows": len(rows), "issues": []}
    add_issue(
        result,
        severity="hard",
        code="jsonl_parse_errors",
        message="Generation JSONL contains parse or schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    texts = [normalize_text(assistant_text(row)) for row in rows]
    duplicate_count, duplicate_examples = duplicate_instances(texts)
    duplicate_rate = duplicate_count / len(rows) if rows else 0.0
    result["duplicate_assistant_output_rate"] = round(duplicate_rate, 4)
    add_issue(
        result,
        severity="hard" if duplicate_rate > args.max_duplicate_rate else "warning",
        code="duplicate_assistant_outputs",
        message="Assistant completions repeat exactly after normalization.",
        count=duplicate_count,
        examples=duplicate_examples,
    )

    keys = []
    for row in rows:
        meta = metadata(row)
        key = (meta.get("song_id"), meta.get("section_id"), meta.get("bar_index"))
        if all(part is not None for part in key):
            keys.append("|".join(str(part) for part in key))
    duplicate_keys, key_examples = duplicate_instances(keys)
    add_issue(
        result,
        severity="hard",
        code="duplicate_bar_keys",
        message="Duplicate (song_id, section_id, bar_index) records found.",
        count=duplicate_keys,
        examples=key_examples,
    )

    short_rows = []
    unknown_licenses = []
    themes: list[str] = []
    emotions: list[str] = []
    splits: list[str] = []
    licenses: list[str] = []
    section_types: list[str] = []
    assistant_word_counts: list[int] = []
    for index, row in enumerate(rows):
        meta = metadata(row)
        section_type = str(meta.get("section_type") or row.get("section_type") or "").lower()
        text = assistant_text(row)
        assistant_word_counts.append(count_words(text))
        if count_words(text) < args.min_assistant_tokens and section_type not in SECTION_TYPES_ALLOWING_SHORT_LINES:
            short_rows.append({"row": index, "tokens": count_words(text), "text": text[:120]})
        license_value = str(meta.get("source_license") or row.get("source_license") or "").lower()
        splits.append(split_value(row))
        licenses.append(license_value)
        section_types.append(section_type)
        if license_value and license_value not in allowed_licenses:
            unknown_licenses.append({"row": index, "source_license": license_value})
        themes.extend(listify(meta.get("themes") or row.get("themes") or meta.get("theme_tags") or row.get("theme_tags")))
        emotions.extend(listify(meta.get("emotions") or row.get("emotions") or meta.get("emotion_tags") or row.get("emotion_tags")))
    result["split_counts"] = top_counts(splits)
    result["source_license_counts"] = top_counts(licenses)
    result["section_type_counts"] = top_counts(section_types)
    result["assistant_word_count_summary"] = numeric_summary(assistant_word_counts)
    add_issue(
        result,
        severity="hard",
        code="very_short_assistant_outputs",
        message="Non-intro/adlib assistant completions are below the token floor.",
        count=len(short_rows),
        examples=short_rows[:5],
    )
    add_issue(
        result,
        severity="hard",
        code="unsafe_or_unknown_source_license",
        message="Rows use source_license values outside the allowed set.",
        count=len(unknown_licenses),
        examples=unknown_licenses[:5],
    )

    song_leak_count, song_leak_examples = cross_split_values(rows, lambda row: str(metadata(row).get("song_id") or ""))
    section_leak_count, section_leak_examples = cross_split_values(
        rows, lambda row: str(metadata(row).get("section_id") or "")
    )
    text_leak_count, text_leak_examples = cross_split_values(rows, lambda row: normalize_text(assistant_text(row)))
    add_issue(
        result,
        severity="hard",
        code="song_id_split_leakage",
        message="The same song_id appears in multiple splits.",
        count=song_leak_count,
        examples=song_leak_examples,
    )
    add_issue(
        result,
        severity="hard",
        code="section_id_split_leakage",
        message="The same section_id appears in multiple splits.",
        count=section_leak_count,
        examples=section_leak_examples,
    )
    add_issue(
        result,
        severity="hard",
        code="normalized_text_split_leakage",
        message="The same normalized assistant text appears in multiple splits.",
        count=text_leak_count,
        examples=text_leak_examples,
    )

    theme_count, theme_examples = concentration_warning(themes, args.label_concentration_warning)
    emotion_count, emotion_examples = concentration_warning(emotions, args.label_concentration_warning)
    add_issue(
        result,
        severity="warning",
        code="theme_label_concentration",
        message="One theme label dominates the dataset.",
        count=theme_count,
        examples=theme_examples,
    )
    add_issue(
        result,
        severity="warning",
        code="emotion_label_concentration",
        message="One emotion label dominates the dataset.",
        count=emotion_count,
        examples=emotion_examples,
    )
    return result


def audit_mutation(path: Path) -> dict[str, Any]:
    rows, parse_errors = load_jsonl(path)
    result: dict[str, Any] = {"path": str(path), "rows": len(rows), "issues": []}
    add_issue(
        result,
        severity="hard",
        code="jsonl_parse_errors",
        message="Mutation JSONL contains parse or schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    mismatches = []
    empty_outputs = []
    splits: list[str] = []
    licenses: list[str] = []
    requested_output_counts: list[Any] = []
    for index, row in enumerate(rows):
        controls = row.get("controls") if isinstance(row.get("controls"), dict) else {}
        requested = controls.get("output_bars")
        output_bars = row.get("output_bars") if isinstance(row.get("output_bars"), list) else []
        meta = metadata(row)
        splits.append(split_value(row))
        licenses.append(str(meta.get("source_license") or row.get("source_license") or "").lower())
        requested_output_counts.append(requested)
        if requested is not None and int(requested) != len(output_bars):
            mismatches.append({"row": index, "controls_output_bars": requested, "actual_output_bars": len(output_bars)})
        if not output_bars or any(not str(item).strip() for item in output_bars):
            empty_outputs.append({"row": index, "actual_output_bars": len(output_bars)})
    result["split_counts"] = top_counts(splits)
    result["source_license_counts"] = top_counts(licenses)
    result["requested_output_bar_counts"] = top_counts(requested_output_counts)
    add_issue(
        result,
        severity="hard",
        code="mutation_output_bar_count_mismatch",
        message="controls.output_bars does not match len(output_bars).",
        count=len(mismatches),
        examples=mismatches[:5],
    )
    add_issue(
        result,
        severity="hard",
        code="mutation_empty_outputs",
        message="Mutation examples contain empty output_bars.",
        count=len(empty_outputs),
        examples=empty_outputs[:5],
    )
    return result


def audit_preferences(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows, parse_errors = load_jsonl(path)
    result: dict[str, Any] = {"path": str(path), "rows": len(rows), "issues": []}
    add_issue(
        result,
        severity="hard",
        code="jsonl_parse_errors",
        message="Preference JSONL contains parse or schema errors.",
        count=len(parse_errors),
        examples=parse_errors[:5],
    )
    identical = []
    inverted_quality = []
    below_min_gap = []
    synthetic = []
    splits: list[str] = []
    licenses: list[str] = []
    quality_gaps: list[float] = []
    for index, row in enumerate(rows):
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        meta = metadata(row)
        splits.append(split_value(row))
        licenses.append(str(meta.get("source_license") or row.get("source_license") or "").lower())
        if normalize_text(chosen) == normalize_text(rejected):
            identical.append({"row": index, "text": chosen[:120]})
        quality_chosen = meta.get("quality_chosen")
        quality_rejected = meta.get("quality_rejected")
        try:
            quality_gap = float(quality_chosen) - float(quality_rejected)
            quality_gaps.append(quality_gap)
            if quality_gap <= 0:
                inverted_quality.append(
                    {
                        "row": index,
                        "quality_chosen": quality_chosen,
                        "quality_rejected": quality_rejected,
                    }
                )
            elif quality_gap < float(args.min_preference_quality_gap):
                below_min_gap.append(
                    {
                        "row": index,
                        "quality_chosen": quality_chosen,
                        "quality_rejected": quality_rejected,
                        "quality_gap": round(quality_gap, 4),
                    }
                )
        except (TypeError, ValueError):
            inverted_quality.append(
                {"row": index, "quality_chosen": quality_chosen, "quality_rejected": quality_rejected}
            )
        if FILLER_MARKER_RE.search(rejected):
            synthetic.append({"row": index, "rejected": rejected[:120]})
    synthetic_rate = len(synthetic) / len(rows) if rows else 0.0
    result["synthetic_preference_reject_rate"] = round(synthetic_rate, 4)
    result["split_counts"] = top_counts(splits)
    result["source_license_counts"] = top_counts(licenses)
    result["quality_gap_summary"] = numeric_summary(quality_gaps)
    add_issue(
        result,
        severity="hard",
        code="identical_preference_pair",
        message="Preference rows have identical chosen and rejected text.",
        count=len(identical),
        examples=identical[:5],
    )
    add_issue(
        result,
        severity="hard",
        code="non_positive_preference_quality_gap",
        message="Preference rows do not have quality_chosen > quality_rejected.",
        count=len(inverted_quality),
        examples=inverted_quality[:5],
    )
    add_issue(
        result,
        severity="hard",
        code="preference_quality_gap_below_minimum",
        message=f"Preference rows have quality gaps below {args.min_preference_quality_gap}.",
        count=len(below_min_gap),
        examples=below_min_gap[:5],
    )
    add_issue(
        result,
        severity="hard" if synthetic_rate > args.max_synthetic_preference_rate else "warning",
        code="synthetic_filler_rejected_examples",
        message="Rejected preference examples contain synthetic [filler] markers.",
        count=len(synthetic),
        examples=synthetic[:5],
    )
    return result


def dataframe_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], "missing_file"
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional local env
        return [], f"pandas_unavailable: {exc}"
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return [], f"parquet_read_failed: {exc}"
    return frame.to_dict("records"), None


def audit_processed(path: Path, args: argparse.Namespace, allowed_licenses: set[str]) -> dict[str, Any]:
    rows, error = dataframe_rows(path)
    result: dict[str, Any] = {"path": str(path), "rows": len(rows), "issues": []}
    add_issue(
        result,
        severity="warning" if error and error != "missing_file" else "hard",
        code="processed_dataset_unreadable",
        message="Processed parquet could not be read.",
        count=1 if error else 0,
        examples=[error] if error else [],
    )
    if error:
        return result

    normalized_lines = [normalize_text(row.get("clean_bar_text") or row.get("bar_text") or "") for row in rows]
    duplicate_count, duplicate_examples = duplicate_instances(normalized_lines)
    duplicate_rate = duplicate_count / len(rows) if rows else 0.0
    result["duplicate_clean_bar_rate"] = round(duplicate_rate, 4)
    result["split_counts"] = top_counts(row.get("split") for row in rows)
    result["source_license_counts"] = top_counts(str(row.get("source_license") or "").lower() for row in rows)
    result["section_type_counts"] = top_counts(str(row.get("section_type") or "").lower() for row in rows)
    result["quality_score_summary"] = numeric_summary([row.get("quality_score") for row in rows])
    result["word_count_summary"] = numeric_summary(
        [count_words(row.get("clean_bar_text") or row.get("bar_text") or "") for row in rows]
    )
    add_issue(
        result,
        severity="hard" if duplicate_rate > args.max_duplicate_rate else "warning",
        code="duplicate_clean_bar_text",
        message="Processed rows contain duplicate normalized bar text.",
        count=duplicate_count,
        examples=duplicate_examples,
    )

    unknown_licenses = []
    very_short = []
    themes: list[str] = []
    emotions: list[str] = []
    for index, row in enumerate(rows):
        license_value = str(row.get("source_license") or "").lower()
        if license_value and license_value not in allowed_licenses:
            unknown_licenses.append({"row": index, "source_license": license_value})
        section_type = str(row.get("section_type") or "").lower()
        text = row.get("clean_bar_text") or row.get("bar_text") or ""
        if count_words(text) < args.min_assistant_tokens and section_type not in SECTION_TYPES_ALLOWING_SHORT_LINES:
            very_short.append({"row": index, "tokens": count_words(text), "text": str(text)[:120]})
        themes.extend(listify(row.get("theme_tags")))
        emotions.extend(listify(row.get("emotion_tags")))
    add_issue(
        result,
        severity="hard",
        code="unsafe_or_unknown_source_license",
        message="Processed rows use source_license values outside the allowed set.",
        count=len(unknown_licenses),
        examples=unknown_licenses[:5],
    )
    add_issue(
        result,
        severity="hard",
        code="very_short_processed_rows",
        message="Processed lyric rows are below the token floor.",
        count=len(very_short),
        examples=very_short[:5],
    )

    row_split = lambda row: str(row.get("split") or "").strip()
    splits_by_song: dict[str, set[str]] = defaultdict(set)
    splits_by_section: dict[str, set[str]] = defaultdict(set)
    splits_by_text: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row_split(row)
        if not split:
            continue
        if row.get("song_id") is not None:
            splits_by_song[str(row["song_id"])].add(split)
        if row.get("section_id") is not None:
            splits_by_section[str(row["section_id"])].add(split)
        text_key = normalize_text(row.get("clean_bar_text") or row.get("bar_text") or "")
        if text_key:
            splits_by_text[text_key].add(split)
    for code, message, mapping in (
        ("song_id_split_leakage", "The same song_id appears in multiple splits.", splits_by_song),
        ("section_id_split_leakage", "The same section_id appears in multiple splits.", splits_by_section),
        ("normalized_text_split_leakage", "The same normalized text appears in multiple splits.", splits_by_text),
    ):
        leaked = [(key, sorted(splits)) for key, splits in mapping.items() if len(splits) > 1]
        add_issue(
            result,
            severity="hard",
            code=code,
            message=message,
            count=len(leaked),
            examples=[{"value": key, "splits": splits} for key, splits in leaked[:5]],
        )

    theme_count, theme_examples = concentration_warning(themes, args.label_concentration_warning)
    emotion_count, emotion_examples = concentration_warning(emotions, args.label_concentration_warning)
    add_issue(
        result,
        severity="warning",
        code="theme_label_concentration",
        message="One theme label dominates the processed dataset.",
        count=theme_count,
        examples=theme_examples,
    )
    add_issue(
        result,
        severity="warning",
        code="emotion_label_concentration",
        message="One emotion label dominates the processed dataset.",
        count=emotion_count,
        examples=emotion_examples,
    )
    return result


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    hard_issue_types = 0
    warning_issue_types = 0
    hard_affected_rows = 0
    warning_affected_rows = 0
    for dataset in report["datasets"].values():
        for issue in dataset["issues"]:
            if issue["severity"] == "hard":
                hard_issue_types += 1
                hard_affected_rows += int(issue["count"])
            else:
                warning_issue_types += 1
                warning_affected_rows += int(issue["count"])
    return {
        "status": "fail" if hard_issue_types else "pass",
        "hard_issue_types": hard_issue_types,
        "hard_affected_rows": hard_affected_rows,
        "warning_issue_types": warning_issue_types,
        "warning_affected_rows": warning_affected_rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Integrity Audit",
        "",
        f"- status: `{report['summary']['status']}`",
        f"- hard issue types: {report['summary']['hard_issue_types']}",
        f"- hard affected rows: {report['summary']['hard_affected_rows']}",
        f"- warning issue types: {report['summary']['warning_issue_types']}",
        f"- warning affected rows: {report['summary']['warning_affected_rows']}",
        "",
    ]
    for name, dataset in report["datasets"].items():
        lines.extend([f"## {name}", "", f"- path: `{dataset['path']}`", f"- rows: {dataset['rows']}", ""])
        for key in (
            "split_counts",
            "source_license_counts",
            "section_type_counts",
            "requested_output_bar_counts",
            "assistant_word_count_summary",
            "word_count_summary",
            "quality_score_summary",
            "quality_gap_summary",
        ):
            if key in dataset:
                lines.append(f"- {key}: `{json.dumps(dataset[key], ensure_ascii=False)}`")
        if any(
            key in dataset
            for key in (
                "split_counts",
                "source_license_counts",
                "section_type_counts",
                "requested_output_bar_counts",
                "assistant_word_count_summary",
                "word_count_summary",
                "quality_score_summary",
                "quality_gap_summary",
            )
        ):
            lines.append("")
        if not dataset["issues"]:
            lines.extend(["- no issues", ""])
            continue
        for issue in dataset["issues"]:
            lines.append(
                f"- `{issue['severity']}` `{issue['code']}` count={issue['count']}: {issue['message']}"
            )
            if issue["examples"]:
                lines.append(f"  - examples: `{json.dumps(issue['examples'][:3], ensure_ascii=False)[:500]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    allowed_licenses = {item.lower() for item in SAFE_LICENSES | set(args.allowed_license)}
    report: dict[str, Any] = {
        "config": {
            "min_assistant_tokens": args.min_assistant_tokens,
            "max_duplicate_rate": args.max_duplicate_rate,
            "max_synthetic_preference_rate": args.max_synthetic_preference_rate,
            "min_preference_quality_gap": args.min_preference_quality_gap,
            "allowed_licenses": sorted(allowed_licenses),
        },
        "datasets": {},
    }
    report["datasets"]["generation"] = audit_generation(args.generation, args, allowed_licenses)
    report["datasets"]["mutation"] = audit_mutation(args.mutation)
    report["datasets"]["preferences"] = audit_preferences(args.preferences, args)
    report["datasets"]["processed"] = audit_processed(args.processed, args, allowed_licenses)
    report["summary"] = summarize(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out = args.markdown_out or args.out.with_suffix(".md")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown_report(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["status"] == "fail" and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
