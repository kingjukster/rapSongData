from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import PRIVATE_RESEARCH_POLICY, command_record, iter_jsonl, utc_now, write_json
from .evaluation import (
    is_prompt_leakage,
    is_section_label,
    legacy_lyric_lines,
    lyric_lines,
    target_lines,
)


def controlled_lines(text: str) -> list[str]:
    return [
        line
        for line in lyric_lines(text)
        if not is_section_label(line) and not is_prompt_leakage(line)
    ]


def apply_line_controls(text: str, requested_lines: int) -> str:
    lines = controlled_lines(text)
    if requested_lines > 0:
        lines = lines[:requested_lines]
    return "\n".join(lines).strip()


def classify_failure(
    text: str,
    requested_lines: int,
    *,
    generated_tokens: int | None = None,
    token_budget: int = 256,
) -> dict[str, Any]:
    evaluator = legacy_lyric_lines(text)
    controlled = controlled_lines(text)
    labels = [line for line in evaluator if is_section_label(line)]
    leaked = [line for line in evaluator if is_prompt_leakage(line)]
    blank_lines = sum(1 for line in text.splitlines() if not line.strip())
    current_count = len(evaluator)
    controlled_count = len(controlled)
    legacy_exact = requested_lines > 0 and current_count == requested_lines
    corrected_native_exact = requested_lines > 0 and controlled_count == requested_lines
    likely_wrapped = (
        current_count > requested_lines
        and current_count - requested_lines <= 2
        and sum(len(re.findall(r"\w+", line)) <= 3 for line in controlled) >= current_count - requested_lines
    )
    if legacy_exact:
        primary = "exact"
    elif controlled_count == requested_lines and leaked:
        primary = "prompt_leakage"
    elif controlled_count == requested_lines and labels:
        primary = "extra_structural_text"
    elif controlled_count == requested_lines:
        primary = "parser_disagreement"
    elif current_count < requested_lines:
        if generated_tokens is not None and generated_tokens >= token_budget - 2:
            primary = "truncation"
        else:
            primary = "underlength"
    elif likely_wrapped:
        primary = "wrapped_line"
    elif controlled_count > requested_lines:
        primary = "failure_to_terminate"
    else:
        primary = "overlength"
    controlled_text = apply_line_controls(text, requested_lines)
    system_count = len(lyric_lines(controlled_text))
    return {
        "legacy_evaluator_exact": legacy_exact,
        "native_exact": corrected_native_exact,
        "primary_failure": primary,
        "requested_lines": requested_lines,
        "evaluator_line_count": current_count,
        "controlled_line_count_before_stop": controlled_count,
        "system_line_count": system_count,
        "system_exact": requested_lines > 0 and system_count == requested_lines,
        "legacy_parser_false_positive": legacy_exact and not corrected_native_exact,
        "legacy_parser_false_negative": corrected_native_exact and not legacy_exact,
        "line_delta": current_count - requested_lines,
        "generated_tokens": generated_tokens,
        "blank_line_count": blank_lines,
        "section_label_count": len(labels),
        "prompt_leakage_count": len(leaked),
        "suspected_wrapped_line": likely_wrapped,
        "ends_without_line_boundary": bool(text) and not text.endswith("\n"),
    }


def analyze_compliance(args: argparse.Namespace) -> dict[str, Any]:
    generation_payload = json.loads(Path(args.generations).read_text(encoding="utf-8"))
    outputs = list(generation_payload.get("outputs") or [])
    rows = list(iter_jsonl(Path(args.corpus_dir) / "test.jsonl"))[: len(outputs)]
    if len(rows) != len(outputs):
        raise RuntimeError("Test rows and generation outputs do not have the same length.")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, local_files_only=True)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    target_summary: dict[int, Counter[str]] = defaultdict(Counter)
    for index, (row, output) in enumerate(zip(rows, outputs)):
        requested = target_lines(row)
        generated_tokens = None
        if tokenizer is not None:
            generated_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        result = classify_failure(
            output,
            requested,
            generated_tokens=generated_tokens,
            token_budget=args.token_budget,
        )
        result.update({"sample_index": index, "title": row.get("title"), "output": output})
        records.append(result)
        counts[result["primary_failure"]] += 1
        target_summary[requested][result["primary_failure"]] += 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "compliance_failures.csv"
    fields = [key for key in records[0] if key != "output"] + ["output"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    legacy_exact = sum(record["legacy_evaluator_exact"] for record in records)
    native_exact = sum(record["native_exact"] for record in records)
    system_exact = sum(record["system_exact"] for record in records)
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "source_generations": str(args.generations),
        "sample_count": len(records),
        "legacy_evaluator_exact_count": legacy_exact,
        "legacy_evaluator_exact_rate": legacy_exact / len(records),
        "corrected_native_exact_count": native_exact,
        "corrected_native_exact_rate": native_exact / len(records),
        "native_exact_count": native_exact,
        "native_exact_rate": native_exact / len(records),
        "deterministic_system_exact_count": system_exact,
        "deterministic_system_exact_rate": system_exact / len(records),
        "legacy_failure_count": len(records) - legacy_exact,
        "corrected_native_failure_count": len(records) - native_exact,
        "failure_count": len(records) - legacy_exact,
        "failure_classes": dict(sorted(counts.items())),
        "by_target_lines": {
            str(target): dict(sorted(summary.items()))
            for target, summary in sorted(target_summary.items())
        },
        "parser_validation": {
            "blank_lines_ignored": True,
            "canonical_tokens_ignored": True,
            "human_section_labels_ignored": False,
            "alternative_parser_changes_count": sum(
                record["controlled_line_count_before_stop"] != record["evaluator_line_count"]
                for record in records
            ),
            "legacy_false_positives": sum(record["legacy_parser_false_positive"] for record in records),
            "legacy_false_negatives": sum(record["legacy_parser_false_negative"] for record in records),
        },
        "records_csv": str(csv_path),
    }
    write_json(output_dir / "compliance_report.json", report)
    markdown = [
        "# Scratch 30M SFT compliance analysis",
        "",
        f"- Samples: {len(records)}",
        f"- Legacy evaluator compliance: {legacy_exact}/{len(records)} ({legacy_exact / len(records):.1%})",
        f"- Corrected native compliance: {native_exact}/{len(records)} ({native_exact / len(records):.1%})",
        f"- Deterministic-system compliance: {system_exact}/{len(records)} ({system_exact / len(records):.1%})",
        f"- Legacy failures classified: {len(records) - legacy_exact}",
        "",
        "## Primary failure classes",
        "",
    ]
    markdown.extend(f"- {name}: {count}" for name, count in sorted(counts.items()) if name != "exact")
    (output_dir / "compliance_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def add_compliance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--token-budget", type=int, default=256)
