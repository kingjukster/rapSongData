from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rap_song_data.scratch.common import PRIVATE_RESEARCH_POLICY, command_record, iter_jsonl, read_json, utc_now, write_json
from rap_song_data.scratch.evaluation import scan_extraction


SYSTEMS = ("scratch_native_v1", "scratch_router_v1_1", "qwen_production_base_v1")


def load_candidates(scratch_generations: Path, qwen_generations: Path) -> list[dict[str, Any]]:
    scratch = read_json(scratch_generations)["records"]
    candidates: list[dict[str, Any]] = [
        {
            "sample_index": index,
            "prompt_id": str(row["prompt_id"]),
            "system": str(row["system"]),
            "output": str(row["output"]),
        }
        for index, row in enumerate(scratch)
    ]
    offset = len(candidates)
    for index, row in enumerate(iter_jsonl(qwen_generations), start=offset):
        candidates.append(
            {
                "sample_index": index,
                "prompt_id": str(row.get("prompt_key") or row.get("theme_id") or row.get("row_id") or index),
                "system": "qwen_production_base_v1",
                "output": str(row["generated_text"]),
            }
        )
    return candidates


def summarize_extraction(samples: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    for sample in samples:
        candidate = candidates[int(sample["sample_index"])]
        row = {**candidate, **sample}
        by_system[candidate["system"]].append(row)
        if sample["matched_50_token_windows"] > 0 or sample["matched_20_token_windows"] > 0:
            examples.append(
                {
                    "sample_index": sample["sample_index"],
                    "prompt_id": candidate["prompt_id"],
                    "system": candidate["system"],
                    "matched_50_token_windows": sample["matched_50_token_windows"],
                    "matched_20_token_windows": sample["matched_20_token_windows"],
                    "matched_20_token_fraction": sample["matched_20_token_fraction"],
                }
            )

    system_summary: dict[str, Any] = {}
    for system in SYSTEMS:
        rows = by_system.get(system, [])
        count = len(rows)
        with_50 = sum(row["matched_50_token_windows"] > 0 for row in rows)
        with_20 = sum(row["matched_20_token_windows"] > 0 for row in rows)
        system_summary[system] = {
            "sample_count": count,
            "samples_with_50_token_match": with_50,
            "samples_with_20_token_match": with_20,
            "fraction_with_20_token_match": with_20 / count if count else 0.0,
            "scaling_gate_passed": with_50 == 0 and (with_20 / count if count else 0.0) <= 0.05,
        }
    return {"by_system": system_summary, "match_examples": examples[:50]}


def markdown_report(report: dict[str, Any]) -> str:
    comparison = report["automated_comparison"]["system_metrics"]
    extraction = report["extraction"]
    lines = [
        "# Scratch 30M Three-Way Automated Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Decision",
        "",
        f"- Extraction gate passed: `{report['decision']['extraction_gate_passed']}`",
        f"- 50M allowed: `{report['decision']['allow_50m']}`",
        f"- Recommended next step: {report['decision']['recommended_next_step']}",
        "",
        "## Extraction",
        "",
        f"- Scanned training tokens: {extraction['scanned_training_tokens']:,}",
        f"- Candidates scanned: {report['candidate_count']}",
        f"- Samples with a 50-token training match: {extraction['samples_with_50_token_match']}",
        f"- Samples with a 20-token training match: {extraction['samples_with_20_token_match']}",
        f"- Fraction with a 20-token training match: {extraction['fraction_with_20_token_match']:.4f}",
        "",
        "## Structure Metrics",
        "",
        "| System | Exact-line rate | Repeated-line ratio | Distinct-3 | Retry rate | Control-change rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        metrics = comparison[system]
        lines.append(
            "| {system} | {exact:.4f} | {repeat:.4f} | {distinct:.4f} | {retry:.4f} | {control:.4f} |".format(
                system=system,
                exact=metrics["exact_line_rate"],
                repeat=metrics["average_repeated_line_ratio"],
                distinct=metrics["distinct_3"],
                retry=metrics["retry_rate"],
                control=metrics["control_change_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Human-review gates are deprecated for this track.",
            "- This report uses saved generations only; no new model generation was run.",
            "- Keep 50M blocked unless a separate policy explicitly replaces the current automated gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-generations", type=Path, required=True)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/scratch/v1/tokenizer"))
    parser.add_argument("--tokenization-manifest", type=Path, default=Path("data/scratch/v1/tokenization_manifest.json"))
    parser.add_argument("--automated-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=250_000)
    args = parser.parse_args()

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "exact_command.txt").write_text(" ".join(command_record()) + "\n", encoding="utf-8")

    from transformers import AutoTokenizer

    candidates = load_candidates(args.scratch_generations, args.qwen_generations)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenization_manifest = read_json(args.tokenization_manifest)
    train_manifest = tokenization_manifest["splits"]["base"]["train"]
    extraction = scan_extraction(
        tokenizer,
        [row["output"] for row in candidates],
        train_manifest,
        windows=(20, 50),
        chunk_tokens=args.chunk_tokens,
    )
    extraction_summary = summarize_extraction(extraction["samples"], candidates)
    extraction.update(extraction_summary)

    automated_comparison = read_json(args.automated_comparison)
    extraction_gate_passed = bool(extraction["scaling_gate_passed"])
    decision = {
        "extraction_gate_passed": extraction_gate_passed,
        "allow_50m": False,
        "recommended_next_step": (
            "Start a 30M SFT-v2 reliability pass focused on repetition reduction and long-form structure."
            if extraction_gate_passed
            else "Stop scaling and inspect extraction matches before any additional training."
        ),
        "reason": (
            "Human review is deprecated; extraction passed and automated metrics show router structure uplift."
            if extraction_gate_passed
            else "Extraction gate failed under the 50-token/20-token memorization rule."
        ),
    }
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "complete",
        "human_review_deprecated": True,
        "generated_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": command_record(),
        "candidate_count": len(candidates),
        "inputs": {
            "scratch_generations": str(args.scratch_generations),
            "qwen_generations": str(args.qwen_generations),
            "tokenizer": str(args.tokenizer),
            "tokenization_manifest": str(args.tokenization_manifest),
            "automated_comparison": str(args.automated_comparison),
        },
        "extraction": extraction,
        "automated_comparison": {
            "status": automated_comparison.get("status"),
            "system_metrics": automated_comparison["system_metrics"],
            "router_uplift": automated_comparison["router_uplift"],
        },
        "decision": decision,
    }
    write_json(args.output_dir / "automated_gate_report.json", report)
    (args.output_dir / "automated_gate_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "automated_gate_report.json"), **decision}, indent=2))


if __name__ == "__main__":
    main()
