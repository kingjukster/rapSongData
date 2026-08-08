"""Compare two OpenAI-judged generation sweeps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(name: str, run_path: Path, judged_path: Path, report_path: Path, adapter: str) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = read_jsonl(judged_path)
    decisions = Counter(str(row.get("decision")) for row in rows)
    tags = Counter(tag for row in rows for tag in row.get("tags", []) if tag != "none")
    best = sorted(
        rows,
        key=lambda row: (
            -int(row.get("overall_score") or 0),
            -int(row.get("creative_score") or 0),
            -int(row.get("control_score") or 0),
        ),
    )[:5]
    return {
        "name": name,
        "adapter": adapter,
        "run_summary": str(run_path),
        "judged_jsonl": str(judged_path),
        "judged_report": str(report_path),
        "wall_seconds": run.get("wall_seconds"),
        "run_metrics": run.get("run_metrics"),
        "generation_summary": run.get("summary"),
        "judged_count": len(rows),
        "decision_counts": dict(decisions),
        "top_tags": dict(tags.most_common(12)),
        "best_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "decision": row.get("decision"),
                "overall_score": row.get("overall_score"),
                "creative_score": row.get("creative_score"),
                "control_score": row.get("control_score"),
                "prompt_index": row.get("prompt_index"),
                "sample_index": row.get("sample_index"),
                "tags": row.get("tags"),
                "notes": row.get("notes"),
                "best_lines": row.get("best_lines"),
            }
            for row in best
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        summarize(
            "checkpoint_500",
            Path("reports/generation_eval_mixed_sft_500_sweep_960_run/run_summary.json"),
            Path("reports/generation_eval_mixed_sft_500_sweep_960_openai_judged.jsonl"),
            Path("reports/generation_eval_mixed_sft_500_sweep_960_openai_judged.md"),
            "model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-384-500-simple/checkpoint-500",
        ),
        summarize(
            "ai_labeled_2000",
            Path("reports/generation_eval_ai_labeled_balanced_2000_sweep_960_run/run_summary.json"),
            Path("reports/generation_eval_ai_labeled_balanced_2000_sweep_960_openai_judged.jsonl"),
            Path("reports/generation_eval_ai_labeled_balanced_2000_sweep_960_openai_judged.md"),
            "model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-ai-labeled-balanced-384-2000/checkpoint-2000",
        ),
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({item["name"]: item for item in runs}, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Checkpoint 500 vs AI-Labeled 2000 OpenAI-Judged Comparison",
        "",
        "Both runs used the same 12 prompts, 80 samples per prompt, batch size 4, max_new_tokens 180, temperature 0.86, top_p 0.9, repetition_penalty 1.18, and no slur blocking.",
        "",
        "| Run | Candidates | Judged | Keeper | Fixable | Reject | Exact line match | Hook cap pass | No-slur pass | Slur outputs | Wall time | Tok/s | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in runs:
        gen = item["generation_summary"] or {}
        decisions = item["decision_counts"]
        metrics = item["run_metrics"] or {}
        md.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}m | {} | {} |".format(
                item["name"],
                gen.get("record_count"),
                item["judged_count"],
                decisions.get("keeper", 0),
                decisions.get("fixable", 0),
                decisions.get("reject", 0),
                gen.get("exact_line_match_rate"),
                gen.get("hook_line_cap_pass_rate"),
                gen.get("no_slur_prompt_pass_rate"),
                gen.get("slur_violation_prompt_count"),
                round((item.get("wall_seconds") or 0) / 60, 2),
                metrics.get("avg_tokens_per_second"),
                metrics.get("peak_memory_gb"),
            )
        )

    md.extend(
        [
            "",
            "## Read",
            "",
            "- Checkpoint-500 is slightly ahead as a candidate generator: it produced 1 OpenAI-rated keeper and 95 fixables, versus 0 keepers and 93 fixables for AI-labeled 2000.",
            "- AI-labeled 2000 had better exact line-count adherence, but checkpoint-500 had better no-slur prompt adherence and fewer slur-containing outputs overall.",
            "- Both checkpoints share the same core failure modes: awkward phrasing, prose drift, unfinished endings, generic phrasing, and off-prompt drift.",
            "- This supports preference/edit training more than another blind SFT run.",
            "",
            "## Recommended Next Data Move",
            "",
            "Run OpenAI labels earlier in the pipeline, before the refined playlist/train split. Label the broader processed table or broad section/SFT candidate pool, then build train/validation from labels instead of labeling only the already-narrow refined SFT set.",
            "",
            "Suggested broader sources:",
            "",
            "- `data/processed/rap_sections_labeled.parquet`",
            "- existing pre-filter SFT candidate files, if present",
            "- current generation candidate judgments as preference/edit data",
            "",
            "Build target: complete-verse and hook/control-heavy SFT v3, with mutation/fragment caps, plus DPO pairs from judged generations.",
        ]
    )
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"output_md": str(args.output_md), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
