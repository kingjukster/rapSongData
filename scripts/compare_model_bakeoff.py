"""Write a compact comparison report for local generation bakeoffs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=3, metavar=("NAME", "RUN_DIR", "CURATION_DIR"), required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tag_counts(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(tag for row in rows for tag in row.get(field, []) if tag)


def short_text(text: str, max_chars: int = 900) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def load_run(name: str, run_dir: Path, curation_dir: Path, top_n: int) -> dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json")
    curated = read_jsonl(curation_dir / "curated_all.jsonl")
    decisions = Counter(str(row.get("decision_label") or "unknown") for row in curated)
    failure_tags = tag_counts(curated, "failure_tags")
    strength_tags = tag_counts(curated, "strength_tags")
    keepers = [row for row in curated if row.get("decision_label") == "keeper"]
    top = sorted(keepers or curated, key=lambda row: -float(row.get("score") or 0))[:top_n]
    run_metrics = summary.get("run_metrics", {})
    return {
        "name": name,
        "run_dir": str(run_dir),
        "curation_dir": str(curation_dir),
        "base_model": summary.get("base_model"),
        "adapter_dir": summary.get("adapter_dir"),
        "records": len(curated),
        "summary": summary.get("summary", {}),
        "environment": summary.get("environment", {}),
        "run_metrics": run_metrics,
        "decisions": dict(decisions),
        "failure_tags": dict(failure_tags.most_common(12)),
        "strength_tags": dict(strength_tags.most_common(12)),
        "top_outputs": top,
    }


def write_report(path: Path, runs: list[dict[str, Any]]) -> None:
    md: list[str] = ["# Model Bakeoff Report", ""]
    md.extend(
        [
            "## Runs",
            "",
            "| Model | Records | Keeper | Fixable | Reject | Exact line | Hook cap | No-slur pass | Avg tok/s | Peak VRAM GB | Load sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        summary = run["summary"]
        metrics = run["run_metrics"]
        env = run["environment"]
        decisions = run["decisions"]
        md.append(
            "| {name} | {records} | {keeper} | {fixable} | {reject} | {exact} | {hook} | {slur} | {tok} | {vram} | {load} |".format(
                name=run["name"],
                records=run["records"],
                keeper=decisions.get("keeper", 0),
                fixable=decisions.get("fixable", 0),
                reject=decisions.get("reject", 0),
                exact=summary.get("exact_line_match_rate"),
                hook=summary.get("hook_line_cap_pass_rate"),
                slur=summary.get("no_slur_prompt_pass_rate"),
                tok=metrics.get("avg_tokens_per_second"),
                vram=metrics.get("peak_memory_gb"),
                load=env.get("model_load_seconds"),
            )
        )
    md.extend(["", "## Failure Tags", ""])
    for run in runs:
        md.extend([f"### {run['name']}", "", f"- Failure tags: `{run['failure_tags']}`", f"- Strength tags: `{run['strength_tags']}`", ""])
    md.extend(["## Visible Top Outputs", ""])
    for run in runs:
        md.extend([f"### {run['name']}", ""])
        for index, row in enumerate(run["top_outputs"], start=1):
            md.extend(
                [
                    f"#### #{index} prompt={row.get('prompt_index')} sample={row.get('sample_index')} {row.get('decision_label')} score={row.get('score')}",
                    "",
                    f"- failure_tags: `{', '.join(row.get('failure_tags') or []) or 'none'}`",
                    f"- strength_tags: `{', '.join(row.get('strength_tags') or []) or 'none'}`",
                    "",
                    "```text",
                    short_text(str(row.get("generated_text") or row.get("text") or "")),
                    "```",
                    "",
                ]
            )
    md.extend(
        [
            "## Initial Read",
            "",
            "- Treat this as a small generation bakeoff, not a final model decision.",
            "- The model with the best keeper/fixable balance is the best candidate for a larger sweep.",
            "- A lyrics-domain fine-tune can still lose if it weakens instruction adherence, rap cadence, or line-count control.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = [
        load_run(name, Path(run_dir), Path(curation_dir), args.top_n)
        for name, run_dir, curation_dir in args.run
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.output_md, runs)
    print(json.dumps({"status": "complete", "runs": [run["name"] for run in runs], "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
