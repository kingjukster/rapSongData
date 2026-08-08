#!/usr/bin/env python3
"""Run a bounded constraint-first technical lyric generation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.run_teacher_revise_rank import (
    ApiBudget,
    DIMENSIONS,
    JUDGE_SCHEMA,
    POSITIVE_DIMENSIONS,
    api_json,
    append_jsonl,
    judge_system,
    load_prompts,
    read_jsonl,
    resolve_returned_id,
    stable_id,
    strict_pass,
    structural_metrics,
    words,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = Path("configs/prompts/qwen3_4b_12line_targeted_stage2_prompts.json")
DEFAULT_BASELINE = Path("runs/qwen3_4b_teacher_revise_v4/phase1b_technical/judgments.jsonl")
DEFAULT_OUTPUT = Path("runs/qwen3_4b_teacher_revise_v4/phase1c_constraint_first")

CHAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}},
        "lines": {"type": "array", "minItems": 2, "items": {"type": "integer", "minimum": 1, "maximum": 12}},
    },
    "required": ["terms", "lines"],
    "additionalProperties": False,
}
END_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "minItems": 2, "items": {"type": "integer", "minimum": 1, "maximum": 12}},
        "anchors": {"type": "array", "minItems": 2, "items": {"type": "string"}},
    },
    "required": ["lines", "anchors"],
    "additionalProperties": False,
}
PLAN_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "integer", "minimum": 1, "maximum": 4},
                    "plan": {
                        "type": "object",
                        "properties": {
                            "end_rhyme_groups": {"type": "array", "minItems": 3, "items": END_GROUP_SCHEMA},
                            "multisyllabic_chains": {"type": "array", "minItems": 3, "items": CHAIN_SCHEMA},
                            "internal_rhyme_lines": {"type": "array", "minItems": 6, "items": {"type": "integer", "minimum": 1, "maximum": 12}},
                            "narrative_movements": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"}},
                            "final_payoff": {"type": "string"},
                        },
                        "required": ["end_rhyme_groups", "multisyllabic_chains", "internal_rhyme_lines", "narrative_movements", "final_payoff"],
                        "additionalProperties": False,
                    },
                    "lyrics": {"type": "string"},
                },
                "required": ["slot", "plan", "lyrics"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def normalize(text: str) -> str:
    return " ".join(words(text))


def validate_plan(candidate: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    lyrics = str(candidate["lyrics"])
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    plan = candidate["plan"]
    failures: list[str] = []
    if len(lines) != 12:
        failures.append("line_count")

    covered_end_lines: set[int] = set()
    missing_end_anchors: list[str] = []
    for group in plan["end_rhyme_groups"]:
        for line_number, anchor in zip(group["lines"], group["anchors"]):
            if 1 <= int(line_number) <= len(lines) and normalize(lines[int(line_number) - 1]).endswith(normalize(anchor)):
                covered_end_lines.add(int(line_number))
            else:
                missing_end_anchors.append(str(anchor))
    if len(covered_end_lines) < 8:
        failures.append("planned_end_anchors")

    present_chains = 0
    missing_chain_terms: list[str] = []
    lyric_normalized = normalize(lyrics)
    for chain in plan["multisyllabic_chains"]:
        present = [normalize(term) in lyric_normalized for term in chain["terms"]]
        if sum(present) >= 2:
            present_chains += 1
        missing_chain_terms.extend(str(term) for term, found in zip(chain["terms"], present) if not found)
    if present_chains < 3:
        failures.append("planned_multisyllabic_chains")

    internal_targets = {int(value) for value in plan["internal_rhyme_lines"]}
    if len(internal_targets) < 6:
        failures.append("planned_internal_lines")
    metrics = structural_metrics(lyrics)
    if metrics["internal_rhyme_line_count"] < 4:
        failures.append("observed_internal_rhyme")
    if metrics["multisyllabic_rhyme_pair_count"] < 2:
        failures.append("observed_multisyllabic_rhyme")
    return not failures, sorted(set(failures)), {
        "covered_end_lines": sorted(covered_end_lines),
        "present_chain_count": present_chains,
        "missing_end_anchors": missing_end_anchors,
        "missing_chain_terms": missing_chain_terms,
        "structural_metrics_v1": metrics,
    }


def system_prompt() -> str:
    return (
        "You are a meticulous technical rap writer. For each of four distinct candidates, first design a machine-checkable rhyme plan and then write exactly 12 lyric lines from it. "
        "The output is rap, never scene prose. Each end-rhyme group must pair its listed line numbers with anchor words that appear as the exact final words of those lines; cover at least 8 distinct lines. "
        "Provide at least three multisyllabic chains, each with at least two terms that appear verbatim in the lyrics across the listed adjacent lines. Mark at least six internal-rhyme lines. "
        "Build three coherent four-bar narrative movements, keep natural syntax and stable cadence, avoid repeated filler endings, and land the stated payoff on line 12. Return only the schema."
    )


def ensure_manifest(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "prompt_version": "constraint_first_technical_v1",
        "prompt_ids": [stable_id(row["prompt"]) for row in prompts],
        "teacher_model": args.teacher_model,
        "judge_model": args.judge_model,
        "seed": args.seed,
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = args.output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
        raise RuntimeError("Resume refused: constraint-first manifest changed")
    if not path.exists():
        write_json(path, {"fingerprint": fingerprint, "spec": spec})


def record_command(args: argparse.Namespace) -> None:
    command = " ".join([sys.executable, *sys.argv])
    path = args.output_dir / "commands.jsonl"
    command_id = stable_id(command)
    if command_id not in {row.get("command_id") for row in read_jsonl(path)}:
        append_jsonl(path, [{"command_id": command_id, "command": command, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}])


def generate(args: argparse.Namespace, prompts: list[dict[str, Any]], budget: ApiBudget) -> None:
    path = args.output_dir / "candidates.jsonl"
    completed = {row["prompt_id"] for row in read_jsonl(path)}
    for prompt in prompts:
        prompt_id = stable_id(prompt["prompt"])
        if prompt_id in completed:
            continue
        parsed, meta = api_json(
            model=args.teacher_model,
            name="constraint_first_generate",
            schema=PLAN_GENERATION_SCHEMA,
            system=system_prompt(),
            user=str(prompt["prompt"]),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
        )
        rows = []
        for item in parsed["candidates"]:
            candidate_id = stable_id(prompt_id, "constraint-first", item["slot"])
            valid, failures, evidence = validate_plan(item)
            rows.append({
                "candidate_id": candidate_id,
                "prompt_id": prompt_id,
                "family": "technical",
                "theme": prompt["theme"],
                "prompt": prompt["prompt"],
                "source": "constraint_first_generate",
                "slot": item["slot"],
                "plan": item["plan"],
                "lyrics": str(item["lyrics"]).strip(),
                "plan_valid": valid,
                "plan_failures": failures,
                "plan_evidence": evidence,
                "model": args.teacher_model,
                "response": meta,
            })
        append_jsonl(path, rows)


def judge(args: argparse.Namespace, budget: ApiBudget) -> None:
    candidates = [row for row in read_jsonl(args.output_dir / "candidates.jsonl") if row["plan_valid"]]
    rng = random.Random(args.seed)
    mapping = [{"blind_id": stable_id(args.seed, row["candidate_id"], "constraint-blind"), **row} for row in candidates]
    rng.shuffle(mapping)
    write_json(args.output_dir / "blind_map.json", {"rows": mapping})
    path = args.output_dir / "judgments.jsonl"
    completed = {row["blind_id"] for row in read_jsonl(path)}
    for start in range(0, len(mapping), args.judge_batch_size):
        batch = [row for row in mapping[start : start + args.judge_batch_size] if row["blind_id"] not in completed]
        if not batch:
            continue
        payload = [{"blind_id": row["blind_id"], "family": "technical", "prompt": row["prompt"], "lyrics": row["lyrics"]} for row in batch]
        parsed, meta = api_json(
            model=args.judge_model,
            name="constraint_first_blind_judge",
            schema=JUDGE_SCHEMA,
            system=judge_system(),
            user=json.dumps(payload, ensure_ascii=False),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
        )
        by_id = {row["blind_id"]: row for row in batch}
        output = []
        for item in parsed["judgments"]:
            blind_id, repaired = resolve_returned_id(str(item["blind_id"]), by_id)
            source = by_id[blind_id]
            output.append({**item, "blind_id": blind_id, "id_repaired": repaired, **{key: source[key] for key in ("candidate_id", "prompt_id", "family", "theme", "prompt", "lyrics", "source", "model")}, "judge_model": args.judge_model, "response": meta})
        if len(output) != len(batch) or len({row["blind_id"] for row in output}) != len(batch):
            raise RuntimeError("Judge did not return exactly one row per candidate")
        append_jsonl(path, output)
        completed.update(row["blind_id"] for row in output)


def report(args: argparse.Namespace, budget: ApiBudget) -> dict[str, Any]:
    candidates = read_jsonl(args.output_dir / "candidates.jsonl")
    judgments = {row["candidate_id"]: row for row in read_jsonl(args.output_dir / "judgments.jsonl")}
    evaluated = []
    for candidate in candidates:
        judgment = judgments.get(candidate["candidate_id"])
        failures = list(candidate["plan_failures"])
        passed = False
        if judgment:
            passed, strict_failures = strict_pass(judgment)
            failures.extend(strict_failures)
        else:
            failures.append("not_judged_plan_invalid")
        passed = bool(candidate["plan_valid"] and judgment and passed)
        evaluated.append({**candidate, "scores": judgment.get("scores") if judgment else None, "judge_evidence": judgment.get("evidence") if judgment else None, "strict_pass": passed, "strict_failures": sorted(set(failures))})
    baseline = [row for row in read_jsonl(args.baseline) if row.get("source") == "teacher_generate" and row.get("family") == "technical"]
    baseline_pass = sum(strict_pass(row)[0] for row in baseline)
    passed_rows = [row for row in evaluated if row["strict_pass"]]
    themes = len({row["theme"] for row in passed_rows})
    failure_counts = dict(sorted(Counter(failure for row in evaluated for failure in row["strict_failures"]).items()))
    guardrail_failures = sum(any(name in row["strict_failures"] for name in ("repeated_endings", "score_flow_cadence", "score_thematic_depth")) for row in evaluated)
    advance = len(passed_rows) >= 4 and themes >= 3 and guardrail_failures == 0 and len(passed_rows) > baseline_pass
    payload = {
        "schema_version": 1,
        "decision": "advance_to_dataset_campaign" if advance else "stop_synthetic_technical_generation",
        "constraint_first": {"rows": len(evaluated), "plan_valid": sum(row["plan_valid"] for row in evaluated), "strict_pass": len(passed_rows), "strict_pass_rate": round(len(passed_rows) / max(1, len(evaluated)), 4), "passing_themes": themes, "guardrail_failure_rows": guardrail_failures, "failure_counts": failure_counts},
        "saved_direct_baseline": {"rows": len(baseline), "strict_pass": baseline_pass, "strict_pass_rate": round(baseline_pass / max(1, len(baseline)), 4)},
        "api_totals": dict(zip(("calls", "total_tokens"), budget.totals())),
    }
    write_json(args.output_dir / "report.json", payload)
    append_jsonl(args.output_dir / "accepted.jsonl", passed_rows) if passed_rows and not (args.output_dir / "accepted.jsonl").exists() else None
    lines = ["# Constraint-first technical smoke", "", f"Decision: **{payload['decision'].upper()}**", "", f"- Plan-valid: {payload['constraint_first']['plan_valid']}/{len(evaluated)}", f"- Strict pass: {len(passed_rows)}/{len(evaluated)}", f"- Passing themes: {themes}", f"- Saved direct baseline: {baseline_pass}/{len(baseline)}", f"- Guardrail failure rows: {guardrail_failures}", "", "## Failure counts", ""]
    lines.extend(f"- {name}: {count}" for name, count in failure_counts.items())
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-model", default=os.getenv("OPENAI_TEACHER_MODEL") or "gpt-5.4")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--themes", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-api-calls", type=int, default=12)
    parser.add_argument("--max-total-tokens", type=int, default=150000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts, args.themes, ["technical"])
    ensure_manifest(args, prompts)
    record_command(args)
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    generate(args, prompts, budget)
    judge(args, budget)
    print(json.dumps(report(args, budget), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
