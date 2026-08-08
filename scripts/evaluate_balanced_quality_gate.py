#!/usr/bin/env python3
"""Compare paired ranked generation JSONL and apply balanced quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILY_ORDER = ("melodic", "story", "technical", "clean")
EPSILON = 1e-12
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "expected_prompt_count": 16,
        "expected_theme_count": 4,
        "minimum_exact_line_count": 15,
        "minimum_average_quality_delta": 0.0,
        "minimum_paired_wins": None,
        "minimum_family_quality_delta": -0.02,
    },
    "promotion": {
        "expected_prompt_count": 60,
        "expected_theme_count": 15,
        "minimum_exact_line_count": 59,
        "minimum_average_quality_delta": 0.02,
        "minimum_paired_wins": 36,
        "minimum_family_quality_delta": 0.0,
    },
    "confirmation": {
        "expected_prompt_count": 24,
        "expected_theme_count": 6,
        "minimum_exact_line_count": 24,
        "minimum_average_quality_delta": 0.01,
        "minimum_paired_wins": 14,
        "minimum_family_quality_delta": -0.01,
    },
    "weak_family_probe": {
        "expected_prompt_count": 12,
        "expected_theme_count": 6,
        "minimum_exact_line_count": 12,
        "minimum_average_quality_delta": 0.01,
        "minimum_paired_wins": 7,
        "minimum_family_quality_delta": 0.0,
        "active_families": ("story", "clean"),
    },
}
HARD_FAILURE_FIELDS = (
    "slur",
    "prompt_leakage",
    "repetition",
    "incomplete_ending",
    "high_copy",
)


class GateInputError(ValueError):
    """Raised when gate evidence is incomplete, ambiguous, or unbalanced."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ranked", type=Path, required=True, help="Base quality-ranked JSONL.")
    parser.add_argument("--adapter-ranked", type=Path, required=True, help="Adapter quality-ranked JSONL.")
    parser.add_argument("--prompts", type=Path, required=True, help="Structured balanced prompt JSON.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--adapter-label", default="adapter")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateInputError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise GateInputError(f"Expected an object at {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise GateInputError(f"Ranked input contains no rows: {path}")
    return rows


def normalize_family(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    for family in FAMILY_ORDER:
        if normalized == family or normalized.endswith(f"_{family}"):
            return family
    raise GateInputError(f"Unsupported prompt family: {value!r}")


def required_nonempty_text(item: dict[str, Any], field: str, *, source: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GateInputError(f"{source} is missing a non-empty {field!r} field.")
    return value.strip()


def load_prompt_specs(path: Path, *, profile: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateInputError(f"Invalid prompt JSON at {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise GateInputError("Prompt JSON must contain a list of structured prompt objects.")

    specs: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_keys: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise GateInputError(f"Prompt item {index} is not an object.")
        source = f"prompt item {index}"
        prompt = required_nonempty_text(item, "prompt", source=source)
        theme = required_nonempty_text(item, "theme", source=source)
        family = normalize_family(item.get("prompt_family"))
        try:
            target_line_count = int(item.get("target_line_count"))
        except (TypeError, ValueError) as exc:
            raise GateInputError(f"{source} has an invalid target_line_count.") from exc
        if target_line_count != 12:
            raise GateInputError(f"{source} targets {target_line_count} lines; balanced gates require 12.")
        prompt_key = str(item.get("prompt_key") or f"{family}:{theme}").strip()
        if prompt in seen_prompts:
            raise GateInputError(f"Prompt file contains duplicate prompt text at item {index}.")
        if prompt_key in seen_keys:
            raise GateInputError(f"Prompt file contains duplicate prompt_key={prompt_key!r}.")
        seen_prompts.add(prompt)
        seen_keys.add(prompt_key)
        specs.append(
            {
                "prompt_key": prompt_key,
                "prompt": prompt,
                "theme": theme,
                "family": family,
                "target_line_count": target_line_count,
            }
        )

    thresholds = PROFILES[profile]
    expected_count = int(thresholds["expected_prompt_count"])
    if len(specs) != expected_count:
        raise GateInputError(
            f"{profile} profile requires {expected_count} prompts; prompt file contains {len(specs)}."
        )
    family_counts = Counter(spec["family"] for spec in specs)
    active_families = tuple(thresholds.get("active_families", FAMILY_ORDER))
    expected_per_family = expected_count // len(active_families)
    expected_family_counts = {family: expected_per_family for family in active_families}
    if dict(family_counts) != expected_family_counts:
        raise GateInputError(
            f"Prompt family counts must be {expected_family_counts}; observed {dict(family_counts)}."
        )
    themes: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        themes[spec["theme"]].append(spec["family"])
    expected_theme_count = int(thresholds["expected_theme_count"])
    if len(themes) != expected_theme_count:
        raise GateInputError(
            f"{profile} profile requires {expected_theme_count} themes; observed {len(themes)}."
        )
    bad_themes = {
        theme: sorted(families)
        for theme, families in themes.items()
        if Counter(families) != Counter(active_families)
    }
    if bad_themes:
        raise GateInputError(f"Every theme must contain each family exactly once: {bad_themes}")
    return specs


def nested_metric(row: dict[str, Any], field: str, *, source: str, prompt_key: str) -> Any:
    structural = row.get("structural_metrics")
    if not isinstance(structural, dict) or field not in structural or structural[field] is None:
        raise GateInputError(f"{source} row {prompt_key!r} is missing structural_metrics.{field}.")
    return structural[field]


def required_bool(value: Any, *, source: str, prompt_key: str, field: str) -> bool:
    if not isinstance(value, (bool, int)) or value not in (True, False, 0, 1):
        raise GateInputError(f"{source} row {prompt_key!r} has non-boolean {field}.")
    return bool(value)


def required_nonnegative_float(value: Any, *, source: str, prompt_key: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GateInputError(f"{source} row {prompt_key!r} has invalid {field}.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise GateInputError(f"{source} row {prompt_key!r} has invalid {field}={value!r}.")
    return result


def observation(row: dict[str, Any], spec: dict[str, Any], *, source: str) -> dict[str, Any]:
    prompt_key = spec["prompt_key"]
    try:
        quality_score = float(row.get("quality_score"))
    except (TypeError, ValueError) as exc:
        raise GateInputError(f"{source} row {prompt_key!r} has invalid quality_score.") from exc
    if not math.isfinite(quality_score):
        raise GateInputError(f"{source} row {prompt_key!r} has non-finite quality_score.")

    # The prompt file is authoritative. Older ranked artifacts inferred family
    # from prompt keywords and can misclassify technical prompts whose theme
    # contains words such as "clean".
    row_theme = row.get("theme")
    if row_theme is not None and str(row_theme).strip() and str(row_theme).strip() != spec["theme"]:
        raise GateInputError(f"{source} row {prompt_key!r} theme conflicts with the prompt file.")

    exact = required_bool(
        nested_metric(row, "exact_line_match", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="exact_line_match",
    )
    slur_count = required_nonnegative_float(
        nested_metric(row, "slur_count", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="slur_count",
    )
    repeated_line_ratio = required_nonnegative_float(
        nested_metric(row, "repeated_line_ratio", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="repeated_line_ratio",
    )
    prompt_leakage = required_bool(
        nested_metric(row, "prompt_leakage", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="prompt_leakage",
    )
    incomplete = required_bool(
        nested_metric(row, "incomplete_ending", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="incomplete_ending",
    )
    high_copy = required_bool(
        nested_metric(row, "high_copy_similarity", source=source, prompt_key=prompt_key),
        source=source,
        prompt_key=prompt_key,
        field="high_copy_similarity",
    )
    return {
        "quality_score": quality_score,
        "exact_line_match": exact,
        "failures": {
            "slur": slur_count > 0.0,
            "prompt_leakage": prompt_leakage,
            "repetition": repeated_line_ratio > 0.0,
            "incomplete_ending": incomplete,
            "high_copy": high_copy,
        },
    }


def select_rows(
    rows: list[dict[str, Any]], specs: list[dict[str, Any]], *, source: str
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    specs_by_prompt = {spec["prompt"]: spec for spec in specs}
    selected: dict[str, dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or prompt not in specs_by_prompt:
            continue
        spec = specs_by_prompt[prompt]
        prompt_key = spec["prompt_key"]
        if prompt_key in selected:
            duplicate_keys.append(prompt_key)
            continue
        selected[prompt_key] = observation(row, spec, source=source)
    if duplicate_keys:
        raise GateInputError(f"{source} has duplicate selected prompts: {sorted(set(duplicate_keys))}")
    missing = [spec["prompt_key"] for spec in specs if spec["prompt_key"] not in selected]
    if missing:
        raise GateInputError(f"{source} is missing {len(missing)} required prompts: {missing[:20]}")
    return selected, {
        "input_row_count": len(rows),
        "selected_row_count": len(selected),
        "ignored_row_count": len(rows) - len(selected),
    }


def rounded(value: float) -> float:
    return round(value, 6)


def summarize_model(observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures = {
        field: sum(1 for item in observations.values() if item["failures"][field])
        for field in HARD_FAILURE_FIELDS
    }
    scores = [item["quality_score"] for item in observations.values()]
    exact_count = sum(1 for item in observations.values() if item["exact_line_match"])
    return {
        "row_count": len(observations),
        "average_quality_score": rounded(statistics.mean(scores)),
        "median_quality_score": rounded(statistics.median(scores)),
        "exact_line_match_count": exact_count,
        "exact_line_match_rate": rounded(exact_count / len(observations)),
        "hard_failure_counts": failures,
        "hard_failure_total": sum(failures.values()),
    }


def evaluate_gate(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    profile: str,
    base_label: str,
    adapter_label: str,
) -> dict[str, Any]:
    if not base_label.strip() or not adapter_label.strip() or base_label == adapter_label:
        raise GateInputError("Base and adapter labels must be distinct and non-empty.")
    base, base_selection = select_rows(base_rows, specs, source="base-ranked")
    adapter, adapter_selection = select_rows(adapter_rows, specs, source="adapter-ranked")

    per_prompt: list[dict[str, Any]] = []
    family_deltas: dict[str, list[float]] = defaultdict(list)
    family_base_scores: dict[str, list[float]] = defaultdict(list)
    family_adapter_scores: dict[str, list[float]] = defaultdict(list)
    family_wins: dict[str, Counter[str]] = defaultdict(Counter)
    paired_wins: Counter[str] = Counter()
    for spec in specs:
        prompt_key = spec["prompt_key"]
        base_item = base[prompt_key]
        adapter_item = adapter[prompt_key]
        delta = adapter_item["quality_score"] - base_item["quality_score"]
        if delta > EPSILON:
            winner = adapter_label
        elif delta < -EPSILON:
            winner = base_label
        else:
            winner = "tie"
        family = spec["family"]
        paired_wins[winner] += 1
        family_wins[family][winner] += 1
        family_deltas[family].append(delta)
        family_base_scores[family].append(base_item["quality_score"])
        family_adapter_scores[family].append(adapter_item["quality_score"])
        per_prompt.append(
            {
                "prompt_key": prompt_key,
                "theme": spec["theme"],
                "family": family,
                "base_quality_score": rounded(base_item["quality_score"]),
                "adapter_quality_score": rounded(adapter_item["quality_score"]),
                "adapter_minus_base_quality": rounded(delta),
                "winner": winner,
                "base_exact_line_match": base_item["exact_line_match"],
                "adapter_exact_line_match": adapter_item["exact_line_match"],
                "adapter_failures": [
                    field for field in HARD_FAILURE_FIELDS if adapter_item["failures"][field]
                ],
            }
        )

    base_metrics = summarize_model(base)
    adapter_metrics = summarize_model(adapter)
    average_delta = statistics.mean(item["quality_score"] for item in adapter.values()) - statistics.mean(
        item["quality_score"] for item in base.values()
    )
    families = {}
    active_families = tuple(PROFILES[profile].get("active_families", FAMILY_ORDER))
    for family in active_families:
        family_delta = statistics.mean(family_deltas[family])
        families[family] = {
            "row_count": len(family_deltas[family]),
            "base_average_quality_score": rounded(statistics.mean(family_base_scores[family])),
            "adapter_average_quality_score": rounded(statistics.mean(family_adapter_scores[family])),
            "adapter_minus_base_quality": rounded(family_delta),
            "adapter_wins": family_wins[family][adapter_label],
            "base_wins": family_wins[family][base_label],
            "ties": family_wins[family]["tie"],
        }

    thresholds = dict(PROFILES[profile])
    family_floor = float(thresholds["minimum_family_quality_delta"])
    family_failures = [
        family
        for family in active_families
        if families[family]["adapter_minus_base_quality"] + EPSILON < family_floor
    ]
    gates: dict[str, dict[str, Any]] = {
        "exact_line_compliance": {
            "passed": adapter_metrics["exact_line_match_count"]
            >= int(thresholds["minimum_exact_line_count"]),
            "rule": "Adapter exact-line matches must meet the profile minimum.",
            "observed": adapter_metrics["exact_line_match_count"],
            "minimum": thresholds["minimum_exact_line_count"],
        },
        "zero_hard_failures": {
            "passed": adapter_metrics["hard_failure_total"] == 0,
            "rule": "Adapter must have zero slur, leakage, repetition, incomplete-ending, and high-copy failures.",
            "observed": adapter_metrics["hard_failure_counts"],
        },
        "average_quality_delta": {
            "passed": average_delta + EPSILON >= float(thresholds["minimum_average_quality_delta"]),
            "rule": "Adapter mean quality delta must meet the profile minimum.",
            "observed": rounded(average_delta),
            "minimum": thresholds["minimum_average_quality_delta"],
        },
        "family_quality_floor": {
            "passed": not family_failures,
            "rule": "Every prompt-family mean quality delta must meet the profile floor.",
            "minimum": family_floor,
            "failing_families": family_failures,
            "observed": {
                family: families[family]["adapter_minus_base_quality"] for family in active_families
            },
        },
    }
    minimum_wins = thresholds["minimum_paired_wins"]
    if minimum_wins is not None:
        gates["paired_wins"] = {
            "passed": paired_wins[adapter_label] >= int(minimum_wins),
            "rule": "Adapter paired prompt wins must meet the promotion minimum.",
            "observed": paired_wins[adapter_label],
            "minimum": minimum_wins,
        }
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "schema_version": 1,
        "profile": profile,
        "decision": "pass" if not failed_gates else "fail",
        "passed": not failed_gates,
        "labels": {"base": base_label, "adapter": adapter_label, "tie": "tie"},
        "thresholds": thresholds,
        "selection": {"base": base_selection, "adapter": adapter_selection},
        "metrics": {
            "base": base_metrics,
            "adapter": adapter_metrics,
            "adapter_minus_base_average_quality": rounded(average_delta),
        },
        "paired": {
            "prompt_count": len(specs),
            "adapter_wins": paired_wins[adapter_label],
            "base_wins": paired_wins[base_label],
            "ties": paired_wins["tie"],
        },
        "families": families,
        "gates": gates,
        "failed_gates": failed_gates,
        "per_prompt": per_prompt,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path, row_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if row_count is not None:
        record["row_count"] = row_count
    return record


def markdown_report(report: dict[str, Any]) -> str:
    if "error" in report:
        return "\n".join(
            [
                "# Balanced Quality Gate",
                "",
                "Decision: **ERROR**",
                "",
                f"- Type: `{report['error']['type']}`",
                f"- Message: {report['error']['message']}",
                "",
            ]
        )
    metrics = report["metrics"]
    base = metrics["base"]
    adapter = metrics["adapter"]
    labels = report["labels"]
    lines = [
        f"# Balanced Quality Gate: {report['profile']}",
        "",
        f"Decision: **{report['decision'].upper()}**",
        "",
        "## Overall",
        "",
        f"| Metric | {labels['base']} | {labels['adapter']} |",
        "|---|---:|---:|",
        f"| Selected prompts | {base['row_count']} | {adapter['row_count']} |",
        f"| Average quality | {base['average_quality_score']:.4f} | {adapter['average_quality_score']:.4f} |",
        f"| Exact 12-line matches | {base['exact_line_match_count']} | {adapter['exact_line_match_count']} |",
        f"| Hard-failure instances | {base['hard_failure_total']} | {adapter['hard_failure_total']} |",
        "",
        f"- Mean paired quality delta: `{metrics['adapter_minus_base_average_quality']:+.4f}`",
        f"- Paired wins: {labels['adapter']} `{report['paired']['adapter_wins']}`, "
        f"{labels['base']} `{report['paired']['base_wins']}`, ties `{report['paired']['ties']}`",
        "",
        "## Prompt families",
        "",
        "| Family | Base avg | Adapter avg | Delta | Adapter wins | Base wins | Ties |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    active_families = tuple(report["thresholds"].get("active_families", FAMILY_ORDER))
    for family in active_families:
        item = report["families"][family]
        lines.append(
            f"| {family} | {item['base_average_quality_score']:.4f} | "
            f"{item['adapter_average_quality_score']:.4f} | "
            f"{item['adapter_minus_base_quality']:+.4f} | {item['adapter_wins']} | "
            f"{item['base_wins']} | {item['ties']} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Result | Rule |",
            "|---|---|---|",
        ]
    )
    for name, gate in report["gates"].items():
        result = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"| `{name}` | {result} | {gate['rule']} |")
    lines.extend(["", "## Adapter hard failures", ""])
    failures = adapter["hard_failure_counts"]
    lines.extend(f"- {field}: `{failures[field]}`" for field in HARD_FAILURE_FIELDS)
    lines.append("")
    return "\n".join(lines)


def write_reports(json_path: Path, markdown_path: Path, report: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        specs = load_prompt_specs(args.prompts, profile=args.profile)
        base_rows = read_jsonl(args.base_ranked)
        adapter_rows = read_jsonl(args.adapter_ranked)
        report = evaluate_gate(
            base_rows,
            adapter_rows,
            specs,
            profile=args.profile,
            base_label=args.base_label,
            adapter_label=args.adapter_label,
        )
        report["inputs"] = {
            "prompts": input_record(args.prompts),
            "base_ranked": input_record(args.base_ranked, len(base_rows)),
            "adapter_ranked": input_record(args.adapter_ranked, len(adapter_rows)),
        }
        write_reports(args.output_json, args.output_md, report)
        print(json.dumps({"decision": report["decision"], "failed_gates": report["failed_gates"]}))
        return 0 if report["passed"] else 1
    except (OSError, GateInputError) as exc:
        report = {
            "schema_version": 1,
            "profile": args.profile,
            "decision": "error",
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_reports(args.output_json, args.output_md, report)
        print(f"balanced quality gate input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
