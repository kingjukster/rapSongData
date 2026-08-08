from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scripts import evaluate_balanced_quality_gate as gate


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROMPTS = ROOT / "configs" / "prompts" / "qwen3_4b_12line_balanced_smoke_prompts.json"
EXPANDED_PROMPTS = ROOT / "configs" / "prompts" / "qwen3_4b_12line_expanded_eval_prompts.json"
CONFIRMATION_PROMPTS = ROOT / "configs" / "prompts" / "qwen3_4b_12line_router_confirmation_prompts.json"
SCRIPT = ROOT / "scripts" / "evaluate_balanced_quality_gate.py"


def prompt_payload(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ranked_rows(
    prompts: list[dict[str, Any]],
    *,
    score_for: Callable[[int, dict[str, Any]], float],
    exact_for: Callable[[int, dict[str, Any]], bool] | None = None,
    failures_for: Callable[[int, dict[str, Any]], dict[str, bool]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, prompt in enumerate(prompts):
        failures = failures_for(index, prompt) if failures_for else {}
        rows.append(
            {
                "prompt": prompt["prompt"],
                "theme": prompt["theme"],
                "prompt_family": prompt["prompt_family"],
                "quality_score": score_for(index, prompt),
                "structural_metrics": {
                    "exact_line_match": exact_for(index, prompt) if exact_for else True,
                    "slur_count": int(failures.get("slur", False)),
                    "prompt_leakage": failures.get("prompt_leakage", False),
                    "repeated_line_ratio": 0.25 if failures.get("repetition", False) else 0.0,
                    "incomplete_ending": failures.get("incomplete_ending", False),
                    "high_copy_similarity": failures.get("high_copy", False),
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BalancedQualityGateTests(unittest.TestCase):
    def test_weak_family_probe_accepts_story_and_clean_only(self) -> None:
        prompts = [
            row for row in prompt_payload(ROOT / "configs" / "prompts" / "qwen3_4b_12line_v5_confirmation_prompts.json")
            if gate.normalize_family(row["prompt_family"]) in {"story", "clean"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "weak.json"
            prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
            specs = gate.load_prompt_specs(prompt_path, profile="weak_family_probe")
        base = ranked_rows(prompts, score_for=lambda _index, _row: 0.5)
        adapter = ranked_rows(prompts, score_for=lambda _index, _row: 0.52)
        report = gate.evaluate_gate(
            base, adapter, specs, profile="weak_family_probe", base_label="base", adapter_label="v51"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["families"]), {"story", "clean"})

    def test_confirmation_asset_and_profile_are_balanced_and_locked(self) -> None:
        prompts = prompt_payload(CONFIRMATION_PROMPTS)
        specs = gate.load_prompt_specs(CONFIRMATION_PROMPTS, profile="confirmation")
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len(specs), 24)
        self.assertEqual(Counter(row["theme"] for row in prompts), Counter({theme: 4 for theme in {
            row["theme"] for row in prompts
        }}))
        self.assertEqual(
            Counter(gate.normalize_family(row["prompt_family"]) for row in prompts),
            Counter({family: 6 for family in gate.FAMILY_ORDER}),
        )
        self.assertEqual(gate.PROFILES["confirmation"], {
            "expected_prompt_count": 24,
            "expected_theme_count": 6,
            "minimum_exact_line_count": 24,
            "minimum_average_quality_delta": 0.01,
            "minimum_paired_wins": 14,
            "minimum_family_quality_delta": -0.01,
        })

    def test_smoke_asset_is_balanced_deterministic_expanded_subset(self) -> None:
        smoke = prompt_payload(SMOKE_PROMPTS)
        expanded = prompt_payload(EXPANDED_PROMPTS)
        expected_source_indices = [
            1,
            2,
            3,
            4,
            21,
            22,
            23,
            24,
            41,
            42,
            43,
            44,
            57,
            58,
            59,
            60,
        ]

        self.assertEqual([row["source_prompt_index"] for row in smoke], expected_source_indices)
        self.assertEqual(len({row["prompt_key"] for row in smoke}), 16)
        for row in smoke:
            source = expanded[row["source_prompt_index"] - 1]
            for field in ("prompt", "theme", "style", "target_line_count", "prompt_family"):
                self.assertEqual(row[field], source[field])
        self.assertEqual(Counter(row["theme"] for row in smoke), Counter({theme: 4 for theme in {
            row["theme"] for row in smoke
        }}))
        self.assertEqual(
            Counter(gate.normalize_family(row["prompt_family"]) for row in smoke),
            Counter({family: 4 for family in gate.FAMILY_ORDER}),
        )

    def test_smoke_gate_passes_at_boundaries_and_filters_full_base_run(self) -> None:
        smoke_payload = prompt_payload(SMOKE_PROMPTS)
        expanded_payload = prompt_payload(EXPANDED_PROMPTS)
        specs = gate.load_prompt_specs(SMOKE_PROMPTS, profile="smoke")
        family_scores = {
            "melodic": 0.52,
            "story": 0.51,
            "technical": 0.481,
            "clean": 0.509,
        }
        base_rows = ranked_rows(expanded_payload, score_for=lambda _index, _row: 0.5)
        adapter_rows = ranked_rows(
            smoke_payload,
            score_for=lambda _index, row: family_scores[gate.normalize_family(row["prompt_family"])],
            exact_for=lambda index, _row: index != 15,
        )
        # Prompt metadata is authoritative because legacy ranker inference can
        # mislabel families when a theme contains a family keyword.
        adapter_rows[2]["prompt_family"] = "clean"

        report = gate.evaluate_gate(
            base_rows,
            adapter_rows,
            specs,
            profile="smoke",
            base_label="base",
            adapter_label="v3",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_gates"], [])
        self.assertEqual(report["selection"]["base"]["input_row_count"], 60)
        self.assertEqual(report["selection"]["base"]["ignored_row_count"], 44)
        self.assertEqual(report["metrics"]["adapter"]["exact_line_match_count"], 15)
        self.assertAlmostEqual(report["families"]["technical"]["adapter_minus_base_quality"], -0.019)
        self.assertEqual(report["paired"], {"prompt_count": 16, "adapter_wins": 12, "base_wins": 4, "ties": 0})

    def test_smoke_cli_fails_and_reports_each_locked_regression(self) -> None:
        smoke_payload = prompt_payload(SMOKE_PROMPTS)
        failure_names = list(gate.HARD_FAILURE_FIELDS)

        def failures(index: int, _row: dict[str, Any]) -> dict[str, bool]:
            return {failure_names[index]: True} if index < len(failure_names) else {}

        family_scores = {"melodic": 0.5, "story": 0.5, "technical": 0.47, "clean": 0.49}
        base_rows = ranked_rows(smoke_payload, score_for=lambda _index, _row: 0.5)
        adapter_rows = ranked_rows(
            smoke_payload,
            score_for=lambda _index, row: family_scores[gate.normalize_family(row["prompt_family"])],
            exact_for=lambda index, _row: index >= 2,
            failures_for=failures,
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base_path = directory / "base.jsonl"
            adapter_path = directory / "adapter.jsonl"
            output_json = directory / "gate.json"
            output_md = directory / "gate.md"
            write_jsonl(base_path, base_rows)
            write_jsonl(adapter_path, adapter_rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base-ranked",
                    str(base_path),
                    "--adapter-ranked",
                    str(adapter_path),
                    "--prompts",
                    str(SMOKE_PROMPTS),
                    "--profile",
                    "smoke",
                    "--adapter-label",
                    "v3",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            report = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            set(report["failed_gates"]),
            {"exact_line_compliance", "zero_hard_failures", "average_quality_delta", "family_quality_floor"},
        )
        self.assertEqual(
            report["metrics"]["adapter"]["hard_failure_counts"],
            {name: 1 for name in gate.HARD_FAILURE_FIELDS},
        )
        self.assertEqual(report["gates"]["family_quality_floor"]["failing_families"], ["technical"])
        self.assertIn("Decision: **FAIL**", markdown)

    def test_promotion_profile_enforces_59_exact_36_wins_and_nonnegative_families(self) -> None:
        expanded_payload = prompt_payload(EXPANDED_PROMPTS)
        specs = gate.load_prompt_specs(EXPANDED_PROMPTS, profile="promotion")
        base_rows = ranked_rows(expanded_payload, score_for=lambda _index, _row: 0.5)
        passing_rows = ranked_rows(
            expanded_payload,
            score_for=lambda _index, _row: 0.52,
            exact_for=lambda index, _row: index != 59,
        )
        passing = gate.evaluate_gate(
            base_rows,
            passing_rows,
            specs,
            profile="promotion",
            base_label="base",
            adapter_label="v3",
        )

        self.assertTrue(passing["passed"])
        self.assertEqual(passing["metrics"]["adapter"]["exact_line_match_count"], 59)
        self.assertEqual(passing["paired"]["adapter_wins"], 60)

        only_35_wins = ranked_rows(
            expanded_payload,
            score_for=lambda index, _row: 0.535 if index < 35 else 0.5,
            exact_for=lambda index, _row: index != 59,
        )
        failing = gate.evaluate_gate(
            base_rows,
            only_35_wins,
            specs,
            profile="promotion",
            base_label="base",
            adapter_label="v3",
        )

        self.assertFalse(failing["passed"])
        self.assertEqual(failing["failed_gates"], ["paired_wins"])
        self.assertGreaterEqual(failing["metrics"]["adapter_minus_base_average_quality"], 0.02)
        self.assertTrue(all(row["adapter_minus_base_quality"] >= 0 for row in failing["families"].values()))


if __name__ == "__main__":
    unittest.main()
