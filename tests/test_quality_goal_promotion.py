from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_quality_goal_promotion.py"


def scored_row(
    row_id: str,
    *,
    exact: bool = True,
    slur: bool = False,
    prompt_leakage: bool = False,
    high_copy: bool = False,
    incomplete: bool = False,
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "quality_tags": tags or [],
        "structural_metrics": {
            "exact_line_match": exact,
            "slur_count": int(slur),
            "prompt_leakage": prompt_leakage,
            "high_copy_similarity": high_copy,
            "incomplete_ending": incomplete,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class QualityGoalPromotionTests(unittest.TestCase):
    def run_gate(
        self,
        directory: Path,
        base_rows: list[dict[str, object]],
        candidate_rows: list[dict[str, object]],
        preferences: list[dict[str, object]],
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        base = directory / "base.jsonl"
        candidate = directory / "candidate.jsonl"
        preference_path = directory / "preferences.json"
        out = directory / "promotion.json"
        write_jsonl(base, base_rows)
        write_jsonl(candidate, candidate_rows)
        preference_path.write_text(json.dumps({"preferences": preferences}), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base-scored",
                str(base),
                "--candidate-scored",
                str(candidate),
                "--preferences",
                str(preference_path),
                "--candidate-label",
                "adapter-e2",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result, json.loads(out.read_text(encoding="utf-8"))

    def test_passes_when_all_locked_gates_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base_rows = [
                # Alias spellings from the ranker are normalized into the locked names.
                scored_row(f"row-{index}", tags=["generic_motivation"])
                for index in range(20)
            ]
            candidate_rows = [
                scored_row(
                    f"row-{index}",
                    tags=["generic_motivation"] if index < 16 else [],
                )
                for index in range(20)
            ]
            preferences = [{"row_id": f"row-{index}", "winner": "adapter-e2"} for index in range(20)]

            result, report = self.run_gate(directory, base_rows, candidate_rows, preferences)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(report["passed"])
            self.assertEqual(report["decision"], "pass")
            self.assertEqual(report["failed_gates"], [])
            self.assertGreater(report["preferences"]["wilson_95_lower_bound"], 0.5)
            self.assertEqual(report["pairing"]["paired_row_count"], 20)
            self.assertAlmostEqual(
                report["gates"]["target_issue_burden"]["relative_improvement"],
                0.2,
            )

    def test_fails_with_evidence_for_each_regression_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base_rows = [scored_row(f"row-{index}", tags=["weak_imagery"]) for index in range(100)]
            candidate_rows = []
            for index in range(100):
                tags: list[str] = []
                if index < 95:
                    tags.append("weak_imagery")
                if index < 4:
                    tags.append("generic_motivation")
                candidate_rows.append(
                    scored_row(
                        f"row-{index}",
                        exact=index >= 3,
                        slur=index == 0,
                        prompt_leakage=index == 1,
                        high_copy=index == 2,
                        incomplete=index < 2,
                        tags=tags,
                    )
                )
            preferences = [
                {"row_id": f"row-{index}", "winner": "adapter-e2" if index < 52 else "base"}
                for index in range(100)
            ]

            result, report = self.run_gate(directory, base_rows, candidate_rows, preferences)

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(report["passed"])
            self.assertEqual(
                set(report["failed_gates"]),
                {
                    "preference_wilson_lcb",
                    "exact_line_match_noninferiority",
                    "no_new_row_level_safety_failures",
                    "incomplete_ending_noninferiority",
                    "target_issue_burden",
                    "individual_issue_noninferiority",
                },
            )
            safety = report["gates"]["no_new_row_level_safety_failures"]
            self.assertEqual(safety["new_failure_row_ids"]["slur"], ["row-0"])
            self.assertEqual(safety["new_failure_row_ids"]["prompt_leakage"], ["row-1"])
            self.assertEqual(safety["new_failure_row_ids"]["high_copy"], ["row-2"])
            self.assertAlmostEqual(
                report["gates"]["individual_issue_noninferiority"]["issue_rate_deltas"]["generic"],
                0.04,
            )

    def test_rejects_nonidentical_paired_row_id_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base_rows = [scored_row("shared"), scored_row("base-only")]
            candidate_rows = [scored_row("shared"), scored_row("candidate-only")]
            preferences = [{"winner": "adapter-e2"}]

            result, report = self.run_gate(directory, base_rows, candidate_rows, preferences)

            self.assertEqual(result.returncode, 2)
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"]["type"], "PromotionInputError")
            self.assertIn("missing_from_candidate=['base-only']", report["error"]["message"])
            self.assertIn("extra_in_candidate=['candidate-only']", report["error"]["message"])


if __name__ == "__main__":
    unittest.main()
