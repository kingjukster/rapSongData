from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_qwen3_sweep_sft_gpu_max import assess_training_attempt


class Qwen3SweepRunnerStatusTests(unittest.TestCase):
    def assess(self, summary_status: str | None, *, returncode: int, oom: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            if summary_status is not None:
                (output_dir / "training_summary.json").write_text(
                    json.dumps({"result": {"status": summary_status}}),
                    encoding="utf-8",
                )
            return assess_training_attempt(
                result={"returncode": returncode},
                config={"output_dir": str(output_dir)},
                oom_detected=oom,
            )

    def test_zero_exit_requires_complete_full_budget_summary(self):
        assessment = self.assess("complete_full_budget", returncode=0)

        self.assertEqual(assessment["status"], "complete_full_budget")
        self.assertEqual(assessment["exit_code"], 0)
        self.assertTrue(assessment["success"])
        self.assertFalse(assessment["oom_fallback_eligible"])

    def test_partial_and_incomplete_summaries_propagate_as_nonzero(self):
        for status in ["partial_time_budget_reached", "incomplete"]:
            with self.subTest(status=status):
                assessment = self.assess(status, returncode=0)

                self.assertEqual(assessment["status"], status)
                self.assertEqual(assessment["training_summary_status"], status)
                self.assertNotEqual(assessment["exit_code"], 0)
                self.assertFalse(assessment["success"])
                self.assertFalse(assessment["oom_fallback_eligible"])

    def test_zero_exit_without_summary_is_nonzero_and_cannot_fallback(self):
        assessment = self.assess(None, returncode=0, oom=True)

        self.assertEqual(assessment["status"], "missing_training_summary")
        self.assertEqual(assessment["exit_code"], 2)
        self.assertFalse(assessment["oom_fallback_eligible"])

    def test_only_failed_oom_attempt_is_fallback_eligible(self):
        assessment = self.assess(None, returncode=137, oom=True)

        self.assertEqual(assessment["status"], "failed")
        self.assertEqual(assessment["exit_code"], 137)
        self.assertTrue(assessment["oom_fallback_eligible"])

    def test_nonzero_exit_cannot_succeed_even_with_complete_summary(self):
        assessment = self.assess("complete_full_budget", returncode=1)

        self.assertEqual(assessment["status"], "failed")
        self.assertEqual(assessment["training_summary_status"], "complete_full_budget")
        self.assertEqual(assessment["exit_code"], 1)
        self.assertFalse(assessment["success"])

    def test_non_oom_failure_is_not_fallback_eligible(self):
        assessment = self.assess(None, returncode=1, oom=False)

        self.assertFalse(assessment["oom_fallback_eligible"])


if __name__ == "__main__":
    unittest.main()
