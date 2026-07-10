from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManualRankSummaryTests(unittest.TestCase):
    def test_summarizes_manual_rank_export(self):
        rows = [
            {
                "manual_rank": 1,
                "candidate_id": "keep-a",
                "manual_rating": 5,
                "decision": "keep",
                "judge_quality": 4,
                "judge_issue": "scene_drift",
                "heuristic_score": 0.5,
                "combined_score": 3.5,
                "notes": "",
                "prompt": "Prompt one",
            },
            {
                "manual_rank": 2,
                "candidate_id": "edit-a",
                "manual_rating": 3,
                "decision": "edit",
                "judge_quality": 4,
                "judge_issue": "low_rhyme",
                "heuristic_score": 0.4,
                "combined_score": 3.4,
                "notes": "needs sharper rhymes",
                "prompt": "Prompt two",
            },
            {
                "manual_rank": 3,
                "candidate_id": "drop-a",
                "manual_rating": 2,
                "decision": "drop",
                "judge_quality": 4,
                "judge_issue": "low_rhyme",
                "heuristic_score": 0.8,
                "combined_score": 3.8,
                "notes": "too generic",
                "prompt": "Prompt three",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "manual_rank_results.json"
            out_json = tmp_path / "manual_rank_summary.json"
            out_md = tmp_path / "manual_rank_summary.md"
            source.write_text(json.dumps(rows), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_manual_rank_results.py"),
                    "--input",
                    str(source),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--top-n",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_candidates"], 3)
            self.assertEqual(summary["reviewed_candidates"], 3)
            self.assertEqual(summary["decision_counts"], {"drop": 1, "edit": 1, "keep": 1})
            self.assertEqual(summary["manual_rating_counts"], {"2": 1, "3": 1, "5": 1})
            self.assertEqual(summary["by_judge_issue"]["low_rhyme"]["count"], 2)
            self.assertEqual(summary["by_judge_issue"]["low_rhyme"]["decision_counts"], {"drop": 1, "edit": 1})
            self.assertEqual(summary["manual_positive_seeds"][0]["candidate_id"], "keep-a")
            self.assertEqual(summary["manual_negative_seeds"][0]["candidate_id"], "drop-a")
            self.assertEqual(summary["judge_disagreements"][0]["candidate_id"], "drop-a")

            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Manual Rank Summary", markdown)
            self.assertIn("keep-a", markdown)
            self.assertIn("Keep these candidates in the review queue", markdown)


if __name__ == "__main__":
    unittest.main()
