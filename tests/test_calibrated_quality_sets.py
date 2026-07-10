from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def judged_row(candidate_id: str, score: float, bucket: str, issue: str, usable: str = "yes") -> dict:
    judge_quality = 4 if usable == "yes" else 3
    heuristic_5 = round(1 + 4 * score, 4)
    return {
        "candidate_id": candidate_id,
        "prompt": "Write exactly 12 lines about a quiet train platform.",
        "lyrics": f"{candidate_id} line one\n{candidate_id} line two",
        "quality_score": score,
        "heuristic_5": heuristic_5,
        "combined_quality_score": round(0.45 * heuristic_5 + 0.55 * judge_quality, 4),
        "confidence_bucket": bucket,
        "quality_tags": [],
        "judge": {
            "overall_quality": judge_quality,
            "usable_as_is": usable,
            "main_issue": issue,
            "short_reason": "Test candidate.",
            "dimension_scores": {},
        },
    }


class CalibratedQualitySetTests(unittest.TestCase):
    def test_exporter_writes_usable_edit_reject_and_manual_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            judge_dir = tmp_path / "judge"
            out_dir = tmp_path / "sets"
            judge_dir.mkdir()
            rows = [
                judged_row("auto-keep", 0.76, "auto_keep", "weak_imagery"),
                judged_row("needs-edit", 0.70, "needs_review", "weak_payoff"),
                judged_row("low-review", 0.46, "needs_review", "low_rhyme"),
                judged_row("auto-reject", 0.20, "auto_reject", "weak_imagery", usable="no"),
            ]
            (judge_dir / "judged_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            manual = [
                {
                    "candidate_id": "manual-good",
                    "prompt": "same prompt",
                    "lyrics": "strong selected verse",
                    "manual_rating": 4,
                    "decision": "keep",
                },
                {
                    "candidate_id": "manual-bad",
                    "prompt": "same prompt",
                    "lyrics": "weak rejected verse",
                    "manual_rating": 2,
                    "decision": "drop",
                },
            ]
            (judge_dir / "manual_rank_results.json").write_text(json.dumps(manual), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_calibrated_quality_sets.py"),
                    "--judge-dir",
                    str(judge_dir),
                    "--out-dir",
                    str(out_dir),
                    "--edit-threshold",
                    "3.70",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            usable = [json.loads(line) for line in (out_dir / "usable_candidates.jsonl").read_text().splitlines()]
            edit = [json.loads(line) for line in (out_dir / "edit_candidates.jsonl").read_text().splitlines()]
            reject = [json.loads(line) for line in (out_dir / "reject_candidates.jsonl").read_text().splitlines()]
            pairs = [json.loads(line) for line in (out_dir / "manual_preference_pairs.jsonl").read_text().splitlines()]
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual([row["candidate_id"] for row in usable], ["auto-keep"])
            self.assertEqual([row["candidate_id"] for row in edit], ["needs-edit"])
            self.assertIn("auto-reject", [row["candidate_id"] for row in reject])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["metadata"]["chosen_candidate_id"], "manual-good")
            self.assertEqual(summary["usable_count"], 1)
            self.assertTrue((out_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
