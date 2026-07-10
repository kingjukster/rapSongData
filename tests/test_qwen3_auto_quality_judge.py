from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.judge_qwen3_quality_openai import apply_manual_calibration, confidence_bucket, ranking_score  # noqa: E402


def ranked_row(row_id: str, prompt: str, score: float, tags: list[str]) -> dict:
    return {
        "candidate_id": row_id,
        "row_id": row_id,
        "candidate_index": int(row_id.rsplit("-", 1)[-1]),
        "prompt_key": prompt,
        "prompt": prompt,
        "lyrics": "\n".join(f"{row_id} line {index} with concrete room lights" for index in range(1, 13)),
        "quality_score": score,
        "quality_tags": tags,
        "structural_metrics": {
            "structural_pass": True,
            "line_count": 12,
            "target_line_count": 12,
            "slur_count": 0,
            "prompt_leakage": False,
            "incomplete_ending": False,
            "copy_similarity": 0.0,
        },
    }


class Qwen3AutoQualityJudgeTests(unittest.TestCase):
    def calibrated_row(self, score: float, issue: str, *, usable: str = "yes", judge_quality: int = 4) -> dict:
        row = ranked_row("c-1", "Write exactly 12 lines about a crowded bus ride.", score, [])
        row["judge"] = {
            "overall_quality": judge_quality,
            "usable_as_is": usable,
            "main_issue": issue,
            "short_reason": "Test row.",
            "dimension_scores": {},
        }
        row["heuristic_5"] = round(1 + 4 * score, 4)
        row["combined_quality_score"] = round(0.45 * row["heuristic_5"] + 0.55 * judge_quality, 4)
        row["confidence_bucket"] = confidence_bucket(row)
        return apply_manual_calibration(row)

    def test_manual_calibration_preserves_auto_keep_and_penalizes_needs_review(self):
        auto_keep = self.calibrated_row(0.75, "weak_imagery")
        low_rhyme_review = self.calibrated_row(0.54, "low_rhyme")
        weak_imagery_review = self.calibrated_row(0.54, "weak_imagery")
        neutral_review = self.calibrated_row(0.54, "weak_payoff")

        self.assertEqual(auto_keep["confidence_bucket"], "auto_keep")
        self.assertEqual(auto_keep["manual_calibration"]["penalty"], 0.0)
        self.assertEqual(ranking_score(auto_keep), auto_keep["combined_quality_score"])

        self.assertEqual(low_rhyme_review["confidence_bucket"], "needs_review")
        self.assertGreater(low_rhyme_review["manual_calibration"]["penalty"], weak_imagery_review["manual_calibration"]["penalty"])
        self.assertGreater(weak_imagery_review["manual_calibration"]["penalty"], neutral_review["manual_calibration"]["penalty"])
        self.assertLess(ranking_score(low_rhyme_review), low_rhyme_review["combined_quality_score"])

    def test_mock_auto_judge_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ranked = tmp_path / "ranked.jsonl"
            output_dir = tmp_path / "judge"
            prompt_a = "Write exactly 12 lines about a first show in a half-empty room."
            prompt_b = "Write exactly 12 lines about walking home under elevated train tracks."
            rows = [
                ranked_row("a-1", prompt_a, 0.82, []),
                ranked_row("a-2", prompt_a, 0.45, ["weak_imagery"]),
                ranked_row("b-1", prompt_b, 0.78, ["low_rhyme_density"]),
                ranked_row("b-2", prompt_b, 0.52, ["generic_motivation"]),
            ]
            ranked.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "judge_qwen3_quality_openai.py"),
                    "--input-ranked",
                    str(ranked),
                    "--output-dir",
                    str(output_dir),
                    "--top-count",
                    "4",
                    "--stratified-sample",
                    "0",
                    "--per-prompt-top-k",
                    "2",
                    "--top-review-count",
                    "3",
                    "--mock-judge",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((output_dir / "quality_judge_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["ranker"], "qwen3_4b_base_12line_v1_auto_quality_judge_v1")
            self.assertEqual(metrics["judged_candidates"], 4)
            self.assertGreaterEqual(metrics["per_prompt_winner_count"], 2)
            self.assertTrue((output_dir / "top_100_auto_judged.md").exists())
            self.assertTrue((output_dir / "top_100_disagreements.md").exists())
            self.assertTrue((output_dir / "per_prompt_winners.md").exists())
            judged = [json.loads(line) for line in (output_dir / "judged_candidates.jsonl").read_text().splitlines()]
            self.assertTrue(all("judge" in row for row in judged))


if __name__ == "__main__":
    unittest.main()
