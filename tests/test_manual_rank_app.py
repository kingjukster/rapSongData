from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManualRankAppTests(unittest.TestCase):
    def test_build_manual_rank_app_embeds_disagreement_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            judge_dir = tmp_path / "judge"
            judge_dir.mkdir()
            (judge_dir / "top_100_disagreements.md").write_text(
                "\n".join(
                    [
                        "# Auto-Judge Disagreement Queue",
                        "",
                        "## 1. cand-a",
                        "## 2. cand-b",
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "candidate_id": "cand-a",
                    "prompt": "Write exactly 12 lines about discipline.",
                    "lyrics": "line one\nline two",
                    "quality_score": 0.7,
                    "combined_quality_score": 3.8,
                    "quality_tags": ["low_rhyme_density"],
                    "judge": {
                        "overall_quality": 4,
                        "usable_as_is": "yes",
                        "main_issue": "low_rhyme",
                        "short_reason": "Good but rhyme-light.",
                        "dimension_scores": {},
                    },
                    "structural_metrics": {"line_count": 12, "target_line_count": 12, "structural_pass": True},
                },
                {
                    "candidate_id": "cand-b",
                    "prompt": "Write exactly 12 lines about patience.",
                    "lyrics": "line one\nline two",
                    "quality_score": 0.5,
                    "combined_quality_score": 3.1,
                    "quality_tags": ["weak_imagery"],
                    "judge": {
                        "overall_quality": 3,
                        "usable_as_is": "no",
                        "main_issue": "weak_imagery",
                        "short_reason": "Needs more concrete images.",
                        "dimension_scores": {},
                    },
                    "structural_metrics": {"line_count": 12, "target_line_count": 12, "structural_pass": True},
                },
            ]
            (judge_dir / "judged_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            out = tmp_path / "manual.html"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_manual_rank_app.py"),
                    "--judge-dir",
                    str(judge_dir),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            html = out.read_text(encoding="utf-8")
            self.assertIn("Manual Candidate Ranking", html)
            self.assertIn("cand-a", html)
            self.assertIn("cand-b", html)
            self.assertIn("Export JSON", html)

    def test_build_manual_rank_app_can_select_auto_keep_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            judge_dir = tmp_path / "judge"
            judge_dir.mkdir()
            rows = []
            for candidate_id, bucket, combined in [
                ("keep-high", "auto_keep", 4.2),
                ("reject-one", "auto_reject", 2.1),
                ("keep-low", "auto_keep", 3.8),
            ]:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "prompt": f"Write exactly 12 lines about {candidate_id}.",
                        "lyrics": "line one\nline two",
                        "quality_score": 0.75,
                        "combined_quality_score": combined,
                        "confidence_bucket": bucket,
                        "quality_tags": [],
                        "judge": {
                            "overall_quality": 4,
                            "usable_as_is": "yes",
                            "main_issue": "other",
                            "short_reason": "Bucket test row.",
                            "dimension_scores": {},
                        },
                        "structural_metrics": {"line_count": 12, "target_line_count": 12, "structural_pass": True},
                    }
                )
            (judge_dir / "judged_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            out = tmp_path / "auto_keep.html"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_manual_rank_app.py"),
                    "--judge-dir",
                    str(judge_dir),
                    "--source",
                    "auto_keep",
                    "--limit",
                    "2",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            html = out.read_text(encoding="utf-8")
            self.assertIn("keep-high", html)
            self.assertIn("keep-low", html)
            self.assertNotIn("reject-one", html)
            self.assertIn("manual_rank_auto_keep_results", html)
            self.assertIn("auto_keep", html)


if __name__ == "__main__":
    unittest.main()
