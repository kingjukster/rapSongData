from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManualRankAppTests(unittest.TestCase):
    def test_review_target_fingerprint_changes_with_rubric_or_app_source(self):
        from scripts.build_manual_rank_app import review_target_fingerprint

        candidates = [{"candidate_id": "one", "normalized_text_sha256": "abc", "prompt_key": "p1"}]
        base = review_target_fingerprint(
            candidates,
            rubric_version="v1",
            review_session_id="session",
            app_version="app-v1",
            app_source_hash="source-a",
        )
        changed = review_target_fingerprint(
            candidates,
            rubric_version="v2",
            review_session_id="session",
            app_version="app-v1",
            app_source_hash="source-a",
        )
        changed_source = review_target_fingerprint(
            candidates,
            rubric_version="v1",
            review_session_id="session",
            app_version="app-v1",
            app_source_hash="source-b",
        )

        self.assertNotEqual(base, changed)
        self.assertNotEqual(base, changed_source)

    def test_legacy_rng_provenance_distinguishes_batch_and_retry_seeds(self):
        from scripts.build_quality_goal_review_queue import generation_rng_provenance

        summary = {"settings": {"seed": 100}}
        batch = generation_rng_provenance(
            {
                "candidate_index": 66,
                "seed": 165,
                "generation_attempt_count": 1,
                "accepted_attempt_index": 1,
                "timing": {"batch_size": 64},
            },
            summary,
        )
        retry = generation_rng_provenance(
            {
                "candidate_index": 66,
                "seed": 100_168,
                "initial_seed": 165,
                "generation_attempt_count": 2,
                "accepted_attempt_index": 2,
                "timing": {"batch_size": 1},
            },
            summary,
        )

        self.assertEqual(batch["manual_seed"], 164)
        self.assertEqual(batch["row_offset_within_batch"], 1)
        self.assertFalse(batch["exact_per_row_seed"])
        self.assertEqual(retry["manual_seed"], 100_168)
        self.assertTrue(retry["exact_per_row_seed"])

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
            self.assertIn("reviewerId", html)
            self.assertIn("humanAttested", html)
            self.assertIn("dimensionRatings", html)

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

    def test_quality_goal_queue_is_unique_balanced_and_blinded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            judge_dir = tmp_path / "judge"
            output_dir = tmp_path / "queue"
            judge_dir.mkdir()
            rows = []
            for index in range(100):
                prompt_index = index % 10
                rows.append(
                    {
                        "candidate_id": f"candidate-{index:03d}",
                        "candidate_index": index,
                        "sample_index": index,
                        "prompt_key": f"prompt-{prompt_index}",
                        "prompt_family": ["clean", "melodic", "technical", "story"][index % 4],
                        "theme": f"theme-{prompt_index}",
                        "prompt": f"Write exactly 12 lines about theme {prompt_index}.",
                        "lyrics": "\n".join(f"candidate {index} line {line}" for line in range(1, 13)),
                        "quality_score": index / 100,
                        "combined_quality_score": index / 20,
                        "quality_tags": [],
                        "judge": {
                            "overall_quality": 4,
                            "usable_as_is": "yes",
                            "main_issue": "weak_imagery" if index % 2 else "weak_payoff",
                            "short_reason": "Hidden during blinded review.",
                            "dimension_scores": {},
                        },
                        "structural_metrics": {
                            "line_count": 12,
                            "target_line_count": 12,
                            "slur_count": 0,
                            "prompt_leakage": False,
                            "high_copy_similarity": False,
                            "structural_pass": True,
                        },
                    }
                )
            (judge_dir / "judged_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_quality_goal_review_queue.py"),
                    "--judge-dir",
                    str(judge_dir),
                    "--calibrated-dir",
                    str(tmp_path / "missing-calibrated"),
                    "--seed-dir",
                    str(tmp_path / "missing-seed"),
                    "--output-dir",
                    str(output_dir),
                    "--target-count",
                    "100",
                    "--legacy-count",
                    "0",
                    "--curated-unreviewed-count",
                    "0",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            queue = [json.loads(line) for line in (output_dir / "review_queue.jsonl").read_text().splitlines()]
            manifest = json.loads((output_dir / "review_queue_manifest.json").read_text(encoding="utf-8"))
            html = (output_dir / "manual_review.html").read_text(encoding="utf-8")
            self.assertEqual(len(queue), 100)
            self.assertEqual(len({row["candidate_id"] for row in queue}), 100)
            self.assertEqual(manifest["prompt_key_count"], 10)
            self.assertEqual(manifest["status"], "awaiting_human_review")
            self.assertTrue(all(row["provenance"]["normalized_text_sha256"] for row in queue))
            self.assertIn('"blind": true', html)
            self.assertNotIn("Hidden during blinded review.", html)
            self.assertNotIn("weak_imagery", html)
            self.assertIn("human_attested", html)


if __name__ == "__main__":
    unittest.main()
