from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def judged_row(
    candidate_id: str,
    prompt: str,
    *,
    score: float,
    judge_quality: int = 4,
    usable: str = "yes",
    issue: str = "weak_payoff",
    tags: list[str] | None = None,
    lines: int = 12,
    structural_pass: bool = True,
) -> dict:
    lyrics = "\n".join(f"{candidate_id} concrete line {index} moves through rain" for index in range(1, lines + 1))
    return {
        "candidate_id": candidate_id,
        "row_id": candidate_id,
        "prompt_key": prompt.replace(" ", "-"),
        "prompt": prompt,
        "lyrics": lyrics,
        "quality_score": 0.7,
        "quality_tags": tags or [],
        "structural_metrics": {
            "structural_pass": structural_pass,
            "exact_line_match": lines == 12,
            "prompt_leakage": False,
            "incomplete_ending": False,
            "high_copy_similarity": False,
            "copy_similarity": 0.0,
            "slur_count": 0,
        },
        "judge": {
            "overall_quality": judge_quality,
            "usable_as_is": usable,
            "main_issue": issue,
        },
        "combined_quality_score": score,
        "calibrated_review_score": score,
    }


class QualitySftSeedTests(unittest.TestCase):
    def test_builder_writes_strict_seed_pairs_and_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "judged.jsonl"
            out_dir = tmp_path / "seed"
            prompt_a = "Write exactly 12 lines about rain on train tracks."
            prompt_b = "Write exactly 12 lines about a first show."
            rows = [
                judged_row("good-a", prompt_a, score=4.2, issue="weak_payoff"),
                judged_row("bad-a", prompt_a, score=3.1, usable="no", issue="weak_imagery", tags=["weak_imagery"]),
                judged_row("good-b", prompt_b, score=4.0, issue="generic"),
                judged_row("short-b", prompt_b, score=4.1, lines=10),
                judged_row("weak-b", prompt_b, score=4.1, issue="scene_drift", tags=["scene_drift"]),
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_12line_quality_sft_seed.py"),
                    "--input-judged",
                    str(source),
                    "--output-dir",
                    str(out_dir),
                    "--min-calibrated-score",
                    "3.7",
                    "--min-pair-delta",
                    "0.75",
                    "--min-training-ready-sft",
                    "500",
                    "--min-training-ready-pairs",
                    "1000",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            train = [json.loads(line) for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            validation = [
                json.loads(line) for line in (out_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            pairs = [
                json.loads(line) for line in (out_dir / "preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            negatives = [
                json.loads(line) for line in (out_dir / "hard_negatives.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(manifest["status"], "seed_only_not_training_ready")
            self.assertEqual(manifest["counts"]["strict_sft_examples"], 2)
            self.assertTrue(train or validation)
            self.assertTrue(all("weak_imagery" not in row["metadata"]["quality_tags"] for row in train + validation))
            self.assertEqual(len(pairs), 1)
            self.assertGreaterEqual(pairs[0]["metadata"]["score_delta"], 0.75)
            self.assertTrue(any(row["metadata"]["judge_issue"] == "weak_imagery" for row in negatives))


if __name__ == "__main__":
    unittest.main()
