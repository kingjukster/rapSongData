from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def candidate(candidate_id: str, prompt: str, score: float, *, lines: int = 12, bucket: str = "usable") -> dict:
    return {
        "candidate_id": candidate_id,
        "selection_bucket": bucket,
        "prompt": prompt,
        "lyrics": "\n".join(f"{candidate_id} line {index} with concrete motion" for index in range(1, lines + 1)),
        "confidence_bucket": "auto_keep" if bucket == "usable" else "needs_review",
        "judge_quality": 4,
        "judge_usable_as_is": "yes",
        "judge_issue": "weak_payoff",
        "combined_quality_score": score,
        "calibrated_review_score": score,
        "calibration_penalty": 0.0,
        "calibration_signals": [],
    }


class Calibrated12LineSftTests(unittest.TestCase):
    def test_builder_writes_prompt_grouped_train_validation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "sets"
            out_dir = tmp_path / "training"
            source_dir.mkdir()
            prompt_a = "Write exactly 12 lines about a train platform."
            prompt_b = "Write exactly 12 lines about family pride."
            prompt_c = "Write exactly 12 lines about public failure."
            usable = [
                candidate("a1", prompt_a, 4.1),
                candidate("a2", prompt_a, 4.0),
                candidate("b1", prompt_b, 3.95),
            ]
            edit = [
                candidate("c1", prompt_c, 3.8, bucket="edit"),
                candidate("skip-short", prompt_c, 3.9, lines=10, bucket="edit"),
            ]
            (source_dir / "usable_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in usable) + "\n",
                encoding="utf-8",
            )
            (source_dir / "edit_candidates.jsonl").write_text(
                "\n".join(json.dumps(row) for row in edit) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_calibrated_12line_sft.py"),
                    "--source-dir",
                    str(source_dir),
                    "--output-dir",
                    str(out_dir),
                    "--usable-repeat",
                    "1",
                    "--edit-repeat",
                    "1",
                    "--validation-ratio",
                    "0.25",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            train = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (out_dir / "validation.jsonl").read_text().splitlines()]
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["counts"]["total_rows"], 4)
            self.assertEqual(manifest["counts"]["skipped_rows"], 1)
            self.assertEqual(manifest["counts"]["line_count_errors"], 0)
            self.assertEqual(manifest["counts"]["prompt_split_leak_count"], 0)
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertTrue(all("training_text" in row for row in train + validation))
            self.assertIn("<|im_start|>assistant", train[0]["training_text"])
            self.assertTrue((out_dir / "preview.md").exists())


if __name__ == "__main__":
    unittest.main()
