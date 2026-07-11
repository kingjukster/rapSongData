from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def chat_record(row_id: str, prompt: str, lyrics: str, prompt_key: str, split: str = "train") -> dict:
    return {
        "id": row_id,
        "training_text": (
            "<|im_start|>system\nGenerate original rap lyrics.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{lyrics}<|im_end|>\n"
        ),
        "metadata": {
            "prompt_key": prompt_key,
            "split": split,
            "target_line_count": 12,
            "actual_line_count": 12,
            "source_bucket": "test",
            "judge_issue": "weak_payoff",
            "quality_tags": [],
        },
    }


def twelve_lines(prefix: str) -> str:
    return "\n".join(f"{prefix} line {index} moves through concrete rain" for index in range(1, 13))


class TrainingJsonlAuditTests(unittest.TestCase):
    def test_request_aware_targets_do_not_force_every_row_to_twelve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            out = tmp_path / "audit.json"
            four_lines = "\n".join(f"hook line {index}" for index in range(1, 5))
            sixteen_lines = "\n".join(f"verse line {index}" for index in range(1, 17))
            train_rows = [
                chat_record("train-4", "Write a hook with 4 short lines.", four_lines, "prompt-4"),
                chat_record("train-16", "Write exactly 16 bars about patience.", sixteen_lines, "prompt-16"),
            ]
            for row in train_rows:
                row["metadata"].pop("target_line_count")
                row["metadata"].pop("actual_line_count")
            train.write_text("\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8")
            validation.write_text(
                json.dumps(chat_record("validation-1", "Prompt B", twelve_lines("valid"), "prompt-b", "validation"))
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--out",
                    str(out),
                    "--require-explicit-target",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["datasets"]["train"]["target_line_count_counts"], {"4": 1, "16": 1})
            self.assertNotIn("line_count_mismatch", {issue["code"] for issue in report["datasets"]["train"]["issues"]})

    def test_strict_provenance_and_human_review_gates_fail_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            out = tmp_path / "audit.json"
            train.write_text(
                json.dumps(chat_record("train-1", "Prompt A", twelve_lines("train"), "prompt-a")) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(chat_record("validation-1", "Prompt B", twelve_lines("valid"), "prompt-b", "validation"))
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--out",
                    str(out),
                    "--require-provenance",
                    "--require-human-review",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            codes = {issue["code"] for issue in report["datasets"]["train"]["issues"]}
            self.assertIn("missing_provenance", codes)
            self.assertIn("missing_verified_human_review", codes)

    def test_clean_training_package_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            manifest = tmp_path / "manifest.json"
            out = tmp_path / "audit.json"
            train.write_text(
                json.dumps(chat_record("train-1", "Prompt A", twelve_lines("train"), "prompt-a")) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(chat_record("validation-1", "Prompt B", twelve_lines("valid"), "prompt-b", "validation"))
                + "\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps({"name": "test", "counts": {"train_rows": 1, "validation_rows": 1, "total_rows": 2}}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--manifest",
                    str(manifest),
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
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["status"], "pass")

    def test_prompt_and_assistant_overlap_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            out = tmp_path / "audit.json"
            lyrics = twelve_lines("same")
            train.write_text(json.dumps(chat_record("train-1", "Prompt A", lyrics, "prompt-a")) + "\n", encoding="utf-8")
            validation.write_text(
                json.dumps(chat_record("validation-1", "Prompt A", lyrics, "prompt-a", "validation")) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            codes = {issue["code"] for issue in report["cross_split_issues"]}
            self.assertIn("train_validation_prompt_overlap", codes)
            self.assertIn("train_validation_assistant_text_overlap", codes)

    def test_preference_delta_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            validation = tmp_path / "validation.jsonl"
            pairs = tmp_path / "pairs.jsonl"
            out = tmp_path / "audit.json"
            train.write_text(
                json.dumps(chat_record("train-1", "Prompt A", twelve_lines("train"), "prompt-a")) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(chat_record("validation-1", "Prompt B", twelve_lines("valid"), "prompt-b", "validation"))
                + "\n",
                encoding="utf-8",
            )
            pairs.write_text(
                json.dumps(
                    {
                        "chosen": "better",
                        "rejected": "worse",
                        "metadata": {"score_delta": 0.2, "split": "train"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--preference-pairs",
                    str(pairs),
                    "--min-preference-score-delta",
                    "0.75",
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            issues = report["datasets"]["preference_pairs"]["issues"]
            self.assertIn("preference_score_delta_below_minimum", {issue["code"] for issue in issues})


if __name__ == "__main__":
    unittest.main()
