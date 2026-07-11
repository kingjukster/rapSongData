from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.build_manual_rank_app import app_source_sha256, candidate_payload, review_target_fingerprint
from scripts.build_human_12line_sft import split_prompt_groups


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = [
    "theme_adherence",
    "specific_imagery",
    "rhyme_cadence",
    "originality",
    "scene_coherence",
    "naturalness",
    "ending_payoff",
]


def normalized_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Human12LineSftTests(unittest.TestCase):
    def test_same_theme_never_crosses_splits_when_prompt_keys_differ(self) -> None:
        rows = [
            {
                "candidate_id": f"candidate-{theme}-{prompt}",
                "theme": f"theme-{theme}",
                "prompt_key": f"prompt-{theme}-{prompt}",
            }
            for theme in range(10)
            for prompt in range(2)
        ]

        splits = split_prompt_groups(rows, 20260710)
        theme_splits: dict[str, set[str]] = {}
        for split, split_rows in splits.items():
            for row in split_rows:
                theme_splits.setdefault(row["theme"], set()).add(split)

        self.assertTrue(all(len(assigned) == 1 for assigned in theme_splits.values()))

    def test_builder_requires_attested_reviews_and_writes_clean_group_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue_path = tmp_path / "queue.jsonl"
            reviews_path = tmp_path / "reviews.json"
            output_dir = tmp_path / "training"
            queue = []
            reviews = []
            vocab = [
                "alley", "bridge", "copper", "drizzle", "engine", "frost",
                "garden", "harbor", "island", "jacket", "kettle", "lantern",
            ]
            for index in range(12):
                lyrics = "\n".join(
                    f"{vocab[(index + line) % len(vocab)]} scene {index} advances with detail {line}"
                    for line in range(12)
                )
                text_hash = normalized_hash(lyrics)
                candidate_id = f"candidate-{index}"
                queue.append(
                    {
                        "candidate_id": candidate_id,
                        "prompt_key": f"prompt-{index % 6}",
                        "prompt": f"Write exactly 12 lines about scene {index % 6}.",
                        "lyrics": lyrics,
                        "theme": f"theme-{index % 6}",
                        "prompt_family": "story",
                        "structural_metrics": {
                            "slur_count": 0,
                            "prompt_leakage": False,
                            "high_copy_similarity": False,
                        },
                        "provenance": {
                            "candidate_id": candidate_id,
                            "normalized_text_sha256": text_hash,
                            "license_scope": "synthetic_model_generated_local_audit",
                            "judge_source_sha256": "a" * 64,
                            "generation_source_sha256": "b" * 64,
                            "generation_summary_sha256": "d" * 64,
                            "generation_run_id": "test-generation-run",
                            "judge_run_id": "test-judge-run",
                            "model_id": "Qwen/Qwen3-4B",
                            "model_revision": "c" * 40,
                            "model_revision_source": "test_fixture",
                            "rng_provenance": {
                                "protocol": "strict_single_row_seed",
                                "manual_seed": 1000 + index,
                                "exact_per_row_seed": True,
                            },
                            "created_at": "2026-07-10T11:00:00+00:00",
                        },
                    }
                )
                reviews.append(
                    {
                        "review_id": f"qwen3-4b-12line-human-v1-review-001:{candidate_id}",
                        "candidate_id": candidate_id,
                        "reviewer_id": "reviewer-a",
                        "reviewer_type": "human",
                        "label_source": "human_entered",
                        "human_attested": True,
                        "reviewed_at": "2026-07-10T12:00:00Z",
                        "session_id": "qwen3-4b-12line-human-v1-review-001",
                        "rubric_version": "rap_12line_quality_v1",
                        "blinded": True,
                        "manual_rating": 5,
                        "decision": "keep",
                        "dimensions": {name: 4 for name in DIMENSIONS},
                        "issue_tags": [],
                        "notes": "",
                        "edited_lyrics": None,
                        "reviewed_text_sha256": text_hash,
                        "prompt_key": f"prompt-{index % 6}",
                        "prompt": f"Write exactly 12 lines about scene {index % 6}.",
                        "lyrics": lyrics,
                        "review_duration_seconds": 30,
                        "blinded": True,
                    }
                )
            source_hash = app_source_sha256()
            target_fingerprint = review_target_fingerprint(
                [candidate_payload(row, rank=index) for index, row in enumerate(queue, start=1)],
                rubric_version="rap_12line_quality_v1",
                review_session_id="qwen3-4b-12line-human-v1-review-001",
                app_version="manual_rank_v2",
                app_source_hash=source_hash,
            )
            for review in reviews:
                review["app_source_sha256"] = source_hash
                review["review_target_fingerprint"] = target_fingerprint
            queue_path.write_text("\n".join(json.dumps(row) for row in queue) + "\n", encoding="utf-8")
            reviews_path.write_text(json.dumps(reviews), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_human_12line_sft.py"),
                    "--reviews",
                    str(reviews_path),
                    "--queue",
                    str(queue_path),
                    "--output-dir",
                    str(output_dir),
                    "--minimum-eligible",
                    "10",
                    "--near-duplicate-threshold",
                    "0.9999",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "training_ready")
            self.assertEqual(manifest["counts"]["verified_human_unique"], 12)
            self.assertEqual(manifest["prompt_split_overlap"]["train_validation"], 0)
            self.assertEqual(manifest["theme_split_overlap"]["train_validation"], 0)
            self.assertEqual(manifest["theme_split_overlap"]["train_test"], 0)
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "validation.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())

            audit = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_training_jsonl.py"),
                    "--train",
                    str(output_dir / "train.jsonl"),
                    "--validation",
                    str(output_dir / "validation.jsonl"),
                    "--manifest",
                    str(output_dir / "manifest.json"),
                    "--out",
                    str(tmp_path / "audit.json"),
                    "--require-explicit-target",
                    "--require-provenance",
                    "--require-human-review",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(audit.returncode, 0, audit.stderr)

    def test_builder_refuses_unattested_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lyrics = "\n".join(f"line {index} with concrete motion" for index in range(12))
            text_hash = normalized_hash(lyrics)
            queue_path = tmp_path / "queue.jsonl"
            reviews_path = tmp_path / "reviews.json"
            queue_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "candidate",
                        "prompt_key": "prompt",
                        "prompt": "Write exactly 12 lines.",
                        "lyrics": lyrics,
                        "structural_metrics": {},
                        "provenance": {
                            "normalized_text_sha256": text_hash,
                            "license_scope": "synthetic_model_generated_local_audit",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reviews_path.write_text(
                json.dumps([{"candidate_id": "candidate", "human_attested": False}]),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_human_12line_sft.py"),
                    "--reviews",
                    str(reviews_path),
                    "--queue",
                    str(queue_path),
                    "--output-dir",
                    str(tmp_path / "training"),
                    "--minimum-eligible",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Only 0 unique eligible", result.stderr)


if __name__ == "__main__":
    unittest.main()
