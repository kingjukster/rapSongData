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

from rap_song_data.cli import _build_mutation_dataset, _build_preference_dataset  # noqa: E402
from scripts.audit_rap_datasets import audit_preferences  # noqa: E402
from scripts.evaluate_generation_outputs import has_prompt_leakage  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_row(song_id: str, section_id: str, index: int, text: str, quality: float, split: str = "train") -> dict:
    return {
        "song_id": song_id,
        "section_id": section_id,
        "bar_id": f"{section_id}_{index}",
        "bar_index": index,
        "clean_bar_text": text,
        "quality_score": quality,
        "source_license": "synthetic_transformed",
        "split": split,
        "section_type": "verse",
        "emotion_tags": ["focused"],
    }


def full_pipeline_row(song_id: str, section_id: str, index: int, text: str, quality: float) -> dict:
    row = base_row(song_id, section_id, index, text, quality)
    row.update(
        {
            "artist_clean": "Test Artist",
            "year": 2026,
            "era": "modern",
            "rap_family": "Lyrical",
            "rap_category": "Conscious Rap",
            "energy_level": "medium",
            "density_level": "medium",
            "syllable_count": 11,
            "rhyme_group": "A",
            "target_completion": text,
            "theme_tags": ["discipline"],
        }
    )
    return row


class RapQualityToolTests(unittest.TestCase):
    def test_prompt_leakage_detector_ignores_lyric_cant(self):
        self.assertFalse(
            has_prompt_leakage(
                "I can't erase the late-night pressure\n"
                "but I keep every promise when the city moves fast"
            )
        )
        self.assertTrue(has_prompt_leakage("Here are 12 lines:\nI keep the pressure low"))
        self.assertTrue(has_prompt_leakage("Verse:\nI keep the pressure low"))

    def test_preference_builder_drops_identical_and_equal_quality_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preferences.jsonl"
            rows = [
                base_row("s1", "sec1", 0, "same line", 1.0),
                base_row("s1", "sec1", 1, "same line", 1.0),
                base_row("s2", "sec2", 0, "weaker line with less detail", 0.4),
                base_row("s2", "sec2", 1, "stronger line with sharper detail", 0.9),
                base_row("s3", "sec3", 0, "slightly weaker line with enough words", 0.72),
                base_row("s3", "sec3", 1, "slightly stronger line with enough words", 0.8),
            ]

            count = _build_preference_dataset(rows, out)

            self.assertEqual(count, 1)
            payload = read_jsonl(out)[0]
            self.assertNotEqual(payload["chosen"], payload["rejected"])
            self.assertGreater(payload["metadata"]["quality_chosen"], payload["metadata"]["quality_rejected"])
            self.assertNotIn("[filler]", payload["rejected"])

    def test_mutation_builder_controls_match_actual_output_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mutation.jsonl"
            rows = [
                base_row("s1", "sec1", index, f"line {index} with enough words", 0.8)
                for index in range(8)
            ]

            count = _build_mutation_dataset(rows, out)

            self.assertGreater(count, 0)
            for payload in read_jsonl(out):
                self.assertEqual(payload["controls"]["output_bars"], len(payload["output_bars"]))
                self.assertEqual(payload["metadata"]["output_bar_count"], len(payload["output_bars"]))

    def test_dataset_audit_fails_on_hard_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generation = tmp_path / "generation.jsonl"
            mutation = tmp_path / "mutation.jsonl"
            preferences = tmp_path / "preferences.jsonl"
            processed = tmp_path / "missing.parquet"
            out = tmp_path / "audit.json"

            gen_row = {
                "messages": [
                    {"role": "system", "content": "Generate original rap lyrics."},
                    {"role": "user", "content": "Write a verse."},
                    {"role": "assistant", "content": "short"},
                ],
                "metadata": {
                    "song_id": "song-a",
                    "section_id": "song-a-verse",
                    "bar_index": 0,
                    "split": "train",
                    "source_license": "unknown",
                    "section_type": "verse",
                },
            }
            generation.write_text(json.dumps(gen_row) + "\n" + json.dumps(gen_row) + "\n", encoding="utf-8")
            mutation.write_text(
                json.dumps({"controls": {"output_bars": 4}, "output_bars": ["one", "two"]}) + "\n",
                encoding="utf-8",
            )
            preferences.write_text(
                json.dumps(
                    {
                        "chosen": "same line",
                        "rejected": "same line",
                        "metadata": {"quality_chosen": 1.0, "quality_rejected": 1.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_rap_datasets.py"),
                    "--generation",
                    str(generation),
                    "--mutation",
                    str(mutation),
                    "--preferences",
                    str(preferences),
                    "--processed",
                    str(processed),
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
            self.assertEqual(report["summary"]["status"], "fail")
            self.assertGreater(report["summary"]["hard_issue_types"], 0)

    def test_preference_audit_enforces_minimum_quality_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            preferences = Path(tmp) / "preferences.jsonl"
            preferences.write_text(
                json.dumps(
                    {
                        "chosen": "stronger image with a clear scene",
                        "rejected": "weaker image with a plain scene",
                        "metadata": {
                            "quality_chosen": 0.8,
                            "quality_rejected": 0.7,
                            "source_license": "unknown",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "max_synthetic_preference_rate": 0.05,
                    "min_preference_quality_gap": 0.25,
                },
            )()

            report = audit_preferences(preferences, args)

            issues = {issue["code"]: issue for issue in report["issues"]}
            self.assertEqual(issues["preference_quality_gap_below_minimum"]["count"], 1)

    def test_dataset_audit_pass_report_includes_distribution_summaries(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generation = tmp_path / "generation.jsonl"
            mutation = tmp_path / "mutation.jsonl"
            preferences = tmp_path / "preferences.jsonl"
            processed = tmp_path / "processed.parquet"
            out = tmp_path / "audit.json"
            markdown = tmp_path / "audit.md"

            generation_rows = [
                {
                    "messages": [
                        {"role": "system", "content": "Generate original rap lyrics."},
                        {"role": "user", "content": "Write a verse."},
                        {"role": "assistant", "content": "steady work builds a sharper clean routine"},
                    ],
                    "metadata": {
                        "song_id": "song-a",
                        "section_id": "song-a-verse",
                        "bar_index": 0,
                        "split": "train",
                        "source_license": "synthetic_transformed",
                        "section_type": "verse",
                        "emotion_tags": ["focused"],
                    },
                },
                {
                    "messages": [
                        {"role": "system", "content": "Generate original rap lyrics."},
                        {"role": "user", "content": "Write a hook."},
                        {"role": "assistant", "content": "late lights keep the promise moving forward"},
                    ],
                    "metadata": {
                        "song_id": "song-b",
                        "section_id": "song-b-hook",
                        "bar_index": 0,
                        "split": "validation",
                        "source_license": "synthetic_transformed",
                        "section_type": "hook",
                        "emotion_tags": ["focused"],
                    },
                },
            ]
            generation.write_text(
                "\n".join(json.dumps(row) for row in generation_rows) + "\n",
                encoding="utf-8",
            )
            mutation.write_text(
                json.dumps(
                    {
                        "controls": {"output_bars": 2},
                        "output_bars": ["first clean output line", "second clean output line"],
                        "metadata": {"split": "train", "source_license": "synthetic_transformed"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            preferences.write_text(
                json.dumps(
                    {
                        "chosen": "stronger line with sharper detail",
                        "rejected": "weaker line with plain wording",
                        "metadata": {
                            "split": "train",
                            "source_license": "synthetic_transformed",
                            "quality_chosen": 0.9,
                            "quality_rejected": 0.7,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "song_id": "song-a",
                        "section_id": "song-a-verse",
                        "split": "train",
                        "source_license": "synthetic_transformed",
                        "section_type": "verse",
                        "quality_score": 0.8,
                        "clean_bar_text": "steady work builds a sharper clean routine",
                    },
                    {
                        "song_id": "song-b",
                        "section_id": "song-b-hook",
                        "split": "validation",
                        "source_license": "synthetic_transformed",
                        "section_type": "hook",
                        "quality_score": 0.9,
                        "clean_bar_text": "late lights keep the promise moving forward",
                    },
                ]
            ).to_parquet(processed)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_rap_datasets.py"),
                    "--generation",
                    str(generation),
                    "--mutation",
                    str(mutation),
                    "--preferences",
                    str(preferences),
                    "--processed",
                    str(processed),
                    "--out",
                    str(out),
                    "--markdown-out",
                    str(markdown),
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
            self.assertEqual(report["datasets"]["generation"]["split_counts"]["train"], 1)
            self.assertEqual(
                report["datasets"]["generation"]["source_license_counts"]["synthetic_transformed"],
                2,
            )
            self.assertEqual(report["datasets"]["mutation"]["requested_output_bar_counts"]["2"], 1)
            self.assertAlmostEqual(report["datasets"]["preferences"]["quality_gap_summary"]["mean"], 0.2)
            self.assertEqual(report["datasets"]["processed"]["quality_score_summary"]["max"], 0.9)
            self.assertIn("source_license_counts", markdown.read_text(encoding="utf-8"))

    def test_dataset_audit_allows_unknown_license_only_when_explicit(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generation = tmp_path / "generation.jsonl"
            mutation = tmp_path / "mutation.jsonl"
            preferences = tmp_path / "preferences.jsonl"
            processed = tmp_path / "processed.parquet"
            out = tmp_path / "audit.json"

            generation.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Write a verse."},
                            {"role": "assistant", "content": "steady work builds a sharper clean routine"},
                        ],
                        "metadata": {
                            "song_id": "song-a",
                            "section_id": "song-a-verse",
                            "bar_index": 0,
                            "split": "train",
                            "source_license": "unknown",
                            "section_type": "verse",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mutation.write_text(
                json.dumps(
                    {
                        "controls": {"output_bars": 2},
                        "output_bars": ["first clean output line", "second clean output line"],
                        "metadata": {"split": "train", "source_license": "unknown"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            preferences.write_text(
                json.dumps(
                    {
                        "chosen": "stronger line with sharper detail",
                        "rejected": "weaker line with plain wording",
                        "metadata": {
                            "split": "train",
                            "source_license": "unknown",
                            "quality_chosen": 0.9,
                            "quality_rejected": 0.7,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "song_id": "song-a",
                        "section_id": "song-a-verse",
                        "split": "train",
                        "source_license": "unknown",
                        "section_type": "verse",
                        "quality_score": 0.8,
                        "clean_bar_text": "steady work builds a sharper clean routine",
                    }
                ]
            ).to_parquet(processed)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_rap_datasets.py"),
                    "--generation",
                    str(generation),
                    "--mutation",
                    str(mutation),
                    "--preferences",
                    str(preferences),
                    "--processed",
                    str(processed),
                    "--out",
                    str(out),
                    "--allowed-license",
                    "unknown",
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
            self.assertEqual(report["datasets"]["generation"]["source_license_counts"]["unknown"], 1)

    def test_generation_evaluator_reports_line_and_copy_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sweep = tmp_path / "sweep.jsonl"
            prompts = tmp_path / "prompts.json"
            train = tmp_path / "train.jsonl"
            out = tmp_path / "metrics.json"
            sample = tmp_path / "samples.md"
            prompt = "Write exactly 4 lines of original rap lyrics about discipline."
            generated = "line one with focus\nline two with focus\nline three with focus\nline four with focus"
            underlength = "alpha beta gamma\ncity lights move\npromise stays clean"
            sweep.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "row_id": "r1",
                                "prompt_key": "p1",
                                "prompt": prompt,
                                "generated_text": generated,
                                "hit_token_cap": False,
                            }
                        ),
                        json.dumps(
                            {
                                "row_id": "r2",
                                "prompt_key": "p1",
                                "prompt": prompt,
                                "generated_text": underlength,
                                "hit_token_cap": False,
                                "generation_attempt_count": 3,
                                "underlength_retry_count": 2,
                                "failure_tags": ["fixable_underlength"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prompts.write_text(
                json.dumps([{"prompt": prompt, "target_line_count": 4}]),
                encoding="utf-8",
            )
            train.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "Generate."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": generated},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_generation_outputs.py"),
                    "--input",
                    str(sweep),
                    "--prompts",
                    str(prompts),
                    "--train-corpus",
                    str(train),
                    "--out",
                    str(out),
                    "--sample-md",
                    str(sample),
                    "--similarity-threshold",
                    "0.8",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["row_count"], 2)
            self.assertEqual(report["summary"]["exact_line_match_rate"], 0.5)
            self.assertEqual(report["summary"]["underlength_miss_count"], 1)
            self.assertEqual(report["summary"]["underlength_retry_triggered_count"], 1)
            self.assertEqual(report["summary"]["underlength_retry_generation_count"], 2)
            self.assertEqual(report["summary"]["underlength_retry_exhausted_count"], 1)
            self.assertEqual(report["summary"]["high_copy_similarity_count"], 1)
            self.assertTrue(sample.exists())

    def test_build_datasets_applies_quality_filter_to_all_outputs(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional local dependency
            self.skipTest(f"pandas unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "processed.parquet"
            rows = [
                full_pipeline_row("s1", "sec1", 0, "high quality line zero with detail", 0.75),
                full_pipeline_row("s1", "sec1", 1, "high quality line one with detail", 0.8),
                full_pipeline_row("s1", "sec1", 2, "high quality line two with detail", 0.85),
                full_pipeline_row("s1", "sec1", 3, "high quality line three with detail", 0.9),
                full_pipeline_row("s1", "sec1", 4, "low quality fragment", 0.4),
            ]
            pd.DataFrame(rows).to_parquet(source)
            run_dir = tmp_path / "run"
            generation = tmp_path / "generation.jsonl"
            mutation = tmp_path / "mutation.jsonl"
            preferences = tmp_path / "preferences.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "rap_fast_pipeline.py"),
                    "build-datasets",
                    "--input",
                    str(source),
                    "--run-dir",
                    str(run_dir),
                    "--generation-out",
                    str(generation),
                    "--mutation-out",
                    str(mutation),
                    "--preference-out",
                    str(preferences),
                    "--include-risk",
                    "--min-quality-score",
                    "0.70",
                    "--preference-min-delta",
                    "0.25",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((run_dir / "dataset_build_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_rows"], 5)
            self.assertEqual(summary["quality_filtered_rows"], 4)
            self.assertEqual(summary["preference_min_delta"], 0.25)
            combined = read_jsonl(generation) + read_jsonl(mutation) + read_jsonl(preferences)
            self.assertNotIn("low quality fragment", json.dumps(combined))


if __name__ == "__main__":
    unittest.main()
