from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Qwen3SweepPackagingTests(unittest.TestCase):
    def test_expanded_eval_prompt_builder_outputs_unique_12line_prompts(self):
        from scripts.build_qwen3_eval_prompts import build_prompts

        prompts = build_prompts()
        prompt_texts = [row["prompt"] for row in prompts]

        self.assertEqual(len(prompts), 60)
        self.assertEqual(len(set(prompt_texts)), len(prompt_texts))
        self.assertTrue(all(row["target_line_count"] == 12 for row in prompts))
        self.assertGreaterEqual(len({row["prompt_family"] for row in prompts}), 4)

    def test_generation_postprocess_can_trim_to_target_line_count(self):
        from scripts.run_qwen3_generation_sweep import needs_underlength_retry, postprocess_text, structural_failure_tags

        raw = "\n".join(f"line {index} with enough words" for index in range(1, 17))

        cleaned, actions = postprocess_text(raw, target_line_count=12)

        self.assertEqual(len(cleaned.splitlines()), 12)
        self.assertIn("trim_to_target_line_count", actions)
        self.assertFalse(needs_underlength_retry(cleaned, target_line_count=12))
        self.assertEqual(structural_failure_tags(cleaned, target_line_count=12), [])

        underlength = "\n".join(f"short line {index}" for index in range(1, 9))
        self.assertTrue(needs_underlength_retry(underlength, target_line_count=12))
        self.assertEqual(structural_failure_tags(underlength, target_line_count=12), ["fixable_underlength"])

    def test_cli_packages_annotated_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sweep = tmp_path / "sweep.jsonl"
            rows = [
                {
                    "id": "keeper-raw",
                    "prompt_id": "p1",
                    "prompt": "Write a clean 16-line verse about rebuilding after failure.",
                    "raw_generated_text": "line one\nline two\nline three\nline four",
                    "generated_text": "line one\nline two\nline three\nline four",
                    "decision_label": "keeper",
                    "score": 0.91,
                    "failure_tags": [],
                    "strength_tags": ["complete_verse_shape"],
                    "postprocess_actions": [],
                    "slur_present": False,
                },
                {
                    "id": "keeper-cleaned",
                    "prompt_id": "p1",
                    "prompt": "Write a clean 16-line verse about rebuilding after failure.",
                    "raw_generated_text": "good line\nwhy did it end like this?",
                    "generated_text": "good line",
                    "decision_label": "keeper",
                    "score": 0.82,
                    "failure_tags": ["question_ending"],
                    "postprocess_actions": ["drop_dangling_final_line"],
                    "raw_line_count": 2,
                    "postprocessed_line_count": 1,
                },
                {
                    "id": "fixable-1",
                    "prompt_id": "p2",
                    "prompt": "Write a hook about pressure and loyalty.",
                    "raw_generated_text": "pressure by the door\nloyalty in the rain\nshould I stay?",
                    "generated_text": "pressure by the door\nloyalty in the rain",
                    "decision_label": "fixable",
                    "score": 0.74,
                    "failure_tags": ["question_ending"],
                    "postprocess_actions": ["drop_dangling_final_line"],
                },
                {
                    "id": "reject-1",
                    "prompt_id": "p1",
                    "prompt": "Write a clean 16-line verse about rebuilding after failure.",
                    "raw_generated_text": "I cannot follow this request, what do you think?",
                    "generated_text": "I cannot follow this request, what do you think?",
                    "decision_label": "reject",
                    "score": 0.21,
                    "failure_tags": ["question_drift"],
                    "postprocess_actions": [],
                },
            ]
            sweep.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output_dir = tmp_path / "packaged"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "package_qwen3_sweep_datasets.py"),
                    "--input",
                    str(sweep),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            summary = json.loads((output_dir / "packaging_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["local_only"])
            self.assertEqual(summary["base_model_scope"], "Qwen/Qwen3-4B")
            self.assertEqual(summary["positive_sft_rows"], 1)
            self.assertEqual(summary["repair_pair_sft_rows"], 2)
            self.assertEqual(summary["manual_repair_rows"], 1)
            self.assertGreaterEqual(summary["dpo_pairs"], 2)

            positive = read_jsonl(output_dir / "positive_sft.jsonl")
            self.assertEqual(positive[0]["messages"][0]["role"], "system")
            self.assertEqual(positive[0]["messages"][2]["content"], rows[0]["generated_text"])

            repair = read_jsonl(output_dir / "repair_pairs_sft.jsonl")
            self.assertIn("Raw model output", repair[0]["messages"][1]["content"])

            dpo = read_jsonl(output_dir / "dpo_pairs.jsonl")
            reasons = {row["metadata"]["reason"] for row in dpo}
            self.assertIn("postprocess_repair_preference", reasons)
            self.assertIn("keeper_over_reject", reasons)

    def test_cli_accepts_split_curation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            keepers = tmp_path / "keepers.jsonl"
            fixable = tmp_path / "fixable.jsonl"
            rejects = tmp_path / "rejects.jsonl"
            keepers.write_text(
                json.dumps(
                    {
                        "prompt_id": "p1",
                        "prompt": "Write four clean hook lines.",
                        "raw_generated_text": "one\ntwo\nthree\nfour",
                        "generated_text": "one\ntwo\nthree\nfour",
                        "score": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fixable.write_text(
                json.dumps(
                    {
                        "prompt_id": "p2",
                        "prompt": "Write a verse with a stronger ending.",
                        "raw_generated_text": "first line\nis this enough?",
                        "generated_text": "first line",
                        "score": 0.72,
                        "failure_tags": ["weak_ending"],
                        "postprocess_actions": ["drop_dangling_final_line"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rejects.write_text(
                json.dumps(
                    {
                        "prompt_id": "p1",
                        "prompt": "Write four clean hook lines.",
                        "raw_generated_text": "what is this doing here?",
                        "generated_text": "what is this doing here?",
                        "score": 0.1,
                        "failure_tags": ["question_drift"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "packaged"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "package_qwen3_sweep_datasets.py"),
                    "--keepers",
                    str(keepers),
                    "--fixable",
                    str(fixable),
                    "--rejects",
                    str(rejects),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output_dir / "packaging_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_rows"], 3)
            self.assertEqual(summary["positive_sft_rows"], 1)
            self.assertEqual(summary["repair_pair_sft_rows"], 1)
            self.assertGreaterEqual(summary["dpo_pairs"], 2)

    def test_cli_builds_training_text_files_from_packaged_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packaged = tmp_path / "packaged"
            packaged.mkdir()
            positive = {
                "messages": [
                    {"role": "system", "content": "Generate original rap lyrics. Do not copy existing songs."},
                    {"role": "user", "content": "Write a clean hook about pressure."},
                    {"role": "assistant", "content": "Pressure at the door\nLoyalty on my sleeve"},
                ],
                "metadata": {"row_id": "positive-1", "prompt_key": "p1"},
            }
            repair = {
                "messages": [
                    {"role": "system", "content": "Repair rap generations."},
                    {
                        "role": "user",
                        "content": "Original prompt:\nWrite a better ending.\n\nRaw model output:\nline one?<|im_end|><|im_end|>",
                    },
                    {"role": "assistant", "content": "line one lands clean"},
                ],
                "metadata": {"row_id": "repair-1", "prompt_key": "p2"},
            }
            (packaged / "positive_sft.jsonl").write_text(json.dumps(positive) + "\n", encoding="utf-8")
            (packaged / "repair_pairs_sft.jsonl").write_text(json.dumps(repair) + "\n", encoding="utf-8")
            output_dir = tmp_path / "training"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_qwen3_sft_training_files.py"),
                    "--packaged-dir",
                    str(packaged),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["local_only"])
            self.assertEqual(manifest["base_model_scope"], "Qwen/Qwen3-4B")
            self.assertEqual(manifest["counts"]["total_training_rows"], 2)
            train_rows = read_jsonl(output_dir / "train.jsonl")
            validation_rows = read_jsonl(output_dir / "validation.jsonl")
            self.assertGreaterEqual(len(train_rows), 1)
            self.assertGreaterEqual(len(validation_rows), 1)
            combined_text = "\n".join(row["training_text"] for row in train_rows + validation_rows)
            self.assertIn("<|im_start|>system", combined_text)
            self.assertIn("<|im_start|>assistant", combined_text)
            self.assertNotIn("line one?<|im_end|><|im_end|>", combined_text)

    def test_gpu_max_runner_prepare_only_writes_profile_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            packaged = tmp_path / "packaged"
            packaged.mkdir()
            row = {
                "messages": [
                    {"role": "system", "content": "Generate original rap lyrics. Do not copy existing songs."},
                    {"role": "user", "content": "Write a clean verse about discipline."},
                    {"role": "assistant", "content": "I keep the meter steady\nI keep the ending clean"},
                ],
                "metadata": {"row_id": "positive-1", "prompt_key": "p1"},
            }
            (packaged / "positive_sft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (packaged / "repair_pairs_sft.jsonl").write_text("", encoding="utf-8")
            training_dir = tmp_path / "training"
            run_dir = tmp_path / "run"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_qwen3_sweep_sft_gpu_max.py"),
                    "--packaged-dir",
                    str(packaged),
                    "--training-dir",
                    str(training_dir),
                    "--run-dir",
                    str(run_dir),
                    "--prepare-only",
                    "--profile",
                    "seq768_bs2_ga2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((run_dir / "run_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "prepared")
            self.assertEqual(plan["base_model"], "Qwen/Qwen3-4B")
            config_path = Path(plan["profiles"][0]["config_path"])
            self.assertTrue(config_path.exists())
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["base_model"], "Qwen/Qwen3-4B")
            self.assertEqual(config["training"]["sequence_length"], 768)
            self.assertEqual(config["training"]["per_device_train_batch_size"], 2)

    def test_local_curation_splits_sweep_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sweep = tmp_path / "sweep_raw.jsonl"
            rows = [
                {
                    "row_id": "good",
                    "prompt_key": "p1",
                    "prompt": "Write a 12-line verse about discipline.",
                    "raw_generated_text": "\n".join(
                        f"clean line {i} with rhythm and focus{'.' if i == 11 else ''}" for i in range(12)
                    ),
                    "generated_text": "\n".join(
                        f"clean line {i} with rhythm and focus{'.' if i == 11 else ''}" for i in range(12)
                    ),
                    "postprocess_actions": [],
                },
                {
                    "row_id": "fix",
                    "prompt_key": "p2",
                    "prompt": "Write a hook with 4 short lines about pressure.",
                    "raw_generated_text": "pressure at the door\nloyalty in the rain\nshould I stay?",
                    "generated_text": "pressure at the door\nloyalty in the rain",
                    "postprocess_applied": True,
                    "postprocess_actions": ["drop_dangling_final_line"],
                },
                {
                    "row_id": "bad",
                    "prompt_key": "p3",
                    "prompt": "Write a clean radio-safe verse. No slurs.",
                    "raw_generated_text": "what do you think?\nwhat should I do?",
                    "generated_text": "what do you think?\nwhat should I do?",
                    "postprocess_actions": [],
                },
            ]
            sweep.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output_dir = tmp_path / "curation"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "curate_qwen3_sweep.py"),
                    "--input",
                    str(sweep),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output_dir / "curation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["keepers"], 1)
            self.assertGreaterEqual(summary["fixable"], 1)
            self.assertGreaterEqual(summary["rejects"], 1)

    def test_rebuild_pipeline_skip_sweep_prepare_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            adapter_dir = tmp_path / "fake_adapter"
            adapter_dir.mkdir()
            sweep_dir = tmp_path / "sweep"
            sweep_dir.mkdir()
            sweep = sweep_dir / "sweep_raw.jsonl"
            good_text = "\n".join(f"clean line {i} with rhythm and focus{'.' if i == 11 else ''}" for i in range(12))
            fixed_raw = "pressure at the door\nloyalty in the rain\nshould I stay?"
            rows = [
                {
                    "row_id": "good",
                    "prompt_key": "p1",
                    "prompt": "Write a 12-line verse about discipline.",
                    "raw_generated_text": good_text,
                    "generated_text": good_text,
                    "postprocess_actions": [],
                },
                {
                    "row_id": "fix",
                    "prompt_key": "p2",
                    "prompt": "Write a hook with 4 short lines about pressure.",
                    "raw_generated_text": fixed_raw,
                    "generated_text": "pressure at the door\nloyalty in the rain",
                    "postprocess_applied": True,
                    "postprocess_actions": ["drop_dangling_final_line"],
                },
            ]
            sweep.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            run_dir = tmp_path / "run"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "rebuild_qwen3_sweep_pipeline.py"),
                    "--adapter-dir",
                    str(adapter_dir),
                    "--skip-sweep",
                    "--sweep-dir",
                    str(sweep_dir),
                    "--curation-dir",
                    str(tmp_path / "curation"),
                    "--packaged-dir",
                    str(tmp_path / "packaged"),
                    "--training-dir",
                    str(tmp_path / "training"),
                    "--run-dir",
                    str(run_dir),
                    "--training-profile",
                    "seq768_bs2_ga2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["training_started"])
            self.assertTrue((tmp_path / "training" / "train.jsonl").exists())
            self.assertTrue((run_dir / "05_training" / "run_plan.json").exists())


if __name__ == "__main__":
    unittest.main()
