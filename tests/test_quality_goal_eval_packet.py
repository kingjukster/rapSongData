from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from scripts.build_quality_goal_eval_packet import (
    build_evaluation_packet,
    parse_generation_specs,
    sha256_file,
)


PROMPT = {
    "prompt_key": "quality-prompt-1",
    "prompt": (
        "Write exactly 12 lines of original rap lyrics about walking home under elevated train tracks. "
        "Keep one continuous scene moving forward and end with a complete declarative payoff."
    ),
    "theme_id": "theme-1",
    "theme": "walking home under elevated train tracks",
    "instruction_family": "continuous_scene",
    "prompt_family": "quality_goal_continuous_scene",
    "target_line_count": 12,
    "evaluation_split": "development",
    "samples_per_model": 1,
}


RAW_TWELVE = "\n".join(
    [
        "Train brakes scrape while my shoes tap the track",
        "Rain beads bright on the rail by my back",
        "Corner store lights make the sidewalk flash",
        "I count each step with a notebook in my bag",
        "Speakers leak bass from a window above",
        "I fold my fear where the platform was",
        "Brick walls echo every promise I pack",
        "My pen keeps pace with the wheels on the track",
        "A turnstile clicks like a snare in the dark",
        "I hold that rhythm through the rust and spark",
        "Home gets closer with the city on my jacket",
        "I reach my door with the whole night mastered.",
    ]
)


def generation_row(row_id: str, raw_text: str, fingerprint: str) -> dict:
    return {
        **PROMPT,
        "row_id": row_id,
        "candidate_index": 1,
        "sample_index": 0,
        "seed": 101,
        "raw_generated_text": raw_text,
        "generated_text": "postprocessed output deliberately has one line",
        "postprocess_applied": True,
        "hit_token_cap": False,
        "finish_reason": "eos",
        "settings": {"run_fingerprint": fingerprint},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class QualityGoalEvalPacketTests(unittest.TestCase):
    def test_builds_raw_scored_blinded_packet_and_records_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompts.json"
            base_file = root / "base.jsonl"
            adapter_file = root / "adapter.jsonl"
            train_file = root / "train.jsonl"
            output_dir = root / "evaluation"
            prompt_file.write_text(json.dumps([PROMPT]), encoding="utf-8")
            write_jsonl(base_file, [generation_row("row-1", RAW_TWELVE, "base-run-fingerprint")])
            write_jsonl(
                adapter_file,
                [generation_row("row-1", "\n".join(RAW_TWELVE.splitlines()[:-1]), "adapter-run-fingerprint")],
            )
            write_jsonl(
                train_file,
                [{"messages": [{"role": "assistant", "content": RAW_TWELVE}]}],
            )
            generations = parse_generation_specs(
                [f"base-model-secret={base_file}", f"adapter-model-secret={adapter_file}"]
            )

            manifest = build_evaluation_packet(
                generations,
                prompt_file=prompt_file,
                train_jsonls=[train_file],
                output_dir=output_dir,
                seed=73,
            )

            base_scored = [
                json.loads(line)
                for line in (output_dir / "raw_scored" / "base-model-secret.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(base_scored[0]["lyrics"], RAW_TWELVE)
            self.assertEqual(base_scored[0]["score_source"]["field"], "raw_generated_text")
            self.assertFalse(base_scored[0]["score_source"]["postprocessing_used"])
            self.assertTrue(base_scored[0]["target_issue_flags"]["exact12"])
            self.assertTrue(base_scored[0]["target_issue_flags"]["high_copy"])

            base_summary = json.loads(
                (output_dir / "summaries" / "base-model-secret.json").read_text(encoding="utf-8")
            )
            adapter_summary = json.loads(
                (output_dir / "summaries" / "adapter-model-secret.json").read_text(encoding="utf-8")
            )
            self.assertEqual(base_summary["exact_12_line_rate"], 1.0)
            self.assertEqual(adapter_summary["exact_12_line_rate"], 0.0)
            for issue in (
                "slur",
                "prompt_leakage",
                "incomplete_ending",
                "high_copy",
                "weak_imagery",
                "generic",
                "low_rhyme",
                "weak_payoff",
                "scene_drift",
            ):
                self.assertIn(issue, base_summary["issue_rates"])

            public_text = (output_dir / "comparison_packet.json").read_text(encoding="utf-8")
            public_packet = json.loads(public_text)
            self.assertNotIn("base-model-secret", public_text)
            self.assertNotIn("adapter-model-secret", public_text)
            self.assertNotIn("quality_score", public_text)
            candidates = public_packet["comparisons"][0]["candidates"]
            self.assertEqual({item["candidate_alias"] for item in candidates}, {"A", "B"})
            self.assertEqual(
                {item["lyrics"] for item in candidates},
                {RAW_TWELVE, "\n".join(RAW_TWELVE.splitlines()[:-1])},
            )

            private_key = json.loads(
                (output_dir / "comparison_alias_key.private.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {item["generation_label"] for item in private_key["assignments"][0]["candidates"]},
                {"base-model-secret", "adapter-model-secret"},
            )
            self.assertEqual(manifest["prompt_input"]["sha256"], sha256_file(prompt_file))
            self.assertEqual(manifest["training_inputs"][0]["sha256"], sha256_file(train_file))
            runs = {item["label"]: item["run"]["fingerprint"] for item in manifest["generation_inputs"]}
            self.assertEqual(runs["base-model-secret"], "base-run-fingerprint")
            self.assertEqual(runs["adapter-model-secret"], "adapter-run-fingerprint")

    def test_rejects_nonidentical_row_id_sets_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompts.json"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output_dir = root / "evaluation"
            prompt_file.write_text(json.dumps([PROMPT]), encoding="utf-8")
            write_jsonl(first, [generation_row("row-1", RAW_TWELVE, "run-a")])
            write_jsonl(second, [generation_row("row-2", RAW_TWELVE, "run-b")])

            with self.assertRaisesRegex(ValueError, "row-id sets differ"):
                build_evaluation_packet(
                    parse_generation_specs([f"first={first}", f"second={second}"]),
                    prompt_file=prompt_file,
                    train_jsonls=[],
                    output_dir=output_dir,
                    seed=1,
                )

            self.assertFalse(output_dir.exists())

    def test_rejects_prompt_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompts.json"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output_dir = root / "evaluation"
            prompt_file.write_text(json.dumps([PROMPT]), encoding="utf-8")
            write_jsonl(first, [generation_row("row-1", RAW_TWELVE, "run-a")])
            altered = generation_row("row-1", RAW_TWELVE, "run-b")
            altered["theme_id"] = "changed-theme"
            write_jsonl(second, [altered])

            with self.assertRaisesRegex(ValueError, "Prompt metadata mismatch"):
                build_evaluation_packet(
                    parse_generation_specs([f"first={first}", f"second={second}"]),
                    prompt_file=prompt_file,
                    train_jsonls=[],
                    output_dir=output_dir,
                    seed=1,
                )

            self.assertFalse(output_dir.exists())

    def test_accepts_null_generation_target_in_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_file = root / "prompts.json"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output_dir = root / "evaluation"
            prompt_file.write_text(json.dumps([PROMPT]), encoding="utf-8")
            left = generation_row("row-1", RAW_TWELVE, "run-a")
            right = generation_row("row-1", RAW_TWELVE, "run-b")
            left["target_line_count"] = None
            right["target_line_count"] = None
            write_jsonl(first, [left])
            write_jsonl(second, [right])

            manifest = build_evaluation_packet(
                parse_generation_specs([f"first={first}", f"second={second}"]),
                prompt_file=prompt_file,
                train_jsonls=[],
                output_dir=output_dir,
                seed=1,
            )

            self.assertEqual(manifest["row_count_per_model"], 1)


if __name__ == "__main__":
    unittest.main()
