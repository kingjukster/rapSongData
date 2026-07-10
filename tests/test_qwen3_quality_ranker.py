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

from scripts.rank_qwen3_quality import score_row  # noqa: E402


PROMPT = (
    "Write exactly 12 lines of original rap lyrics about walking home under elevated train tracks. "
    "Make every line move the same scene forward. Keep the ending complete and declarative. "
    "No intro, no commentary."
)


def row(row_id: str, text: str) -> dict:
    return {
        "row_id": row_id,
        "prompt_key": "p1",
        "candidate_index": 1,
        "sample_index": 0,
        "prompt": PROMPT,
        "generated_text": text,
        "target_line_count": 12,
        "hit_token_cap": False,
        "finish_reason": "eos",
    }


class Qwen3QualityRankerTests(unittest.TestCase):
    def test_quality_ranker_prefers_concrete_scene_over_generic_motivation(self):
        strong = "\n".join(
            [
                "Train brakes scrape while my shoes tap the track",
                "Rain beads bright on the rail by my back",
                "Corner store lights make the sidewalk flash",
                "I count each step with the notebook in my bag",
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
        generic = "\n".join(
            [
                "I never give up when I chase my dreams",
                "I keep on going and I rise above",
                "I believe in myself through everything",
                "I work hard daily with a heart full of love",
                "I stay strong every time life gets tough",
                "I keep grinding because I want success",
                "I shine bright when the road gets rough",
                "I reach the sky and forget the stress",
                "I make it through with dreams on my mind",
                "I keep pushing with hope in my chest",
                "I follow my dreams and leave doubt behind",
                "I never give up because I know I'm blessed.",
            ]
        )

        strong_score = score_row(
            row("strong", strong),
            prompt_targets={PROMPT: 12},
            train_index=[],
            ngram_size=5,
            similarity_threshold=0.85,
        )
        generic_score = score_row(
            row("generic", generic),
            prompt_targets={PROMPT: 12},
            train_index=[],
            ngram_size=5,
            similarity_threshold=0.85,
        )
        artifact_score = score_row(
            row("artifact", strong.replace("whole night mastered.", "whole night mastered é‡‰.")),
            prompt_targets={PROMPT: 12},
            train_index=[],
            ngram_size=5,
            similarity_threshold=0.85,
        )

        self.assertGreater(strong_score["quality_score"], generic_score["quality_score"])
        self.assertGreater(strong_score["quality_score"], artifact_score["quality_score"])
        self.assertTrue(strong_score["structural_metrics"]["structural_pass"])
        self.assertIn("generic_motivation", generic_score["quality_tags"])
        self.assertIn("weak_imagery", generic_score["quality_tags"])
        self.assertIn("awkward_phrase", artifact_score["quality_tags"])

    def test_quality_ranker_cli_writes_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sweep = tmp_path / "sweep.jsonl"
            prompts = tmp_path / "prompts.json"
            output_md = tmp_path / "queue.md"
            output_jsonl = tmp_path / "ranked.jsonl"
            summary_json = tmp_path / "summary.json"
            text = "\n".join(
                [
                    "Train brakes scrape while my shoes tap the track",
                    "Rain beads bright on the rail by my back",
                    "Corner store lights make the sidewalk flash",
                    "I count each step with the notebook in my bag",
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
            sweep.write_text(json.dumps(row("strong", text)) + "\n", encoding="utf-8")
            prompts.write_text(json.dumps([{"prompt": PROMPT, "target_line_count": 12}]), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "rank_qwen3_quality.py"),
                    "--input",
                    str(sweep),
                    "--prompts",
                    str(prompts),
                    "--output-md",
                    str(output_md),
                    "--output-jsonl",
                    str(output_jsonl),
                    "--summary-json",
                    str(summary_json),
                    "--top-n",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["ranker"], "qwen3_4b_base_12line_v1_quality_ranker_v1")
            self.assertEqual(summary["structural_pass_count"], 1)
            self.assertIn("human_label", output_md.read_text(encoding="utf-8"))
            ranked = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ranked[0]["candidate_id"], "strong")


if __name__ == "__main__":
    unittest.main()
