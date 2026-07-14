from __future__ import annotations

import json
import unittest
from pathlib import Path

from rap_song_data.scratch.source_planning import build_scale_plan, validate_registry


ROOT = Path(__file__).resolve().parents[1]


class ScratchSourcePlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = json.loads(
            (ROOT / "configs/datasets/scratch_source_registry_v2.json").read_text(encoding="utf-8")
        )

    def test_checked_in_registry_is_valid(self):
        validate_registry(self.registry)

    def test_wasabi_is_not_counted_as_available_full_text(self):
        wasabi = next(source for source in self.registry["sources"] if source["source_id"] == "wasabi_metadata")
        self.assertFalse(wasabi["full_text_available"])
        self.assertEqual(wasabi["training_eligibility"], "metadata_only")

    def test_unknown_rights_corpus_is_private_only(self):
        current = next(
            source for source in self.registry["sources"]
            if source["source_id"] == "existing_song_lyrics_private_v1"
        )
        self.assertEqual(current["training_eligibility"], "private_only")

    def test_scale_plan_uses_actual_unique_tokens(self):
        manifest = {
            "corpus_dir": "data/scratch/v1",
            "acceptance": {"unique_training_tokens": 558_627_328},
            "splits": {"base": {"train": {"documents": 904_550}}},
        }
        plan = build_scale_plan(
            self.registry,
            manifest,
            model_sizes=[150_000_000, 300_000_000],
            tokens_per_parameter=20.0,
        )
        first, second = plan["model_targets"]
        self.assertEqual(first["target_training_tokens"], 3_000_000_000)
        self.assertEqual(first["token_gap"], 2_441_372_672)
        self.assertEqual(second["target_training_tokens"], 6_000_000_000)
        self.assertEqual(second["token_gap"], 5_441_372_672)


if __name__ == "__main__":
    unittest.main()
