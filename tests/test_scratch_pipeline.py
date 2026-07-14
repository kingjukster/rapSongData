from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rap_song_data.scratch.common import read_json
from rap_song_data.scratch.compliance import apply_line_controls, classify_failure
from rap_song_data.scratch.comparison import prompt_matrix, validate_prompt_matrix
from rap_song_data.scratch.corpus import (
    DedupeStore,
    canonicalize_lyrics,
    content_flags,
    language_decision,
    normalize_artist,
    split_for_artist,
)
from rap_song_data.scratch.evaluation import (
    build_blind_packet,
    distinct_n,
    repeated_line_ratio,
    score_blind_review,
    summarize_outputs,
    lyric_lines,
    target_lines,
)
from rap_song_data.scratch.modeling import ScratchModelSpec
from rap_song_data.scratch.sweep import length_bucket, stratified_indices, token_budget
from rap_song_data.scratch.router import structured_prompt


class ScratchCorpusUnitTests(unittest.TestCase):
    def test_normalization_preserves_canonical_sections(self):
        lyrics, sections, line_count = canonicalize_lyrics(
            "[Verse 1: Example]\nFirst bar\nSecond bar\n[Chorus]\nHook line\n2 Contributors Lyrics\n"
        )
        self.assertIn("<|verse|>", lyrics)
        self.assertIn("<|chorus|>", lyrics)
        self.assertEqual(sections, ["<|verse|>", "<|chorus|>"])
        self.assertEqual(line_count, 3)

    def test_language_any_english_and_disagreement(self):
        accepted, disagreement, labels = language_decision(
            {"language": "es", "language_cld3": "en", "language_ft": "en"}
        )
        self.assertTrue(accepted)
        self.assertTrue(disagreement)
        self.assertEqual(labels, ["es", "en", "en"])

    def test_explicit_content_is_flagged_not_removed(self):
        self.assertEqual(content_flags("This shit stays in the legitimate lyric"), ["explicit_language"])

    def test_artist_split_is_deterministic(self):
        artist = normalize_artist("Beyoncé & Guest")
        self.assertEqual(split_for_artist(artist), split_for_artist(artist))

    def test_dedupe_store_finds_exact_and_near_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = DedupeStore(Path(temporary) / "dedupe.sqlite3", threshold=0.8)
            try:
                text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
                self.assertEqual(store.classify("one", text, batch_id=0)[0], "unique")
                self.assertEqual(store.classify("two", text, batch_id=0)[0], "exact")
                near = text + " lambda"
                duplicate_type, representative, score = store.classify("three", near, batch_id=0)
                self.assertEqual(duplicate_type, "near")
                self.assertEqual(representative, "one")
                self.assertGreaterEqual(score, 0.8)
            finally:
                store.close()


class ScratchModelAndEvaluationTests(unittest.TestCase):
    def test_default_model_is_30m_class(self):
        count = ScratchModelSpec().parameter_count()
        self.assertGreater(count, 28_000_000)
        self.assertLess(count, 30_000_000)

    def test_generation_metrics(self):
        rows = [
            {"sft_text": "<|target_lines|>2\n<|lyrics|>\na\nb<|eos|>"},
            {"sft_text": "<|target_lines|>2\n<|lyrics|>\nc\nd<|eos|>"},
        ]
        outputs = ["alpha rhyme\nbeta time", "same line\nsame line"]
        summary = summarize_outputs(rows, outputs)
        self.assertEqual(summary["exact_line_match_rate"], 1.0)
        self.assertGreater(repeated_line_ratio(outputs[1]), 0.0)
        self.assertGreater(distinct_n(outputs, 2), 0.0)

    def test_compliance_classifier_separates_decoding_failures(self):
        under = classify_failure("one\ntwo", 4, generated_tokens=20)
        self.assertEqual(under["primary_failure"], "underlength")
        truncated = classify_failure("one\ntwo", 4, generated_tokens=255)
        self.assertEqual(truncated["primary_failure"], "truncation")
        continued = classify_failure("one\ntwo\nthree", 2, generated_tokens=20)
        self.assertEqual(continued["primary_failure"], "wrapped_line")

    def test_line_controls_remove_labels_and_stop_at_target(self):
        output = "[Verse 1]\nfirst bar\nsecond bar\nthird bar"
        self.assertEqual(apply_line_controls(output, 2), "first bar\nsecond bar")
        result = classify_failure(output, 2, generated_tokens=20)
        self.assertTrue(result["system_exact"])

    def test_corrected_parser_excludes_human_section_labels(self):
        self.assertEqual(lyric_lines("[Verse 1]\nfirst bar\nsecond bar"), ["first bar", "second bar"])

    def test_dynamic_budget_and_stratification(self):
        self.assertEqual(token_budget(32, "line_adjusted", 256, 576), 576)
        self.assertEqual(token_budget(4, "line_adjusted", 256, 576), 72)
        rows = [
            {"sft_text": f"<|target_lines|>{target}\n<|lyrics|>\n"}
            for target in ([4] * 4 + [12] * 4 + [20] * 4 + [32] * 4)
        ]
        selected = stratified_indices(rows, 8, 7)
        buckets = [length_bucket(target_lines(rows[index])) for index in selected]
        self.assertEqual(len(set(buckets)), 4)

    def test_controlled_router_builds_canonical_prompt(self):
        prompt = structured_prompt(
            {"title": "Cold Signal", "section": "hook", "target_lines": 8, "content_policy": "clean"}
        )
        self.assertIn("<|section|>hook", prompt)
        self.assertIn("<|target_lines|>8", prompt)
        self.assertTrue(prompt.endswith("<|lyrics|>\n"))

    def test_three_way_prompt_matrix_is_exactly_stratified(self):
        rows = prompt_matrix()
        validate_prompt_matrix(rows)
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({row["prompt_id"] for row in rows}), 64)
        for family in ("melodic", "story", "technical", "clean"):
            for target in (4, 8, 16, 32):
                selected = [row for row in rows if row["family"] == family and row["target_lines"] == target]
                self.assertEqual(len(selected), 4)
                self.assertTrue(all(f"exactly {target} lines" in row["prompt"] for row in selected))
                self.assertTrue(all("lyrics only" in row["prompt"].lower() for row in selected))

    def test_blind_packet_hides_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            rows = [{"sft_text": "<|task_generate|><|lyrics|>\n"}]
            result = build_blind_packet(
                output_dir,
                rows,
                {"sft": ["first anonymous output"], "baseline": ["second anonymous output"]},
                count=1,
                seed=20260713,
            )
            packet = Path(result["packet"]).read_text(encoding="utf-8")
            self.assertNotIn("sft", packet)
            self.assertNotIn("baseline", packet)
            key = read_json(Path(result["key"]))
            self.assertEqual(len(key["rows"]), 1)

    def test_blind_review_gate_requires_30_scored_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            rows = [{"sft_text": f"prompt {index}<|lyrics|>\n"} for index in range(30)]
            result = build_blind_packet(
                output_dir,
                rows,
                {"sft": [f"scratch {index}" for index in range(30)], "baseline": [f"base {index}" for index in range(30)]},
                count=30,
                seed=20260713,
            )
            packet_path = Path(result["packet"])
            with packet_path.open("r", encoding="utf-8", newline="") as handle:
                packet_rows = list(csv.DictReader(handle))
                fields = list(packet_rows[0])
            for row in packet_rows:
                for metric in ("coherence_a", "coherence_b", "structure_a", "structure_b", "originality_a", "originality_b"):
                    row[metric] = "4"
            with packet_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(packet_rows)
            scored = score_blind_review(packet_path, Path(result["key"]))
            self.assertEqual(scored["reviewed_comparisons"], 30)
            self.assertTrue(scored["human_gate_passed"])


class ScratchEndToEndCpuTests(unittest.TestCase):
    def _artist_for_split(self, split: str, prefix: str) -> str:
        for index in range(10_000):
            candidate = f"{prefix} {index}"
            if split_for_artist(normalize_artist(candidate)) == split:
                return candidate
        raise AssertionError(f"Could not find fixture artist for {split}")

    def _write_fixture_csv(self, path: Path) -> None:
        artists = {
            "train": self._artist_for_split("train", "Train Artist"),
            "validation": self._artist_for_split("validation", "Validation Artist"),
            "test": self._artist_for_split("test", "Test Artist"),
        }
        fields = [
            "title", "tag", "artist", "year", "views", "features", "lyrics", "id",
            "language_cld3", "language_ft", "language",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            record_id = 1
            for split, artist in artists.items():
                for song in range(3):
                    lines = [
                        f"{split} song {song} original bar {line} rhythm pressure ambition skyline"
                        for line in range(1, 13)
                    ]
                    writer.writerow(
                        {
                            "title": f"{split.title()} Song {song}",
                            "tag": "rap",
                            "artist": artist,
                            "year": 2026,
                            "views": 100,
                            "features": "{}",
                            "lyrics": "[Verse 1]\n" + "\n".join(lines),
                            "id": record_id,
                            "language_cld3": "en",
                            "language_ft": "en",
                            "language": "en",
                        }
                    )
                    record_id += 1

    def test_fixture_corpus_tokenizer_cpu_train_and_resume(self):
        from rap_song_data.scratch.corpus import build_corpus
        from rap_song_data.scratch.tokenization import train_and_tokenize
        from rap_song_data.scratch.training import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "songs.csv"
            self._write_fixture_csv(source)
            corpus_dir = root / "scratch"
            corpus_args = argparse.Namespace(
                input=source,
                output_dir=corpus_dir,
                rap_cache=root / "rap.parquet",
                batch_size=3,
                min_chars=100,
                max_chars=20_000,
                min_lines=4,
                near_duplicate_threshold=0.99,
                seed=20260713,
                min_retained=1,
                limit=None,
                force=False,
                force_cache=False,
            )
            corpus_manifest = build_corpus(corpus_args)
            self.assertEqual(corpus_manifest["counts"]["retained"], 9)
            artists_by_split = {
                split: {row["artist_clean"] for row in self._read_jsonl(corpus_dir / f"{split}.jsonl")}
                for split in ("train", "validation", "test")
            }
            self.assertFalse(artists_by_split["train"] & artists_by_split["validation"])
            self.assertFalse(artists_by_split["train"] & artists_by_split["test"])

            token_args = argparse.Namespace(
                corpus_dir=corpus_dir,
                output_dir=corpus_dir,
                vocab_size=512,
                sequence_length=64,
                shard_tokens=256,
                min_train_tokens=1,
                limit=None,
                force=False,
            )
            token_manifest = train_and_tokenize(token_args)
            self.assertTrue(token_manifest["acceptance"]["full_pilot_allowed"])
            self.assertGreater(token_manifest["splits"]["base"]["train"]["blocks"], 0)

            config = root / "tiny_config.json"
            config.write_text(
                json.dumps(
                    {
                        "training": {
                            "sequence_length": 64,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "learning_rate": 0.0003,
                            "final_learning_rate": 0.00003,
                            "target_tokens": 10_000,
                            "max_epochs": 2.0,
                            "max_hours": 0.1,
                            "eval_every_steps": 1,
                            "eval_batches": 1,
                            "checkpoint_minutes": 60,
                            "dataloader_workers": 0
                        }
                    }
                ),
                encoding="utf-8",
            )
            first_dir = root / "train"
            first_args = argparse.Namespace(
                data_dir=corpus_dir,
                output_dir=first_dir,
                config=config,
                resume=None,
                model_dir=None,
                smoke=True,
                max_steps=2,
                cpu=True,
                tiny_model=True,
                ignore_data_gate=False,
            )
            first = train(first_args, mode="base")
            self.assertEqual(first["result"]["global_step"], 2)
            self.assertEqual(len(first["result"]["validation_history"]), 2)
            checkpoint = Path(first["result"]["final_checkpoint"])
            checkpoint_state = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(checkpoint_state["validation_history"]), 2)
            resumed_args = argparse.Namespace(**vars(first_args))
            resumed_args.resume = checkpoint
            resumed_args.max_steps = 3
            resumed = train(resumed_args, mode="base")
            self.assertEqual(resumed["result"]["global_step"], 3)
            self.assertEqual(len(resumed["result"]["validation_history"]), 3)
            self.assertEqual(
                resumed["result"]["first_validation_loss"],
                first["result"]["first_validation_loss"],
            )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
