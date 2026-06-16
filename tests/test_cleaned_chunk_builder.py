from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_record(index: int) -> dict:
    lines = [f"line {index} number {line} with clean ambition pressure rhythm" for line in range(1, 21)]
    return {
        "record_id": f"rec-{index}",
        "title": f"Song {index}",
        "artist_clean": "Fixture Artist",
        "rap_family": "Lyrical",
        "rap_category": "Conscious Rap",
        "quality_tier": "gold" if index % 2 else "silver",
        "quality_score": 0.95,
        "lyrics_cleaned": "[VERSE]\n" + "\n".join(lines),
    }


class CleanedChunkBuilderTests(unittest.TestCase):
    def test_cli_builds_chunk_files_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps(make_record(index)) for index in range(1, 8)) + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "chunked"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_cleaned_chunk_corpus.py"),
                    "--source",
                    str(source),
                    "--validation-source",
                    str(tmp_path / "missing_validation.jsonl"),
                    "--output-dir",
                    str(output_dir),
                    "--target-lines",
                    "8",
                    "--min-lines",
                    "4",
                    "--max-words",
                    "80",
                    "--max-chunks-per-record",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads((output_dir / "cleaned_chunk_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_mode"], "deterministic_record_id_hash")
            self.assertTrue((output_dir / "categorized_rap_corpus_train_chunks.txt").exists())
            self.assertTrue((output_dir / "categorized_rap_corpus_validation_chunks.txt").exists())
            self.assertGreater(manifest["train"]["chunk_count"], 0)
            self.assertLessEqual(manifest["train"]["word_count"]["max"], 80)


if __name__ == "__main__":
    unittest.main()
