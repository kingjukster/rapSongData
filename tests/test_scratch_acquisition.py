from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from rap_song_data.scratch.acquisition import (
    normalized_record,
    select_gutenberg_candidates,
    strip_gutenberg_wrapper,
)


class ScratchAcquisitionTests(unittest.TestCase):
    def test_select_gutenberg_candidates_prefers_music_and_poetry(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "pg_catalog.csv.gz"
            with gzip.open(catalog, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Text#",
                        "Type",
                        "Issued",
                        "Title",
                        "Language",
                        "Authors",
                        "Subjects",
                        "LoCC",
                        "Bookshelves",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Text#": "1",
                        "Type": "Text",
                        "Title": "A Book of Ballads and Songs",
                        "Language": "en",
                        "Subjects": "Ballads; Songs",
                        "Bookshelves": "Poetry",
                    }
                )
                writer.writerow(
                    {
                        "Text#": "2",
                        "Type": "Text",
                        "Title": "A Plain Manual",
                        "Language": "en",
                        "Subjects": "Manuals",
                    }
                )
                writer.writerow(
                    {
                        "Text#": "3",
                        "Type": "Text",
                        "Title": "Poems",
                        "Language": "fr",
                        "Subjects": "Poetry",
                    }
                )
            selected = select_gutenberg_candidates(catalog, limit=10)
            self.assertEqual([row["Text#"] for row in selected], ["1"])

    def test_strip_gutenberg_wrapper_removes_header_and_footer(self):
        text = (
            "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
            "Line one\nLine two\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter"
        )
        self.assertEqual(strip_gutenberg_wrapper(text), "Line one\nLine two")

    def test_normalized_record_preserves_item_review_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "1.txt"
            raw.write_text(
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                + "\n".join(f"line {index}" for index in range(300))
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n",
                encoding="utf-8",
            )
            record = normalized_record(
                {
                    "Text#": "1",
                    "Title": "A Book of Songs",
                    "Authors": "Example",
                    "Issued": "2026-01-01",
                    "Language": "en",
                    "Subjects": "Songs",
                    "Bookshelves": "Poetry",
                    "LoCC": "PR",
                },
                raw.read_text(encoding="utf-8"),
                raw,
                "https://example.test/1.txt",
                "2026-07-14T00:00:00+00:00",
            )
            self.assertEqual(record["source_id"], "project_gutenberg_songbooks")
            self.assertEqual(record["rights_status"], "item_review_required")
            self.assertEqual(record["license_evidence_scope"], "item_required")
            self.assertEqual(record["split_group_status"], "pending_deduplication")
            self.assertGreater(record["approx_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
