from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from rap_song_data.scratch.acquisition import (
    normalized_record,
    review_gutenberg_record,
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

    def test_review_gutenberg_record_approves_standard_header(self):
        record = {
            "source_item_id": "1",
            "record_id": "abc",
            "title": "A Book of Songs",
            "authors": "Example",
            "source_url": "https://example.test/1.txt",
            "source_url_hash": "hash",
            "text_sha256": "text",
            "normalized_text_sha256": "norm",
            "language": "en",
            "word_count": 300,
            "approx_tokens": 390,
        }
        raw_text = (
            "This eBook is for the use of anyone anywhere in the United States "
            "and most other parts of the world at no cost and with almost no restrictions whatsoever.\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nBody"
        )
        review = review_gutenberg_record(record, raw_text)
        self.assertEqual(review["rights_decision"], "approved_release_candidate")
        self.assertEqual(review["license_evidence_scope"], "item_header")

    def test_review_gutenberg_record_approves_older_standard_header(self):
        record = {
            "source_item_id": "1",
            "record_id": "abc",
            "language": "en",
            "word_count": 300,
            "approx_tokens": 390,
        }
        raw_text = (
            "This eBook is for the use of anyone anywhere at no cost and with\n"
            "almost no restrictions whatsoever. You may copy it, give it away or re-use it.\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***\nBody"
        )
        review = review_gutenberg_record(record, raw_text)
        self.assertEqual(review["rights_decision"], "approved_release_candidate")

    def test_review_gutenberg_record_quarantines_restricted_header(self):
        record = {
            "source_item_id": "1",
            "record_id": "abc",
            "language": "en",
            "word_count": 300,
            "approx_tokens": 390,
        }
        raw_text = (
            "This eBook is posted with the permission of the copyright holder.\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nBody"
        )
        review = review_gutenberg_record(record, raw_text)
        self.assertEqual(review["rights_decision"], "quarantine")
        self.assertIn("restricted_marker:permission of the copyright holder", review["issues"])


if __name__ == "__main__":
    unittest.main()
