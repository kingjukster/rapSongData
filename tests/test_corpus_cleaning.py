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

from rap_song_data.corpus.cleaning import (  # noqa: E402
    CorpusCleaningConfig,
    audit_records,
    classify_language,
    clean_corpus,
    clean_lyrics_text,
)


GOOD_LINES = [
    "I clock in when the city lights flicker over the rain and the rail line hums",
    "My notebook got the overtime numbers and the pressure in the margins",
    "Every bar is another brick laid straight for the roof I keep imagining",
    "I count the small wins twice because the rent still wants a chorus",
    "The floor fan cuts the silence while ambition keeps the meter running",
    "I learned to make a map from the bus stops and the tired morning windows",
    "No shortcut in the pocket just a promise folded under my badge",
    "The skyline looks expensive but I study every shadow for a doorway",
    "Coffee cools beside the register while my plans stay hot enough to carry",
    "I leave the shift with dust on my shoes and a cleaner line of focus",
    "Tomorrow tries to tax me but tonight I put the lesson into rhythm",
    "I keep the hook honest and the verse awake until the sunrise answers",
]


def good_lyrics(marker: str = "") -> str:
    suffix = f" {marker}" if marker else ""
    return "[Verse 1]\n" + "\n".join(f"{line}{suffix}" for line in GOOD_LINES)


def record(title: str, lyrics: str, **overrides):
    payload = {
        "id": title,
        "title": title,
        "artist": "Test Artist",
        "artist_clean": "Test Artist",
        "rap_category": "Conscious Rap",
        "rap_family": "Lyrical",
        "language": "en",
        "lyrics": lyrics,
    }
    payload.update(overrides)
    return payload


class CorpusCleaningTests(unittest.TestCase):
    def test_unicode_cleanup(self):
        cleaned, flags = clean_lyrics_text(
            "\ufeff[Verse 1]\nCaf\u00e9&nbsp;dreams\u200b \u00e2\u20ac\u2122til dawn\ufffd\n"
        )
        self.assertIn("[VERSE]", cleaned)
        self.assertIn("Caf\u00e9 dreams 'til dawn", cleaned)
        self.assertNotIn("\ufeff", cleaned)
        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("\ufffd", cleaned)
        self.assertIn("html_entity_decoded", flags)

    def test_junk_line_removal(self):
        cleaned, flags = clean_lyrics_text(
            "Lyrics\nEmbed\nYou might also like\nhttps://example.com/song\n"
            "Read More\nI keep the line clean for the late shift\n"
        )
        self.assertEqual(cleaned, "I keep the line clean for the late shift")
        self.assertIn("scrape_junk_removed", flags)
        self.assertIn("url_removed", flags)

    def test_section_tag_normalization_and_adjacent_dedupe(self):
        cleaned, flags = clean_lyrics_text(
            "[Verse 1: Artist]\n[Verse 2]\n[Chorus]\n[Hook]\n[Bridge]\n[Intro]\n[Outro]\nbar"
        )
        self.assertEqual(cleaned.splitlines(), ["[VERSE]", "[HOOK]", "[BRIDGE]", "[INTRO]", "[OUTRO]", "bar"])
        self.assertIn("duplicate_adjacent_section_tag_removed", flags)

    def test_exact_and_near_duplicate_detection(self):
        near = good_lyrics("base").replace("sunrise answers base", "sunrise replies base")
        rows = [
            record("original", good_lyrics("base")),
            record("exact", good_lyrics("base")),
            record("near", near),
        ]
        audited = audit_records(rows, CorpusCleaningConfig(dedupe="both"))
        duplicate_types = {item["title"]: item["duplicate_type"] for item in audited}
        self.assertIsNone(duplicate_types["original"])
        self.assertEqual(duplicate_types["exact"], "exact")
        self.assertEqual(duplicate_types["near"], "near")
        self.assertEqual([item for item in audited if item["title"] == "exact"][0]["decision"], "drop")

    def test_repeated_line_detection(self):
        spam = "\n".join(["same line same line same line"] * 20)
        audited = audit_records([record("spam", spam)], CorpusCleaningConfig())
        item = audited[0]
        self.assertIn("repeated_line_spam", item["flags"])
        self.assertIn(item["quality_tier"], {"quarantine", "drop"})
        self.assertFalse(item["included_in_train"])

    def test_language_fallback(self):
        self.assertEqual(classify_language(good_lyrics(), {}), "en")
        self.assertEqual(classify_language("\u591c\u306e\u8857\u3067\u6b4c\u3046\n\u5149\u3068\u5f71", {}), "non_en")
        self.assertEqual(classify_language("I work tonight\n\u591c\u306e\u8857\u3067\u6b4c\u3046", {}), "mixed")

    def test_quality_tier_assignment(self):
        audited = audit_records(
            [
                record("good", good_lyrics("quality")),
                record("weak", "one short fragment"),
            ],
            CorpusCleaningConfig(),
        )
        by_title = {item["title"]: item for item in audited}
        self.assertIn(by_title["good"]["quality_tier"], {"gold", "silver"})
        self.assertTrue(by_title["good"]["included_in_train"])
        self.assertIn(by_title["weak"]["decision"], {"review", "quarantine", "drop"})
        self.assertFalse(by_title["weak"]["included_in_train"])

    def test_reports_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            rows = [record("good", good_lyrics("report")), record("duplicate", good_lyrics("report"))]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary = clean_corpus(
                CorpusCleaningConfig(
                    source_path=source,
                    output_dir=tmp_path / "cleaned",
                    report_dir=tmp_path / "reports",
                    dedupe="both",
                )
            )
            self.assertTrue((tmp_path / "cleaned" / "categorized_rap_corpus_cleaned.jsonl").exists())
            self.assertTrue((tmp_path / "cleaned" / "categorized_rap_corpus_train.txt").exists())
            self.assertTrue((tmp_path / "reports" / "corpus_cleaning_report.md").exists())
            self.assertTrue((tmp_path / "reports" / "corpus_quality_by_rap_category.csv").exists())
            self.assertTrue((tmp_path / "reports" / "corpus_duplicates.csv").exists())
            self.assertTrue((tmp_path / "reports" / "corpus_before_after_samples.jsonl").exists())
            self.assertEqual(summary["counts"]["duplicates"]["exact"], 1)

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            source.write_text(json.dumps(record("cli", good_lyrics("cli"))) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "corpus_cleaning.py"),
                    "--source",
                    str(source),
                    "--output-dir",
                    str(tmp_path / "cleaned"),
                    "--report-dir",
                    str(tmp_path / "reports"),
                    "--dedupe",
                    "both",
                    "--min-quality",
                    "0.70",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((tmp_path / "cleaned" / "corpus_cleaning_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
