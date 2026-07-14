from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from rap_song_data.scratch.common import write_json, write_jsonl
from rap_song_data.scratch.source_governance import (
    CANONICAL_RECORD_FIELDS,
    admit_source,
    audit_source,
    build_profile,
    ingest_source,
    inspect_corpus,
    migrate_source,
    revoke_source,
    revocation_impact,
    verify_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class ScratchSourceGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ROOT / "configs/datasets/scratch_source_registry_v2.json"

    def _fixture_corpus(self, root: Path) -> Path:
        corpus_dir = root / "scratch"
        corpus_dir.mkdir()
        rows = [
            {
                "record_id": "one",
                "split": "train",
                "title": "Fixture One",
                "artist_clean": "artist a",
                "language_labels": ["en"],
                "language_disagreement": False,
                "content_flags": ["explicit_language"],
                "section_tokens": ["<|verse|>", "<|chorus|>"],
                "line_count": 18,
                "lyrics": "line one\nline two",
            },
            {
                "record_id": "two",
                "split": "validation",
                "title": "Fixture Two",
                "artist_clean": "artist b",
                "language_labels": ["en"],
                "language_disagreement": True,
                "content_flags": [],
                "section_tokens": ["<|verse|>"],
                "line_count": 42,
                "lyrics": "line three\nline four",
            },
        ]
        write_jsonl(corpus_dir / "all.jsonl", rows)
        write_json(
            corpus_dir / "corpus_manifest.json",
            {
                "private_research_only": True,
                "public_release_blocked": True,
                "counts": {"retained": 2, "duplicate_exact": 1, "duplicate_near": 2},
            },
        )
        write_json(
            corpus_dir / "tokenization_manifest.json",
            {"splits": {"base": {"train": {"input_tokens": 1234}}}},
        )
        return corpus_dir

    def test_inspect_corpus_reports_composition_and_missing_item_rights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            result = inspect_corpus(
                argparse.Namespace(
                    corpus_dir=corpus_dir,
                    registry=self.registry,
                    source_id=None,
                    output_dir=root / "catalog",
                    max_records=None,
                    sample_records=2,
                    top_n=5,
                )
            )
            self.assertEqual(result["scan"]["records_scanned"], 2)
            self.assertEqual(result["composition"]["explicit_flagged_records"], 1)
            self.assertEqual(result["tokens"]["total_tokens_from_tokenizer_manifest"], 1234)
            self.assertIsNone(result["tokens"]["tokens_observed_in_scan"])
            self.assertIn("source_id", result["canonical_record_gap"]["missing_field_counts"])
            self.assertEqual(result["dedupe"]["duplicate_near"], 2)
            self.assertTrue((root / "catalog" / "corpus_composition.json").exists())

    def test_source_lifecycle_manifests_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            common = {
                "registry": self.registry,
                "corpus_dir": corpus_dir,
                "output_dir": root / "sources",
                "source": "existing_song_lyrics_private_v1",
            }
            ingest = ingest_source(argparse.Namespace(**common, snapshot_dir=None))
            audit = audit_source(argparse.Namespace(**common, allow_conditional=False))
            admit = admit_source(argparse.Namespace(**common, rights_evidence=None, force=False))
            revoke = revoke_source(argparse.Namespace(**common, reason="test removal"))

            self.assertEqual(ingest["status"], "ready")
            self.assertTrue(audit["passed"])
            self.assertEqual(admit["admission_status"], "admitted")
            self.assertFalse(revoke["destructive_changes_performed"])
            self.assertEqual(revoke["impact_estimate"]["retained_records"], 2)

    def test_admission_requires_current_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            common = {
                "registry": self.registry,
                "corpus_dir": corpus_dir,
                "output_dir": root / "sources",
                "source": "existing_song_lyrics_private_v1",
            }
            ingest_source(argparse.Namespace(**common, snapshot_dir=None))
            with self.assertRaises(ValueError):
                admit_source(argparse.Namespace(**common, rights_evidence=None, force=False))

    def test_changed_ingest_manifest_invalidates_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            common = {
                "registry": self.registry,
                "corpus_dir": corpus_dir,
                "output_dir": root / "sources",
                "source": "existing_song_lyrics_private_v1",
            }
            ingest_source(argparse.Namespace(**common, snapshot_dir=None))
            audit_source(argparse.Namespace(**common, allow_conditional=False))
            ingest_path = root / "sources" / "existing_song_lyrics_private_v1" / "ingest_manifest.json"
            payload = json.loads(ingest_path.read_text(encoding="utf-8"))
            payload["status"] = "changed-after-audit"
            write_json(ingest_path, payload)
            with self.assertRaises(ValueError):
                admit_source(argparse.Namespace(**common, rights_evidence=None, force=False))

    def test_conditional_source_requires_rights_evidence_for_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            ingest_source(
                argparse.Namespace(
                    registry=self.registry,
                    corpus_dir=corpus_dir,
                    output_dir=root / "sources",
                    source="project_gutenberg_songbooks",
                    snapshot_dir=None,
                )
            )
            audit_source(
                argparse.Namespace(
                    registry=self.registry,
                    corpus_dir=corpus_dir,
                    output_dir=root / "sources",
                    source="project_gutenberg_songbooks",
                    allow_conditional=True,
                )
            )
            with self.assertRaises(ValueError):
                admit_source(
                    argparse.Namespace(
                        registry=self.registry,
                        corpus_dir=corpus_dir,
                        output_dir=root / "sources",
                        source="project_gutenberg_songbooks",
                        rights_evidence=None,
                        force=False,
                    )
                )

    def test_private_source_cannot_enter_open_core_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            common = {
                "registry": self.registry,
                "corpus_dir": corpus_dir,
                "output_dir": root / "sources",
                "source": "existing_song_lyrics_private_v1",
            }
            ingest_source(argparse.Namespace(**common, snapshot_dir=None))
            audit_source(argparse.Namespace(**common, allow_conditional=False))
            admit_source(argparse.Namespace(**common, rights_evidence=None, force=False))
            result = build_profile(
                argparse.Namespace(
                    profile="scratch-core-open-v1",
                    registry=self.registry,
                    output_dir=corpus_dir,
                    catalog_dir=root / "catalog",
                    sources_dir=root / "sources",
                )
            )
            included = {source["source_id"] for source in result["included_sources"]}
            self.assertNotIn("existing_song_lyrics_private_v1", included)
            self.assertNotIn("musicbrainz_core", included)

    def test_profile_build_blocks_matching_sources_that_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            result = build_profile(
                argparse.Namespace(
                    profile="scratch-private-extended-v1",
                    registry=self.registry,
                    output_dir=corpus_dir,
                    catalog_dir=root / "catalog",
                    sources_dir=root / "sources",
                )
            )
            self.assertEqual(result["included_sources"], [])
            blocked = {source["source_id"]: source for source in result["excluded_sources"]}
            self.assertEqual(blocked["existing_song_lyrics_private_v1"]["blocked_reason"], "not_admitted")

    def test_migration_is_deterministic_and_preserves_legacy_rights(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left_root = Path(left_dir)
            right_root = Path(right_dir)
            left_corpus = self._fixture_corpus(left_root)
            right_corpus = self._fixture_corpus(right_root)
            common = {
                "registry": self.registry,
                "source": "existing_song_lyrics_private_v1",
                "schema_version": 2,
                "shard_size": 1,
                "limit": None,
                "resume": False,
                "force": False,
                "governed_at": "2026-07-14T00:00:00+00:00",
                "reason": "test migration",
            }
            first = migrate_source(
                argparse.Namespace(
                    **common,
                    corpus_dir=left_corpus,
                    output_dir=left_root / "sources",
                )
            )
            second = migrate_source(
                argparse.Namespace(
                    **common,
                    corpus_dir=right_corpus,
                    output_dir=right_root / "sources",
                )
            )
            self.assertEqual(first["counts"]["sidecar_records"], 2)
            self.assertEqual(first["invariants"]["record_count_discrepancy"], 0)
            left_rows = pq.read_table(first["shards"][0]["path"]).to_pylist()
            right_rows = pq.read_table(second["shards"][0]["path"]).to_pylist()
            self.assertEqual(left_rows[0]["record_id"], right_rows[0]["record_id"])
            self.assertEqual(left_rows[0]["text_sha256"], right_rows[0]["text_sha256"])
            self.assertIsNone(left_rows[0]["license_id"])
            self.assertEqual(left_rows[0]["rights_status"], "unknown_legacy")
            self.assertEqual(left_rows[0]["split_group_status"], "pending_deduplication")

    def test_verify_profile_and_revocation_impact_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_dir = self._fixture_corpus(root)
            common = {
                "registry": self.registry,
                "corpus_dir": corpus_dir,
                "output_dir": root / "sources",
                "source": "existing_song_lyrics_private_v1",
            }
            ingest_source(argparse.Namespace(**common, snapshot_dir=None))
            audit_source(argparse.Namespace(**common, allow_conditional=False))
            admit_source(argparse.Namespace(**common, rights_evidence=None, force=False))
            build_profile(
                argparse.Namespace(
                    profile="scratch-core-open-v1",
                    registry=self.registry,
                    output_dir=corpus_dir,
                    catalog_dir=root / "catalog",
                    sources_dir=root / "sources",
                )
            )
            verification = verify_profile(
                argparse.Namespace(
                    profile="scratch-core-open-v1",
                    catalog_dir=root / "catalog",
                    sources_dir=root / "sources",
                )
            )
            impact = revocation_impact(
                argparse.Namespace(
                    source="existing_song_lyrics_private_v1",
                    catalog_dir=root / "catalog",
                    sources_dir=root / "sources",
                    corpus_dir=corpus_dir,
                    model_dir=root / "model",
                )
            )
            self.assertTrue(verification["passed"])
            self.assertEqual(impact["source_id"], "existing_song_lyrics_private_v1")

    def test_canonical_record_field_list_includes_revocation_keys(self):
        self.assertIn("removal_key", CANONICAL_RECORD_FIELDS)
        self.assertIn("license_evidence", CANONICAL_RECORD_FIELDS)


if __name__ == "__main__":
    unittest.main()
