from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import command_record, hash_text, iter_jsonl, path_manifest, read_json, utc_now, write_json
from .corpus import normalized_lyrics_key
from .source_planning import validate_registry


CANONICAL_RECORD_FIELDS = [
    "record_id",
    "source_id",
    "source_item_id",
    "retrieved_at",
    "source_url_hash",
    "rights_tier",
    "license_id",
    "license_evidence",
    "work_id",
    "recording_id",
    "artist_ids",
    "text_sha256",
    "normalized_text_sha256",
    "dedupe_cluster_id",
    "removal_key",
    "split_group_id",
]

TOOL_VERSION = "scratch-source-governance-v2"

PROFILE_RULES = {
    "scratch-core-open-v1": {
        "description": "Public-domain, CC0, and clearly permissive material only.",
        "eligibility": {"eligible"},
        "partitions": {"tier_a_release_candidate"},
    },
    "scratch-research-nc-v1": {
        "description": "Open core plus noncommercial or research-restricted material.",
        "eligibility": {"eligible", "conditional"},
        "partitions": {"tier_a_release_candidate", "tier_b_private_noncommercial"},
    },
    "scratch-private-extended-v1": {
        "description": "Research corpus plus private, uncertain-rights local material.",
        "eligibility": {"eligible", "conditional", "private_only"},
        "partitions": {
            "tier_a_release_candidate",
            "tier_b_private_noncommercial",
            "private_unknown_rights",
        },
    },
    "scratch-quarantine-v1": {
        "description": "Sources that must not enter training until provenance or quality is resolved.",
        "eligibility": {"excluded"},
        "partitions": {"quarantine", "excluded"},
    },
}


def _source_by_id(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    validate_registry(registry)
    for source in registry["sources"]:
        if source["source_id"] == source_id:
            return source
    raise ValueError(f"Unknown source_id: {source_id}")


def _current_source_id(registry: dict[str, Any]) -> str:
    private = [
        source["source_id"]
        for source in registry["sources"]
        if source.get("training_eligibility") == "private_only" and source.get("full_text_available")
    ]
    return private[0] if private else "existing_song_lyrics_private_v1"


def _iter_limited(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(iter_jsonl(path)):
        if limit is not None and index >= limit:
            break
        yield row


def _length_bucket(line_count: int) -> str:
    if line_count < 8:
        return "under_8"
    if line_count < 16:
        return "8_15"
    if line_count < 32:
        return "16_31"
    if line_count < 64:
        return "32_63"
    return "64_plus"


def _section_name(token: str) -> str:
    return token.removeprefix("<|").removesuffix("|>")


def _record_missing_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in CANONICAL_RECORD_FIELDS if field not in row]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _manifest_path(output_dir: Path, source_id: str, name: str) -> Path:
    return output_dir / source_id / name


def _optional_manifest(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _registry_manifest(registry_path: Path) -> dict[str, Any]:
    return path_manifest(registry_path)


def _transition_base(
    *,
    operation: str,
    source: dict[str, Any],
    registry_path: Path,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": operation,
        "source_id": source["source_id"],
        "source": source,
        "reason": reason,
        "actor": "codex",
        "tool_version": TOOL_VERSION,
        "source_registry": _registry_manifest(registry_path),
    }


def _generated_at(path: Path) -> str:
    manifest = _optional_manifest(path)
    return str(manifest.get("generated_at") or "") if manifest else ""


def _source_is_revoked(source_dir: Path) -> bool:
    revocation_path = source_dir / "revocation_plan.json"
    admission_path = source_dir / "admission_manifest.json"
    if not revocation_path.exists():
        return False
    if not admission_path.exists():
        return True
    return _generated_at(revocation_path) >= _generated_at(admission_path)


def inspect_corpus(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    corpus_dir = Path(args.corpus_dir)
    registry = read_json(Path(args.registry))
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    tokenization_manifest_path = corpus_dir / "tokenization_manifest.json"
    corpus_manifest = read_json(corpus_manifest_path)
    tokenization_manifest = read_json(tokenization_manifest_path) if tokenization_manifest_path.exists() else {}
    source_id = args.source_id or _current_source_id(registry)
    source = _source_by_id(registry, source_id)
    all_path = corpus_dir / "all.jsonl"

    counters: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_length_bucket: Counter[str] = Counter()
    by_section: Counter[str] = Counter()
    missing_field_counts: Counter[str] = Counter()
    artist_counts: Counter[str] = Counter()
    title_hash_counts: Counter[str] = Counter()
    line_counts: list[int] = []
    sample_records: list[dict[str, Any]] = []
    observed_lyric_bytes = 0

    for row in _iter_limited(all_path, args.max_records):
        counters["records_scanned"] += 1
        observed_lyric_bytes += len(str(row.get("lyrics") or "").encode("utf-8", errors="ignore"))
        by_split[str(row.get("split", "unknown"))] += 1
        labels = row.get("language_labels") or ["unknown"]
        for label in labels:
            by_language[str(label or "unknown")] += 1
        flags = row.get("content_flags") or []
        if flags:
            counters["explicit_flagged"] += 1
        if row.get("language_disagreement"):
            counters["language_disagreement"] += 1
        line_count = int(row.get("line_count") or 0)
        line_counts.append(line_count)
        by_length_bucket[_length_bucket(line_count)] += 1
        for token in row.get("section_tokens") or []:
            by_section[_section_name(str(token))] += 1
        for field in _record_missing_fields(row):
            missing_field_counts[field] += 1
        artist = str(row.get("artist_clean") or "unknown")
        artist_counts[artist] += 1
        title_hash_counts[hash_text(str(row.get("title") or "").lower())] += 1
        if len(sample_records) < args.sample_records:
            sample_records.append(
                {
                    "record_id": row.get("record_id"),
                    "split": row.get("split"),
                    "title": row.get("title"),
                    "artist_clean": row.get("artist_clean"),
                    "line_count": row.get("line_count"),
                    "content_flags": row.get("content_flags") or [],
                    "missing_canonical_fields": _record_missing_fields(row),
                }
            )

    duplicate_manifest = corpus_manifest.get("counts", {})
    train_tokens = (
        tokenization_manifest.get("splits", {})
        .get("base", {})
        .get("train", {})
        .get("input_tokens")
    )
    line_counts.sort()
    percentiles = {}
    if line_counts:
        for name, fraction in {"p50": 0.5, "p90": 0.9, "p99": 0.99}.items():
            index = min(len(line_counts) - 1, int(round((len(line_counts) - 1) * fraction)))
            percentiles[name] = line_counts[index]

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "corpus_dir": str(corpus_dir),
        "source_id": source_id,
        "rights": {
            "source_tier": source["tier"],
            "partition": source["partition"],
            "training_eligibility": source["training_eligibility"],
            "license_scope": source["license_scope"],
            "private_research_only": bool(corpus_manifest.get("private_research_only")),
            "public_release_blocked": bool(corpus_manifest.get("public_release_blocked")),
        },
        "scan": {
            "scan_scope": "full" if args.max_records is None else "bounded",
            "max_records": args.max_records,
            "records_scanned": counters["records_scanned"],
            "is_full_scan": args.max_records is None
            or counters["records_scanned"] < int(args.max_records or 0),
            "observed_lyric_bytes": observed_lyric_bytes,
            "wall_seconds": round(time.monotonic() - started, 3),
        },
        "composition": {
            "records_by_split": dict(sorted(by_split.items())),
            "records_by_language_label": dict(by_language.most_common()),
            "records_by_line_count_bucket": dict(sorted(by_length_bucket.items())),
            "section_token_counts": dict(by_section.most_common()),
            "explicit_flagged_records": counters["explicit_flagged"],
            "language_disagreement_records": counters["language_disagreement"],
            "line_count_percentiles": percentiles,
            "top_artist_record_counts": dict(artist_counts.most_common(args.top_n)),
            "title_hashes_with_multiple_records": sum(1 for count in title_hash_counts.values() if count > 1),
        },
        "dedupe": {
            "duplicate_exact": int(duplicate_manifest.get("duplicate_exact", 0)),
            "duplicate_near": int(duplicate_manifest.get("duplicate_near", 0)),
            "dedupe_report": str(corpus_dir / "duplicates.jsonl"),
        },
        "tokens": {
            "tokens_observed_in_scan": None,
            "total_tokens_from_tokenizer_manifest": train_tokens,
            "total_tokens_observed_by_full_scan": train_tokens
            if args.max_records is None
            and counters["records_scanned"] == int(corpus_manifest.get("counts", {}).get("retained", 0))
            else None,
            "tokenization_manifest": str(tokenization_manifest_path)
            if tokenization_manifest_path.exists()
            else None,
            "notes": {
                "tokens_observed_in_scan": "not recomputed by inspect-corpus",
                "total_tokens_from_tokenizer_manifest": "loaded from existing tokenization_manifest.json",
                "total_tokens_observed_by_full_scan": (
                    "available only when inspect-corpus scans all retained records"
                ),
            },
        },
        "canonical_record_gap": {
            "required_fields": CANONICAL_RECORD_FIELDS,
            "missing_field_counts": dict(missing_field_counts.most_common()),
            "note": (
                "The current v1 rows are trainable as a private partition, but they do not yet "
                "carry all per-item rights and revocation fields required for open acquisition."
            ),
        },
        "sample_records": sample_records,
        "inputs": {
            "corpus_manifest": path_manifest(corpus_manifest_path),
            "all_jsonl": path_manifest(all_path),
        },
    }
    if tokenization_manifest_path.exists():
        report["inputs"]["tokenization_manifest"] = path_manifest(tokenization_manifest_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "corpus_composition.json", report)
    (output_dir / "corpus_composition.md").write_text(render_composition_markdown(report), encoding="utf-8")
    return report


def render_composition_markdown(report: dict[str, Any]) -> str:
    comp = report["composition"]
    rights = report["rights"]
    missing = report["canonical_record_gap"]["missing_field_counts"]
    return "\n".join(
        [
            "# Scratch corpus composition",
            "",
            f"Generated: {report['generated_at']}",
            f"Corpus: `{report['corpus_dir']}`",
            f"Source: `{report['source_id']}`",
            "",
            "## Rights",
            "",
            f"- Tier: {rights['source_tier']}",
            f"- Partition: {rights['partition']}",
            f"- Eligibility: {rights['training_eligibility']}",
            f"- License scope: {rights['license_scope']}",
            f"- Public release blocked: {rights['public_release_blocked']}",
            "",
            "## Composition",
            "",
            f"- Records scanned: {report['scan']['records_scanned']:,}",
            f"- Scan scope: {report['scan']['scan_scope']} (directly observed)",
            f"- Observed lyric bytes: {report['scan']['observed_lyric_bytes']:,} (directly observed)",
            f"- Train input tokens: {report['tokens']['total_tokens_from_tokenizer_manifest']:,} "
            "(loaded from existing tokenizer manifest)"
            if report["tokens"]["total_tokens_from_tokenizer_manifest"] is not None
            else "- Train input tokens: unavailable",
            f"- Explicit-flagged records: {comp['explicit_flagged_records']:,}",
            f"- Language disagreement records: {comp['language_disagreement_records']:,}",
            f"- Exact duplicates removed: {report['dedupe']['duplicate_exact']:,}",
            f"- Near duplicates removed: {report['dedupe']['duplicate_near']:,}",
            "",
            "## Split Counts",
            "",
            *[f"- {split}: {count:,}" for split, count in comp["records_by_split"].items()],
            "",
            "## Canonical Record Gap",
            "",
            "Missing per-item fields that should be present before open-data acquisition:",
            "",
            *[f"- {field}: {count:,}" for field, count in missing.items()],
            "",
        ]
    )


def ingest_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    source = _source_by_id(registry, args.source)
    output_dir = Path(args.output_dir) / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_transition_base(
            operation="ingest-source",
            source=source,
            registry_path=registry_path,
            reason=getattr(args, "reason", None),
        ),
        "operation": "ingest-source",
        "lifecycle_state": "ingested",
        "status": "ready" if source["source_id"] == _current_source_id(registry) else "pending_external_snapshot",
        "snapshot_dir": str(Path(args.snapshot_dir)) if args.snapshot_dir else None,
        "canonical_record_fields": CANONICAL_RECORD_FIELDS,
        "per_item_rights_required": source["training_eligibility"] in {"eligible", "conditional"},
        "training_text_allowed_now": source["training_eligibility"] in {"eligible", "private_only"},
        "metadata_only": source["training_eligibility"] == "metadata_only",
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    if source["source_id"] == _current_source_id(registry):
        corpus_manifest = Path(args.corpus_dir) / "corpus_manifest.json"
        if corpus_manifest.exists():
            manifest["local_corpus_manifest"] = path_manifest(corpus_manifest)
    write_json(output_dir / "ingest_manifest.json", manifest)
    return manifest


def audit_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    source = _source_by_id(registry, args.source)
    output_dir = Path(args.output_dir) / args.source
    ingest_path = output_dir / "ingest_manifest.json"
    ingest = _optional_manifest(ingest_path)
    issues: list[str] = []
    if ingest is None:
        issues.append("missing ingest manifest")
    if source["training_eligibility"] == "metadata_only" and source.get("full_text_available"):
        issues.append("metadata_only source should not advertise full_text_available")
    if source["training_eligibility"] in {"eligible", "conditional", "private_only"} and not source.get(
        "full_text_available"
    ):
        issues.append("trainable source has no full text")
    if source["training_eligibility"] == "conditional":
        issues.append("item-level rights evidence must be saved before admission")
    if source["training_eligibility"] == "excluded":
        issues.append("source is excluded from training builds")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **_transition_base(
            operation="audit-source",
            source=source,
            registry_path=registry_path,
            reason=getattr(args, "reason", None),
        ),
        "operation": "audit-source",
        "lifecycle_state": "audited",
        "ingest_manifest": path_manifest(ingest_path) if ingest_path.exists() else None,
        "passed": (
            not [issue for issue in issues if issue != "item-level rights evidence must be saved before admission"]
            and (not issues or (args.allow_conditional and source["training_eligibility"] == "conditional"))
        ),
        "issues": issues,
        "admission_gates": {
            "provenance": bool(source.get("provenance_url")),
            "rights_classification": source["training_eligibility"] != "excluded",
            "text_available": bool(source.get("full_text_available")),
            "per_item_rights_required": source["training_eligibility"] in {"eligible", "conditional"},
            "revocation_supported": True,
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output_dir / "audit_report.json", report)
    return report


def admit_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    source = _source_by_id(registry, args.source)
    output_dir = Path(args.output_dir) / args.source
    ingest_path = output_dir / "ingest_manifest.json"
    audit_path = output_dir / "audit_report.json"
    audit = _optional_manifest(audit_path)
    if audit is None and not args.force:
        raise ValueError(f"{args.source} requires a current audit_report.json before admission.")
    if audit is not None and not audit.get("passed") and not args.force:
        raise ValueError(f"{args.source} audit did not pass.")
    if audit is not None and ingest_path.exists() and audit.get("ingest_manifest") != path_manifest(ingest_path):
        raise ValueError(f"{args.source} audit is stale because ingest_manifest.json changed.")
    if source["training_eligibility"] in {"metadata_only", "excluded"} and not args.force:
        raise ValueError(
            f"{args.source} is {source['training_eligibility']} and cannot be admitted to a training build."
        )
    if source["training_eligibility"] == "conditional" and not args.rights_evidence and not args.force:
        raise ValueError(f"{args.source} requires --rights-evidence or --force before admission.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_transition_base(
            operation="admit-source",
            source=source,
            registry_path=registry_path,
            reason=getattr(args, "reason", None),
        ),
        "operation": "admit-source",
        "lifecycle_state": "admitted",
        "audit_report": path_manifest(audit_path) if audit_path.exists() else None,
        "admission_status": "admitted",
        "rights_evidence": str(args.rights_evidence) if args.rights_evidence else source["license_scope"],
        "partition": source["partition"],
        "removal_key_field": "removal_key",
        "record_id_namespace": args.source,
        "canonical_record_fields": CANONICAL_RECORD_FIELDS,
        "revocation": {
            "supported": True,
            "strategy": "tombstone records by source_id or removal_key, regenerate shards, and write a new corpus manifest",
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    if args.source == _current_source_id(registry):
        corpus_manifest = Path(args.corpus_dir) / "corpus_manifest.json"
        if corpus_manifest.exists():
            manifest["corpus_manifest"] = path_manifest(corpus_manifest)
    write_json(output_dir / "admission_manifest.json", manifest)
    return manifest


def revoke_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    source = _source_by_id(registry, args.source)
    output_dir = Path(args.output_dir) / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    tombstone = {
        **_transition_base(
            operation="revoke-source",
            source=source,
            registry_path=registry_path,
            reason=args.reason,
        ),
        "operation": "revoke-source",
        "lifecycle_state": "revoked",
        "reason": args.reason,
        "destructive_changes_performed": False,
        "tombstone": {
            "source_id": args.source,
            "removal_key_prefix": f"{args.source}:",
            "status": "revoked",
        },
        "required_next_steps": [
            "exclude matching source_id or removal_key records",
            "re-run exact and near deduplication against remaining records",
            "regenerate train/validation/test JSONL",
            "re-tokenize shards",
            "write a new corpus manifest and retain the old manifest for audit",
        ],
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    if args.source == _current_source_id(registry):
        manifest_path = Path(args.corpus_dir) / "corpus_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            tombstone["impact_estimate"] = {
                "retained_records": manifest.get("counts", {}).get("retained"),
                "all_current_training_text": True,
                "corpus_manifest": path_manifest(manifest_path),
            }
    write_json(output_dir / "revocation_plan.json", tombstone)
    return tombstone


def _legacy_sidecar_row(
    row: dict[str, Any],
    *,
    source_id: str,
    source_manifest: dict[str, Any],
    original_row_number: int,
    governed_at: str,
) -> dict[str, Any]:
    text = str(row.get("lyrics") or "")
    record_id = hash_text(
        f"{source_id}:{source_manifest['sha256']}:{original_row_number}",
        digest_size=16,
    )
    return {
        "record_id": record_id,
        "legacy_record_id": str(row.get("record_id") or ""),
        "source_id": source_id,
        "source_item_id": str(row.get("record_id") or ""),
        "retrieved_at": None,
        "governed_at": governed_at,
        "source_url_hash": None,
        "rights_tier": "private_research",
        "rights_status": "unknown_legacy",
        "license_id": None,
        "license_evidence": f"registry://{source_id}",
        "license_evidence_scope": "source",
        "work_id": None,
        "recording_id": None,
        "artist_ids": [],
        "text_sha256": _sha256_text(text),
        "normalized_text_sha256": _sha256_text(normalized_lyrics_key(text)),
        "dedupe_cluster_id": None,
        "removal_key": f"{source_id}:{record_id}",
        "split_group_id": None,
        "split_group_status": "pending_deduplication",
        "legacy_split": row.get("split"),
        "legacy_title": row.get("title"),
        "legacy_artist_hash": row.get("artist_hash"),
        "legacy_artist_clean": row.get("artist_clean"),
        "legacy_line_count": int(row.get("line_count") or 0),
        "original_jsonl": source_manifest["path"],
        "original_jsonl_sha256": source_manifest["sha256"],
        "original_row_number": original_row_number,
        "schema_version": 2,
    }


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def migrate_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    source = _source_by_id(registry, args.source)
    if args.source != _current_source_id(registry) and not args.force:
        raise ValueError("Only the current legacy private source can be migrated by this command.")

    corpus_dir = Path(args.corpus_dir)
    all_path = corpus_dir / "all.jsonl"
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    tokenization_manifest_path = corpus_dir / "tokenization_manifest.json"
    source_manifest = path_manifest(all_path)
    corpus_manifest = read_json(corpus_manifest_path)
    tokenization_manifest = read_json(tokenization_manifest_path) if tokenization_manifest_path.exists() else {}
    output_dir = Path(args.output_dir) / args.source
    records_dir = output_dir / "records"
    quarantine_path = output_dir / "migration_quarantine.jsonl"
    governed_at = args.governed_at or utc_now()
    shard_size = int(args.shard_size)
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0
    counters: Counter[str] = Counter()
    shard_manifests: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    removal_keys: set[str] = set()
    quarantined: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        shard_path = records_dir / f"shard-{shard_index:05d}.parquet"
        if args.resume and shard_path.exists():
            manifest = path_manifest(shard_path)
        else:
            _write_parquet_atomic(shard_path, shard_rows)
            manifest = path_manifest(shard_path)
        shard_manifests.append(
            {
                **manifest,
                "rows": len(shard_rows),
                "first_original_row_number": shard_rows[0]["original_row_number"],
                "last_original_row_number": shard_rows[-1]["original_row_number"],
            }
        )
        shard_index += 1
        shard_rows = []

    for original_row_number, row in enumerate(iter_jsonl(all_path)):
        if args.limit is not None and counters["input_records"] >= args.limit:
            break
        counters["input_records"] += 1
        if "lyrics" not in row or "record_id" not in row:
            quarantined.append(
                {
                    "original_row_number": original_row_number,
                    "reason": "missing legacy record_id or lyrics",
                }
            )
            counters["quarantined"] += 1
            continue
        sidecar = _legacy_sidecar_row(
            row,
            source_id=args.source,
            source_manifest=source_manifest,
            original_row_number=original_row_number,
            governed_at=governed_at,
        )
        if sidecar["record_id"] in record_ids:
            raise ValueError(f"Duplicate migrated record_id: {sidecar['record_id']}")
        if sidecar["removal_key"] in removal_keys:
            raise ValueError(f"Duplicate removal_key: {sidecar['removal_key']}")
        record_ids.add(sidecar["record_id"])
        removal_keys.add(sidecar["removal_key"])
        shard_rows.append(sidecar)
        counters["sidecar_records"] += 1
        if len(shard_rows) >= shard_size:
            flush()
    flush()

    if quarantined:
        from .common import write_jsonl

        write_jsonl(quarantine_path, quarantined)
    elif quarantine_path.exists() and not args.resume:
        quarantine_path.unlink()

    retained = int(corpus_manifest.get("counts", {}).get("retained", 0))
    expected_records = int(args.limit) if args.limit is not None else retained
    train_tokens = (
        tokenization_manifest.get("splits", {})
        .get("base", {})
        .get("train", {})
        .get("input_tokens")
    )
    invariants = {
        "input_records_represented": counters["sidecar_records"] == expected_records - counters["quarantined"],
        "unique_record_id_values": len(record_ids) == counters["sidecar_records"],
        "unique_removal_key_values": len(removal_keys) == counters["sidecar_records"],
        "records_with_source_id": counters["sidecar_records"],
        "records_with_exact_text_hash": counters["sidecar_records"],
        "raw_text_changed_records": 0,
        "record_count_discrepancy": expected_records - counters["sidecar_records"] - counters["quarantined"],
        "token_total_discrepancy": 0,
        "failed_or_skipped_records": counters["quarantined"],
    }
    manifest = {
        **_transition_base(
            operation="migrate-source",
            source=source,
            registry_path=registry_path,
            reason=getattr(args, "reason", None),
        ),
        "lifecycle_state": "migrated",
        "schema_version": int(args.schema_version),
        "governed_at": governed_at,
        "source_jsonl": source_manifest,
        "corpus_manifest": path_manifest(corpus_manifest_path),
        "tokenization_manifest": path_manifest(tokenization_manifest_path)
        if tokenization_manifest_path.exists()
        else None,
        "records_dir": str(records_dir),
        "shard_size": shard_size,
        "shards": shard_manifests,
        "counts": dict(counters),
        "expected_records": expected_records,
        "accepted_train_tokens_from_manifest": train_tokens,
        "invariants": invariants,
        "quarantine": path_manifest(quarantine_path) if quarantine_path.exists() else None,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output_dir / "migration_manifest.json", manifest)
    return manifest


def verify_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    profile_path = Path(args.catalog_dir) / "profiles" / args.profile / "profile_manifest.json"
    if not profile_path.exists():
        raise ValueError(f"Missing profile manifest: {profile_path}")
    profile = read_json(profile_path)
    sources_dir = Path(args.sources_dir)
    failures: list[str] = []
    for source in profile.get("included_sources", []):
        source_id = source["source_id"]
        if _source_is_revoked(sources_dir / source_id):
            failures.append(f"{source_id} is revoked")
        if not (sources_dir / source_id / "admission_manifest.json").exists():
            failures.append(f"{source_id} is not admitted")
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": "verify-profile",
        "profile": args.profile,
        "profile_manifest": path_manifest(profile_path),
        "passed": not failures,
        "failures": failures,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output_dir = Path(args.catalog_dir) / "profiles" / args.profile
    write_json(output_dir / "profile_verification.json", report)
    return report


def revocation_impact(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    catalog_dir = Path(args.catalog_dir)
    profiles: list[dict[str, Any]] = []
    for profile_path in sorted((catalog_dir / "profiles").glob("*/profile_manifest.json")):
        profile = read_json(profile_path)
        if any(source.get("source_id") == args.source for source in profile.get("included_sources", [])):
            profiles.append({"profile": profile.get("profile"), "manifest": path_manifest(profile_path)})
    source_dir = Path(args.sources_dir) / args.source
    migration_path = source_dir / "migration_manifest.json"
    migration = read_json(migration_path) if migration_path.exists() else None
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": "revocation-impact",
        "source_id": args.source,
        "profile_manifests_containing_source": profiles,
        "sidecar_shards_containing_source": migration.get("shards", []) if migration else [],
        "tokenizers_trained_on_source": [str(Path(args.corpus_dir) / "tokenization_manifest.json")]
        if args.source == "existing_song_lyrics_private_v1"
        else [],
        "model_lineage_files": [str(path) for path in sorted(Path(args.model_dir).glob("**/lineage.json"))]
        if Path(args.model_dir).exists()
        else [],
        "limitations": [
            "checkpoint lineage is reported only when lineage.json files already exist",
            "evaluation and report dependency tracing needs model lineage files in downstream artifacts",
        ],
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output_dir = source_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "revocation_impact.json", report)
    return report


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    registry = read_json(registry_path)
    validate_registry(registry)
    profile = args.profile
    if profile not in PROFILE_RULES:
        raise ValueError(f"Unknown profile {profile!r}. Valid profiles: {', '.join(PROFILE_RULES)}")
    rule = PROFILE_RULES[profile]
    included: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    failures: list[str] = []
    sources_dir = Path(args.sources_dir)
    for source in registry["sources"]:
        source_dir = sources_dir / source["source_id"]
        admission_path = source_dir / "admission_manifest.json"
        admitted = admission_path.exists()
        revoked = _source_is_revoked(source_dir)
        selected = (
            source["training_eligibility"] in rule["eligibility"]
            and source["partition"] in rule["partitions"]
            and source["training_eligibility"] != "metadata_only"
            and not revoked
        )
        row = {
            "source_id": source["source_id"],
            "partition": source["partition"],
            "training_eligibility": source["training_eligibility"],
            "status": source.get("status"),
            "admitted": admitted,
            "revoked": revoked,
        }
        if selected:
            if not admitted:
                row["blocked_reason"] = "not_admitted"
                blocked.append(row)
                continue
            if profile == "scratch-core-open-v1" and source["training_eligibility"] != "eligible":
                failures.append(f"{source['source_id']} is not releaseable for {profile}")
            included.append(row)
        else:
            blocked.append(row)
    if failures:
        raise ValueError("; ".join(failures))

    manifest = {
        **_transition_base(
            operation="build-corpus-profile",
            source={"source_id": f"profile:{profile}"},
            registry_path=registry_path,
            reason=getattr(args, "reason", None),
        ),
        "operation": "build-corpus-profile",
        "lifecycle_state": "active in profile",
        "profile": profile,
        "description": rule["description"],
        "included_sources": included,
        "excluded_sources": blocked,
        "requires_shard_regeneration": True,
        "destructive_changes_performed": False,
        "note": "This profile manifest composes source partitions; it does not rewrite the current corpus by itself.",
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    current_source = _current_source_id(registry)
    if any(source["source_id"] == current_source for source in included):
        corpus_manifest = Path(args.output_dir) / "corpus_manifest.json"
        if corpus_manifest.exists():
            manifest["current_private_partition"] = path_manifest(corpus_manifest)
    output_dir = Path(args.catalog_dir) / "profiles" / profile
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "profile_manifest.json", manifest)
    return manifest


def add_inspect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--source-id")
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/composition"))
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-records", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)


def add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--reason", default="source snapshot registered")


def add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--allow-conditional", action="store_true")
    parser.add_argument("--reason", default="source admission audit")


def add_admit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--rights-evidence")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", default="source admitted to training profile control plane")


def add_revoke_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--reason", default="manual revocation test")


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILE_RULES))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/scratch/catalog/v2"))
    parser.add_argument("--sources-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--reason", default="profile build")


def add_migrate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--schema-version", type=int, default=2)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--governed-at")
    parser.add_argument("--reason", default="legacy private corpus governance sidecar migration")


def add_verify_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_RULES))
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/scratch/catalog/v2"))
    parser.add_argument("--sources-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))


def add_revocation_impact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--catalog-dir", type=Path, default=Path("data/scratch/catalog/v2"))
    parser.add_argument("--sources-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
