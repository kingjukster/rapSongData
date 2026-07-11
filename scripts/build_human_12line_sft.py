#!/usr/bin/env python3
"""DEPRECATED: rebuild the legacy human-reviewed SFT package for history only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.build_manual_rank_app import (
        app_source_sha256,
        candidate_payload,
        review_target_fingerprint,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_manual_rank_app import app_source_sha256, candidate_payload, review_target_fingerprint


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
REQUIRED_DIMENSIONS = {
    "theme_adherence",
    "specific_imagery",
    "rhyme_cadence",
    "originality",
    "scene_coherence",
    "naturalness",
    "ending_payoff",
}
CREATIVE_DIMENSIONS = {"specific_imagery", "rhyme_cadence", "originality", "ending_payoff"}
DEFAULT_REVIEW_SESSION_ID = "qwen3-4b-12line-human-v1-review-001"
DEFAULT_REVIEW_APP_VERSION = "manual_rank_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/curation/qwen3_4b_12line_human_v1/review_queue.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/qwen3_4b_12line_human_v1"),
    )
    parser.add_argument("--minimum-eligible", type=int, default=100)
    parser.add_argument("--rubric-version", default="rap_12line_quality_v1")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--review-session-id", default=DEFAULT_REVIEW_SESSION_ID)
    parser.add_argument("--review-app-version", default=DEFAULT_REVIEW_APP_VERSION)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_reviews(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected review array at {path}")
    return [row for row in value if isinstance(row, dict)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def lyric_lines(value: str) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def stable_int(*parts: str) -> int:
    return int(hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def chatml(prompt: str, lyrics: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{lyrics.strip()}<|im_end|>\n"
    )


def dimension_values(review: dict[str, Any]) -> dict[str, int]:
    raw = review.get("dimensions") if isinstance(review.get("dimensions"), dict) else {}
    values: dict[str, int] = {}
    for name in REQUIRED_DIMENSIONS:
        try:
            values[name] = int(raw.get(name))
        except (TypeError, ValueError):
            values[name] = 0
    return values


def valid_timestamp(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def eligibility_reasons(
    review: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rubric_version: str,
    expected_review_target: str,
    expected_app_source_hash: str,
    expected_review_session_id: str,
) -> list[str]:
    reasons: list[str] = []
    lyrics = str(candidate.get("lyrics") or candidate.get("generated_text") or "")
    dimensions = dimension_values(review)
    structural = candidate.get("structural_metrics") if isinstance(candidate.get("structural_metrics"), dict) else {}
    provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), dict) else {}
    if review.get("reviewer_type") != "human" or review.get("label_source") != "human_entered":
        reasons.append("not_human_entered")
    if review.get("human_attested") is not True:
        reasons.append("missing_human_attestation")
    if not str(review.get("reviewer_id") or "").strip():
        reasons.append("missing_reviewer_id")
    if not str(review.get("review_id") or "").strip():
        reasons.append("missing_review_id")
    if str(review.get("session_id") or "") != expected_review_session_id:
        reasons.append("review_session_id_mismatch")
    if not str(review.get("reviewed_at") or "").strip():
        reasons.append("missing_reviewed_at")
    elif not valid_timestamp(review.get("reviewed_at")):
        reasons.append("invalid_reviewed_at")
    if str(review.get("rubric_version") or "") != rubric_version:
        reasons.append("rubric_version_mismatch")
    if review.get("decision") != "keep":
        reasons.append("decision_not_keep")
    if review.get("blinded") is not True:
        reasons.append("review_not_blinded")
    if str(review.get("review_target_fingerprint") or "") != expected_review_target:
        reasons.append("review_target_fingerprint_mismatch")
    if str(review.get("app_source_sha256") or "") != expected_app_source_hash:
        reasons.append("review_app_source_mismatch")
    try:
        if int(review.get("manual_rating") or 0) not in {4, 5}:
            reasons.append("overall_rating_not_4_or_5")
    except (TypeError, ValueError):
        reasons.append("invalid_overall_rating")
    if set(dimensions) != REQUIRED_DIMENSIONS or any(value < 3 or value > 5 for value in dimensions.values()):
        reasons.append("dimension_outside_3_to_5_or_missing")
    if dimensions.get("scene_coherence", 0) < 4 or dimensions.get("naturalness", 0) < 4:
        reasons.append("coherence_or_naturalness_below_4")
    if sum(dimensions.get(name, 0) >= 4 for name in CREATIVE_DIMENSIONS) < 2:
        reasons.append("insufficient_creative_strengths")
    if len(lyric_lines(lyrics)) != 12:
        reasons.append("not_exactly_12_lines")
    if int(structural.get("slur_count") or 0) > 0:
        reasons.append("slur_violation")
    if bool(structural.get("prompt_leakage")):
        reasons.append("prompt_leakage")
    if bool(structural.get("high_copy_similarity")):
        reasons.append("high_copy_similarity")
    if str(review.get("edited_lyrics") or "").strip():
        reasons.append("edited_text_requires_second_review")
    expected_hash = str(provenance.get("normalized_text_sha256") or "")
    reviewed_hash = str(review.get("reviewed_text_sha256") or "")
    actual_hash = sha256_text(normalize_text(lyrics))
    if not expected_hash or reviewed_hash != expected_hash or actual_hash != expected_hash:
        reasons.append("reviewed_text_hash_mismatch")
    exported_lyrics_hash = sha256_text(normalize_text(str(review.get("lyrics") or "")))
    if exported_lyrics_hash != expected_hash:
        reasons.append("exported_review_text_mismatch")
    candidate_prompt_key = str(candidate.get("prompt_key") or "")
    if not candidate_prompt_key or str(review.get("prompt_key") or "") != candidate_prompt_key:
        reasons.append("review_prompt_key_mismatch")
    if normalize_text(str(review.get("prompt") or "")) != normalize_text(str(candidate.get("prompt") or "")):
        reasons.append("review_prompt_mismatch")
    rng_provenance = (
        provenance.get("rng_provenance") if isinstance(provenance.get("rng_provenance"), dict) else {}
    )
    provenance_requirements = {
        "candidate_id": str(provenance.get("candidate_id") or "") == str(candidate.get("candidate_id") or ""),
        "license_scope": bool(str(provenance.get("license_scope") or "").strip()),
        "judge_source_sha256": bool(str(provenance.get("judge_source_sha256") or "").strip()),
        "generation_source_sha256": bool(str(provenance.get("generation_source_sha256") or "").strip()),
        "generation_summary_sha256": bool(str(provenance.get("generation_summary_sha256") or "").strip()),
        "generation_run_id": bool(str(provenance.get("generation_run_id") or "").strip()),
        "judge_run_id": bool(str(provenance.get("judge_run_id") or "").strip()),
        "model_id": bool(str(provenance.get("model_id") or "").strip()),
        "model_revision": bool(str(provenance.get("model_revision") or "").strip()),
        "model_revision_source": bool(str(provenance.get("model_revision_source") or "").strip()),
        "rng_protocol": bool(str(rng_provenance.get("protocol") or "").strip()),
        "rng_manual_seed": isinstance(rng_provenance.get("manual_seed"), int),
        "created_at": valid_timestamp(provenance.get("created_at")),
    }
    for field, present in provenance_requirements.items():
        if not present:
            reasons.append(f"missing_or_invalid_provenance_{field}")
    return sorted(set(reasons))


def near_duplicate(left: str, right: str, threshold: float) -> bool:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right), autojunk=False).ratio() >= threshold


def deduplicate(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["review"].get("manual_rating") or 0),
            sum(dimension_values(row["review"]).values()),
            str(row["candidate_id"]),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row in ordered:
        duplicate_of = next(
            (
                existing
                for existing in kept
                if normalize_text(row["lyrics"]) == normalize_text(existing["lyrics"])
                or near_duplicate(row["lyrics"], existing["lyrics"], threshold)
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(row)
        else:
            removed.append(
                {
                    "candidate_id": row["candidate_id"],
                    "duplicate_of": duplicate_of["candidate_id"],
                }
            )
    return kept, removed


def split_prompt_groups(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        theme = normalize_text(str(row.get("theme") or ""))
        group_key = f"theme:{theme}" if theme else f"prompt:{row['prompt_key']}"
        by_prompt[group_key].append(row)
    groups = sorted(by_prompt.items(), key=lambda item: stable_int(str(seed), item[0]))
    if len(groups) < 3:
        raise ValueError("At least three unique theme/prompt groups are required for train/validation/test splits")
    validation_groups = max(1, round(len(groups) * 0.10))
    test_groups = max(1, round(len(groups) * 0.10))
    train_groups = len(groups) - validation_groups - test_groups
    if train_groups < 1:
        raise ValueError("Not enough prompt groups to retain a training split")
    assignments = {
        key: "train" if index < train_groups else "validation" if index < train_groups + validation_groups else "test"
        for index, (key, _) in enumerate(groups)
    }
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for key, group_rows in by_prompt.items():
        splits[assignments[key]].extend(group_rows)
    for split in splits:
        splits[split].sort(key=lambda row: stable_int(str(seed), split, row["candidate_id"]))
    return splits


def training_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    review = row["review"]
    provenance = row["candidate"].get("provenance") or {}
    return {
        "id": f"human-v1-{row['candidate_id']}",
        "training_text": chatml(row["prompt"], row["lyrics"]),
        "metadata": {
            "source": "qwen3_4b_12line_human_v1",
            "source_bucket": "verified_human_keep",
            "candidate_id": row["candidate_id"],
            "prompt_key": row["prompt_key"],
            "theme": row.get("theme"),
            "prompt_family": row.get("prompt_family"),
            "split": split,
            "target_line_count": 12,
            "actual_line_count": 12,
            "license_scope": provenance.get("license_scope"),
            "normalized_text_sha256": provenance.get("normalized_text_sha256"),
            "source_provenance": provenance,
            "human_review": {
                "review_id": review.get("review_id"),
                "reviewer_id": review.get("reviewer_id"),
                "reviewer_type": "human",
                "label_source": "human_entered",
                "human_attested": True,
                "reviewed_at": review.get("reviewed_at"),
                "session_id": review.get("session_id"),
                "rubric_version": review.get("rubric_version"),
                "overall_rating": int(review.get("manual_rating")),
                "decision": "keep",
                "dimensions": dimension_values(review),
                "issue_tags": review.get("issue_tags") or [],
                "notes": review.get("notes") or "",
                "review_duration_seconds": review.get("review_duration_seconds"),
                "blinded": bool(review.get("blinded")),
                "app_version": review.get("app_version"),
                "app_commit_sha": review.get("app_commit_sha"),
                "app_source_sha256": review.get("app_source_sha256"),
                "review_target_fingerprint": review.get("review_target_fingerprint"),
            },
            "format": "qwen_chatml_training_text",
        },
    }


def main() -> int:
    print(
        "DEPRECATED: use build_auto_calibrated_12line_sft.py; human review is retired.",
        file=sys.stderr,
    )
    args = parse_args()
    queue_rows = read_jsonl(args.queue)
    queue_by_id = {str(row.get("candidate_id") or row.get("row_id")): row for row in queue_rows}
    reviews = read_reviews(args.reviews)
    review_candidates = [candidate_payload(row, rank=index) for index, row in enumerate(queue_rows, start=1)]
    expected_app_source_hash = app_source_sha256()
    expected_review_target = review_target_fingerprint(
        review_candidates,
        rubric_version=args.rubric_version,
        review_session_id=args.review_session_id,
        app_version=args.review_app_version,
        app_source_hash=expected_app_source_hash,
    )
    rejection_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for review in reviews:
        row_id = str(review.get("candidate_id") or "")
        candidate = queue_by_id.get(row_id)
        if candidate is None:
            rejection_counts["candidate_not_in_queue"] += 1
            continue
        reasons = eligibility_reasons(
            review,
            candidate,
            rubric_version=args.rubric_version,
            expected_review_target=expected_review_target,
            expected_app_source_hash=expected_app_source_hash,
            expected_review_session_id=args.review_session_id,
        )
        if reasons:
            rejection_counts.update(reasons)
            continue
        eligible.append(
            {
                "candidate_id": row_id,
                "prompt_key": str(candidate.get("prompt_key")),
                "prompt": str(candidate.get("prompt") or ""),
                "lyrics": str(candidate.get("lyrics") or candidate.get("generated_text") or ""),
                "theme": str(candidate.get("theme") or ""),
                "prompt_family": str(candidate.get("prompt_family") or ""),
                "candidate": candidate,
                "review": review,
            }
        )

    unique, duplicate_rows = deduplicate(eligible, args.near_duplicate_threshold)
    if len(unique) < args.minimum_eligible:
        raise SystemExit(
            f"Only {len(unique)} unique eligible human-reviewed rows; minimum is {args.minimum_eligible}. "
            f"Rejections: {json.dumps(dict(rejection_counts), sort_keys=True)}"
        )
    splits = split_prompt_groups(unique, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for split, rows in splits.items():
        output = args.output_dir / f"{split}.jsonl"
        write_jsonl(output, [training_record(row, split) for row in rows])
        output_paths[split] = str(output)

    split_prompts = {split: {row["prompt_key"] for row in rows} for split, rows in splits.items()}
    split_themes = {
        split: {normalize_text(str(row.get("theme") or "")) for row in rows if row.get("theme")}
        for split, rows in splits.items()
    }
    manifest = {
        "name": "qwen3_4b_12line_human_v1",
        "status": "training_ready",
        "reviews_path": str(args.reviews),
        "reviews_sha256": sha256_file(args.reviews),
        "queue_path": str(args.queue),
        "queue_sha256": sha256_file(args.queue),
        "rubric_version": args.rubric_version,
        "review_session_id": args.review_session_id,
        "review_app_version": args.review_app_version,
        "review_app_source_sha256": expected_app_source_hash,
        "review_target_fingerprint": expected_review_target,
        "seed": args.seed,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "counts": {
            "review_rows": len(reviews),
            "eligible_before_dedup": len(eligible),
            "duplicate_rows_removed": len(duplicate_rows),
            "verified_human_unique": len(unique),
            "train_rows": len(splits["train"]),
            "validation_rows": len(splits["validation"]),
            "test_rows": len(splits["test"]),
            "prompt_key_count": len({row["prompt_key"] for row in unique}),
            "reviewer_count": len({str(row["review"].get("reviewer_id")) for row in unique}),
        },
        "rejection_counts": dict(rejection_counts),
        "removed_duplicates": duplicate_rows,
        "prompt_split_overlap": {
            "train_validation": len(split_prompts["train"] & split_prompts["validation"]),
            "train_test": len(split_prompts["train"] & split_prompts["test"]),
            "validation_test": len(split_prompts["validation"] & split_prompts["test"]),
        },
        "theme_split_overlap": {
            "train_validation": len(split_themes["train"] & split_themes["validation"]),
            "train_test": len(split_themes["train"] & split_themes["test"]),
            "validation_test": len(split_themes["validation"] & split_themes["test"]),
        },
        "paths": output_paths,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
