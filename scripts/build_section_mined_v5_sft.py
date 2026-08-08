#!/usr/bin/env python3
"""Build balanced v5 SFT with strict story, technical, and clean sections."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_section_mined_v4_sft import (
        FAMILIES, SPLIT_COUNTS, assistant_text, assign_splits, rank_key, read_jsonl,
        sha256_file, source_training_record, stable_int, text_sha256, write_jsonl,
    )
except ImportError:
    from build_section_mined_v4_sft import (
        FAMILIES, SPLIT_COUNTS, assistant_text, assign_splits, rank_key, read_jsonl,
        sha256_file, source_training_record, stable_int, text_sha256, write_jsonl,
    )


def choose_preferred_unique_sources(
    rows: list[dict[str, Any]], family: str, quota: int,
    excluded_songs: set[str], avoid_songs: set[str],
) -> list[dict[str, Any]]:
    candidates = sorted(
        (row for row in rows if row["family"] == family),
        key=lambda row: (str(row["song_key"]) in avoid_songs, rank_key(row, family)),
    )
    chosen: list[dict[str, Any]] = []
    used = set(excluded_songs)
    for row in candidates:
        song_key = str(row["song_key"])
        if song_key in used:
            continue
        if not row.get("computed_strict_pass") or row.get("critical_failure_flags") not in ([], ["none"]):
            continue
        if len([line for line in str(row["text"]).splitlines() if line.strip()]) != 12:
            continue
        chosen.append(row)
        used.add(song_key)
        if len(chosen) >= quota:
            return chosen
    raise RuntimeError(f"Only {len(chosen)} unique-source {family} rows available; need {quota}")


def v5_source_record(
    row: dict[str, Any], split: str, accepted_path: Path, accepted_sha: str,
    summary_path: Path, summary_sha: str, seed: int,
) -> dict[str, Any]:
    record = source_training_record(row, split, accepted_path, accepted_sha, summary_path, summary_sha, seed)
    record["id"] = record["id"].replace("section-v4-", "section-v5-")
    metadata = record["metadata"]
    metadata["source"] = "qwen3_4b_12line_section_mined_v5"
    metadata["source_bucket"] = "gpt55_strict_v5_section_keep"
    metadata["selection"] = {"family_quota": 33, "one_section_per_source_song": True, "cross_family_source_exclusion": True}
    metadata["source_provenance"]["generation_run_id"] = "source_song_section_mining_v5"
    metadata["source_provenance"]["judge_run_id"] = "gpt55_section_quality_v5"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=Path("data/training/qwen3_4b_12line_auto_calibrated_v3"))
    parser.add_argument("--technical-clean", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_combined_accepted_capped.jsonl"))
    parser.add_argument("--technical-clean-summary", type=Path, default=Path("reports/section_quality_judge_gpt55_combined_analysis.json"))
    parser.add_argument("--story", type=Path, default=Path("data/reviews/story_section_quality_judge_gpt55_v5_combined_accepted_capped.jsonl"))
    parser.add_argument("--story-summary", type=Path, default=Path("reports/story_section_quality_judge_gpt55_v5_combined_analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v5"))
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    started = time.time()

    tc_rows = read_jsonl(args.technical_clean)
    story_rows = read_jsonl(args.story)
    all_source = tc_rows + story_rows
    songs = {
        family: {str(row["song_key"]) for row in all_source if row["family"] == family}
        for family in ("technical", "clean", "story")
    }

    technical = choose_preferred_unique_sources(
        all_source, "technical", 33, set(), songs["story"] | songs["clean"],
    )
    used = {str(row["song_key"]) for row in technical}
    story = choose_preferred_unique_sources(all_source, "story", 33, used, songs["clean"])
    used.update(str(row["song_key"]) for row in story)
    clean = choose_preferred_unique_sources(all_source, "clean", 33, used, set())
    selected = {"story": story, "technical": technical, "clean": clean}
    source_splits = {family: assign_splits(rows, family, args.seed) for family, rows in selected.items()}

    tc_sha = sha256_file(args.technical_clean)
    tc_summary_sha = sha256_file(args.technical_clean_summary)
    story_sha = sha256_file(args.story)
    story_summary_sha = sha256_file(args.story_summary)
    outputs: dict[str, list[dict[str, Any]]] = {}
    retained_candidate_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        v3_rows = read_jsonl(args.v3_dir / f"{split}.jsonl")
        retained = [row for row in v3_rows if row.get("metadata", {}).get("prompt_family") == "melodic"]
        retained_candidate_ids.update(str(row.get("metadata", {}).get("candidate_id")) for row in retained)
        for row in retained:
            row["metadata"] = dict(row["metadata"])
            row["metadata"]["v5_role"] = "retained_melodic_control"
            row["metadata"]["legacy_eval_policy"] = "expanded_eval_retired; use frozen_v5_confirmation_only"
        replacements = []
        for family in ("story", "technical", "clean"):
            accepted_path = args.story if family == "story" else args.technical_clean
            accepted_sha = story_sha if family == "story" else tc_sha
            summary_path = args.story_summary if family == "story" else args.technical_clean_summary
            summary_sha = story_summary_sha if family == "story" else tc_summary_sha
            replacements.extend(
                v5_source_record(row, split, accepted_path, accepted_sha, summary_path, summary_sha, args.seed)
                for row in source_splits[family][split]
            )
        outputs[split] = sorted(retained + replacements, key=lambda row: stable_int(str(args.seed), split, row["id"]))

    all_rows = [row for rows in outputs.values() for row in rows]
    text_hashes = [text_sha256(assistant_text(row["training_text"])) for row in all_rows]
    if len(text_hashes) != len(set(text_hashes)):
        raise RuntimeError("Duplicate assistant text detected in v5")
    source_songs_by_split = {
        split: {str(row["metadata"].get("source_song_key")) for row in rows if row["metadata"].get("source_song_key")}
        for split, rows in outputs.items()
    }
    if any(source_songs_by_split[left] & source_songs_by_split[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Source-song leakage across v5 splits")
    selected_source_songs = [str(row["song_key"]) for rows in selected.values() for row in rows]
    if len(selected_source_songs) != len(set(selected_source_songs)):
        raise RuntimeError("Source-song leakage across v5 families")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, rows in outputs.items():
        paths[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(paths[split], rows)
    v3_pairs = read_jsonl(args.v3_dir / "preference_pairs.jsonl")
    pairs = [pair for pair in v3_pairs if str(pair.get("metadata", {}).get("chosen_candidate_id")) in retained_candidate_ids]
    for pair in pairs:
        pair["metadata"] = dict(pair["metadata"])
        pair["metadata"]["v5_role"] = "retained_melodic_preference_control"
    paths["preferences"] = args.output_dir / "preference_pairs.jsonl"
    write_jsonl(paths["preferences"], pairs)

    family_by_split = {
        split: dict(sorted(Counter(row["metadata"]["prompt_family"] for row in rows).items()))
        for split, rows in outputs.items()
    }
    manifest = {
        "name": "qwen3_4b_12line_section_mined_v5", "status": "training_ready",
        "dataset_version": "v5",
        "selection_policy": "retain_melodic_control_replace_story_technical_clean_with_gpt55_strict_unique_source_sections",
        "legacy_eval_policy": "expanded_eval_and_router_confirmation_retired_for_promotion; freeze_v5_confirmation_before_training",
        "seed": args.seed, "wall_time_seconds": round(time.time() - started, 3),
        "command": " ".join([sys.executable, *sys.argv]),
        "family_quotas": {family: 33 for family in FAMILIES}, "split_targets_per_family": SPLIT_COUNTS,
        "source_inputs": {
            "v3_manifest": {"path": str(args.v3_dir / "manifest.json"), "sha256": sha256_file(args.v3_dir / "manifest.json")},
            "technical_clean": {"path": str(args.technical_clean), "sha256": tc_sha},
            "story": {"path": str(args.story), "sha256": story_sha},
        },
        "counts": {"selected": len(all_rows), **{f"{split}_rows": len(rows) for split, rows in outputs.items()}, "preference_pairs": len(pairs)},
        "family_presence_by_split": family_by_split,
        "source_song_overlap": {
            "cross_family": len(selected_source_songs) - len(set(selected_source_songs)),
            "train_validation": len(source_songs_by_split["train"] & source_songs_by_split["validation"]),
            "train_test": len(source_songs_by_split["train"] & source_songs_by_split["test"]),
            "validation_test": len(source_songs_by_split["validation"] & source_songs_by_split["test"]),
        },
        "duplicate_assistant_rows": len(text_hashes) - len(set(text_hashes)),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest["output_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
