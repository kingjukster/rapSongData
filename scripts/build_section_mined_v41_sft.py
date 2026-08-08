#!/usr/bin/env python3
"""Build v4.1 by adding every unused strict melodic/story anchor to v4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import build_auto_calibrated_12line_sft as v3
from scripts.build_section_mined_v4_sft import (
    assistant_text,
    read_jsonl,
    sha256_file,
    stable_int,
    text_sha256,
    write_jsonl,
)


STRONG_FAMILIES = {"melodic", "story"}


def normalized_family(row: dict[str, Any]) -> str:
    return v3.normalize(row.get("prompt_family"))


def theme_split_map(v3_dir: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(v3_dir / f"{split}.jsonl"):
            theme = v3.normalize(row.get("metadata", {}).get("theme"))
            if not theme:
                continue
            previous = assignments.setdefault(theme, split)
            if previous != split:
                raise RuntimeError(f"Theme {theme!r} appears in both {previous} and {split}")
    return assignments


def choose_unused_strong_rows(
    judged: list[dict[str, Any]], used_candidate_ids: set[str], min_score: float
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in judged
        if normalized_family(row) in STRONG_FAMILIES
        and v3.candidate_id(row) not in used_candidate_ids
        and v3.consensus_eligible(row, min_score)
    ]
    return sorted(
        rows,
        key=lambda row: (
            normalized_family(row),
            -v3.score(row),
            v3.candidate_id(row),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=Path("data/training/qwen3_4b_12line_auto_calibrated_v3"))
    parser.add_argument("--v4-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v4"))
    parser.add_argument("--judged", type=Path, default=Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge/judged_candidates.jsonl"))
    parser.add_argument("--raw-generations", type=Path, default=Path("data/sweeps/qwen3_4b_base_12line_v1_quality1200_retry2/sweep_raw.jsonl"))
    parser.add_argument("--generation-summary", type=Path, default=Path("data/sweeps/qwen3_4b_base_12line_v1_quality1200_retry2/sweep_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v41"))
    parser.add_argument("--min-calibrated-score", type=float, default=3.5)
    parser.add_argument("--preference-margin", type=float, default=0.75)
    parser.add_argument("--max-rejects-per-chosen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    started = time.perf_counter()

    v4_rows_by_split = {
        split: read_jsonl(args.v4_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    all_v4_rows = [row for rows in v4_rows_by_split.values() for row in rows]
    used_ids = {
        str(row.get("metadata", {}).get("candidate_id"))
        for row in all_v4_rows
        if row.get("metadata", {}).get("candidate_id")
    }

    judged = v3.read_jsonl(args.judged)
    raw_rows = v3.read_jsonl(args.raw_generations)
    raw_by_id = {v3.candidate_id(row): row for row in raw_rows}
    generation_summary = json.loads(args.generation_summary.read_text(encoding="utf-8"))
    split_by_theme = theme_split_map(args.v3_dir)
    anchors = choose_unused_strong_rows(judged, used_ids, args.min_calibrated_score)
    if Counter(normalized_family(row) for row in anchors) != {"melodic": 10, "story": 3}:
        raise RuntimeError("Expected the audited unused anchor pool to contain 10 melodic and 3 story rows")

    enriched: list[dict[str, Any]] = []
    anchor_split: dict[str, str] = {}
    for row in anchors:
        candidate = v3.candidate_id(row)
        raw = raw_by_id.get(candidate)
        if raw is None:
            raise RuntimeError(f"Missing raw generation provenance for {candidate}")
        theme = v3.normalize(row.get("theme"))
        split = split_by_theme.get(theme)
        if split is None:
            raise RuntimeError(f"No existing heldout split assignment for theme {theme!r}")
        copied = dict(row)
        copied["source_provenance"] = v3.source_provenance(
            row,
            raw,
            judged_path=args.judged,
            raw_path=args.raw_generations,
            summary_path=args.generation_summary,
            summary=generation_summary,
        )
        copied["selection_metadata"] = {
            "v41_role": "unused_strict_strong_family_anchor",
            "completion_anchor": True,
            "preserves_original_theme_split": True,
        }
        enriched.append(copied)
        anchor_split[candidate] = split

    records_by_split = {split: list(rows) for split, rows in v4_rows_by_split.items()}
    for row in enriched:
        split = anchor_split[v3.candidate_id(row)]
        record = v3.training_record(
            row,
            split,
            args.min_calibrated_score,
            dataset_name="qwen3_4b_12line_section_mined_v41",
            record_prefix="section-v41-anchor",
        )
        record["metadata"]["v41_role"] = "unused_strict_strong_family_anchor"
        records_by_split[split].append(record)

    for split, rows in records_by_split.items():
        rows.sort(key=lambda row: stable_int(str(args.seed), split, str(row["id"])))

    all_rows = [row for rows in records_by_split.values() for row in rows]
    hashes = [text_sha256(assistant_text(row["training_text"])) for row in all_rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate assistant text detected in v4.1")
    if any(len([line for line in assistant_text(row["training_text"]).splitlines() if line.strip()]) != 12 for row in all_rows):
        raise RuntimeError("Every v4.1 assistant target must contain exactly 12 non-empty lines")

    theme_sets = {
        split: {
            v3.normalize(row.get("metadata", {}).get("theme"))
            for row in rows
            if row.get("metadata", {}).get("theme")
        }
        for split, rows in records_by_split.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = theme_sets[left] & theme_sets[right]
        if overlap:
            raise RuntimeError(f"Theme leakage between {left} and {right}: {sorted(overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for split, rows in records_by_split.items():
        output_paths[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_paths[split], rows)

    existing_pairs = read_jsonl(args.v4_dir / "preference_pairs.jsonl")
    new_pairs = v3.preference_pairs(
        enriched,
        judged,
        anchor_split,
        margin=args.preference_margin,
        max_rejects=args.max_rejects_per_chosen,
        dataset_name="qwen3_4b_12line_section_mined_v41",
        record_prefix="section-v41-pref",
    )
    pairs = existing_pairs + new_pairs
    if len({pair["id"] for pair in pairs}) != len(pairs):
        raise RuntimeError("Duplicate preference-pair ID detected")
    output_paths["preferences"] = args.output_dir / "preference_pairs.jsonl"
    write_jsonl(output_paths["preferences"], pairs)

    family_by_split = {
        split: dict(sorted(Counter(row["metadata"]["prompt_family"] for row in rows).items()))
        for split, rows in records_by_split.items()
    }
    anchor_by_split = Counter(anchor_split.values())
    manifest = {
        "name": "qwen3_4b_12line_section_mined_v41",
        "status": "training_ready",
        "dataset_version": "v4.1",
        "selection_policy": "v4_plus_all_unused_strict_melodic_story_completion_anchors",
        "seed": args.seed,
        "wall_time_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join([sys.executable, *sys.argv]),
        "source_inputs": {
            "v4_manifest": {"path": str(args.v4_dir / "manifest.json"), "sha256": sha256_file(args.v4_dir / "manifest.json")},
            "v3_manifest": {"path": str(args.v3_dir / "manifest.json"), "sha256": sha256_file(args.v3_dir / "manifest.json")},
            "judged": {"path": str(args.judged), "sha256": sha256_file(args.judged)},
            "raw_generations": {"path": str(args.raw_generations), "sha256": sha256_file(args.raw_generations)},
            "generation_summary": {"path": str(args.generation_summary), "sha256": sha256_file(args.generation_summary)},
        },
        "counts": {
            "selected": len(all_rows),
            **{f"{split}_rows": len(rows) for split, rows in records_by_split.items()},
            "preference_pairs": len(pairs),
            "new_preference_pairs": len(new_pairs),
            "retained_v4_rows": len(all_v4_rows),
            "added_strong_family_anchors": len(enriched),
        },
        "anchor_family_counts": dict(sorted(Counter(normalized_family(row) for row in enriched).items())),
        "anchor_split_counts": dict(sorted(anchor_by_split.items())),
        "family_presence_by_split": family_by_split,
        "duplicate_assistant_rows": len(hashes) - len(set(hashes)),
        "exact_12_line_rows": sum(len([line for line in assistant_text(row["training_text"]).splitlines() if line.strip()]) == 12 for row in all_rows),
        "theme_split_overlap": {
            "train_validation": len(theme_sets["train"] & theme_sets["validation"]),
            "train_test": len(theme_sets["train"] & theme_sets["test"]),
            "validation_test": len(theme_sets["validation"] & theme_sets["test"]),
        },
        "comparability_note": "V4.1 changes only candidate quality/mix by adding 13 strict synthetic anchors; training hyperparameters remain v3/v4-matched.",
        "paths": {key: str(path) for key, path in output_paths.items()},
    }
    manifest["output_sha256"] = {key: sha256_file(path) for key, path in output_paths.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
