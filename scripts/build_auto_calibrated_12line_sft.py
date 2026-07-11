#!/usr/bin/env python3
"""Build strict exact-12-line SFT and preference data from automated consensus scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
DEFAULT_JUDGED = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge/judged_candidates.jsonl")
DEFAULT_RAW = Path("data/sweeps/qwen3_4b_base_12line_v1_quality1200_retry2/sweep_raw.jsonl")
DEFAULT_SUMMARY = DEFAULT_RAW.with_name("sweep_summary.json")
DEFAULT_OUTPUT = Path("data/training/qwen3_4b_12line_auto_calibrated_v2")
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_REVISION_SOURCE = "retrospective_single_local_hf_snapshot_created_before_source_run"
CRITERIA_VERSION = "automated_consensus_v2"
REQUIRED_DIMENSIONS = {
    "theme_adherence": 4,
    "imagery": 3,
    "rhyme_cadence": 3,
    "originality": 3,
    "scene_coherence": 4,
    "ending_payoff": 3,
    "naturalness": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("--raw-generations", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-calibrated-score", type=float, default=3.5)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--preference-margin", type=float, default=0.75)
    parser.add_argument("--max-rejects-per-chosen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def normalized_sha256(value: Any) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def stable_int(*parts: str) -> int:
    return int(hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def lyric_lines(value: Any) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("calibrated_review_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("row_id") or "")


def generation_rng_provenance(raw: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    attempts = int(raw.get("generation_attempt_count") or 1)
    accepted_attempt = int(raw.get("accepted_attempt_index") or 1)
    if attempts > 1 and accepted_attempt > 1:
        return {
            "protocol": "legacy_single_row_underlength_retry",
            "manual_seed": int(raw["seed"]),
            "recorded_initial_seed": raw.get("initial_seed"),
            "accepted_attempt_index": accepted_attempt,
            "batch_size": 1,
            "row_offset_within_batch": 0,
            "exact_per_row_seed": True,
        }
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
    candidate_index = int(raw.get("candidate_index") or 0)
    batch_size = int(timing.get("batch_size") or 0)
    base_seed = settings.get("seed")
    if candidate_index <= 0 or batch_size <= 0 or not isinstance(base_seed, int):
        raise ValueError(f"Could not reconstruct RNG provenance for {candidate_id(raw)}")
    batch_start = candidate_index - ((candidate_index - 1) % batch_size)
    return {
        "protocol": "legacy_batched_shared_rng",
        "manual_seed": base_seed + batch_start - 1,
        "recorded_row_seed_not_used_as_manual_seed": raw.get("seed"),
        "batch_size": batch_size,
        "batch_start_candidate_index": batch_start,
        "row_offset_within_batch": candidate_index - batch_start,
        "exact_per_row_seed": False,
    }


def consensus_eligible(row: dict[str, Any], min_score: float) -> bool:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    dimensions = judge.get("dimension_scores") if isinstance(judge.get("dimension_scores"), dict) else {}
    structural = row.get("structural_metrics") if isinstance(row.get("structural_metrics"), dict) else {}
    return bool(
        judge.get("overall_quality") == 4
        and judge.get("usable_as_is") == "yes"
        and score(row) >= min_score
        and structural.get("structural_pass") is True
        and len(lyric_lines(row.get("lyrics"))) == 12
        and all(int(dimensions.get(name) or 0) >= minimum for name, minimum in REQUIRED_DIMENSIONS.items())
    )


def source_provenance(
    row: dict[str, Any],
    raw: dict[str, Any],
    *,
    judged_path: Path,
    raw_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    lyrics = str(row.get("lyrics") or "")
    if candidate_id(row) != candidate_id(raw):
        raise ValueError("Candidate-id mismatch between judged and raw source")
    if str(row.get("prompt_key")) != str(raw.get("prompt_key")) or normalize(row.get("prompt")) != normalize(raw.get("prompt")):
        raise ValueError(f"Prompt provenance mismatch for {candidate_id(row)}")
    if normalize(lyrics) != normalize(raw.get("generated_text")):
        raise ValueError(f"Generated-text provenance mismatch for {candidate_id(row)}")
    settings = raw.get("settings") if isinstance(raw.get("settings"), dict) else {}
    return {
        "candidate_id": candidate_id(row),
        "normalized_text_sha256": normalized_sha256(lyrics),
        "prompt_key": row.get("prompt_key"),
        "judge_source_path": str(judged_path),
        "judge_source_sha256": sha256_file(judged_path),
        "generation_source_path": str(raw_path),
        "generation_source_sha256": sha256_file(raw_path),
        "generation_summary_path": str(summary_path),
        "generation_summary_sha256": sha256_file(summary_path),
        "generation_run_id": "qwen3_4b_base_12line_v1_quality1200_retry2",
        "judge_run_id": "qwen3_4b_base_12line_v1_auto_quality_judge",
        "model_id": settings.get("base_model") or "Qwen/Qwen3-4B",
        "model_revision": settings.get("model_revision") or MODEL_REVISION,
        "model_revision_source": "source_generation_record" if settings.get("model_revision") else MODEL_REVISION_SOURCE,
        "adapter": settings.get("adapter_dir"),
        "generation_settings": summary.get("settings") or settings,
        "rng_provenance": generation_rng_provenance(raw, summary),
        "created_at": datetime.fromtimestamp(raw_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "license_scope": "synthetic_model_generated_local_audit",
    }


def automated_calibration(row: dict[str, Any], min_score: float) -> dict[str, Any]:
    judge = row["judge"]
    return {
        "label_source": "automated_consensus",
        "human_reviewed": False,
        "criteria_version": CRITERIA_VERSION,
        "judge_provider": "openai_api",
        "judge_model": "historical_run_model_recorded_in_raw_judge_artifact",
        "judge_overall_quality": judge.get("overall_quality"),
        "judge_usable_as_is": judge.get("usable_as_is"),
        "judge_main_issue": judge.get("main_issue"),
        "judge_dimension_scores": judge.get("dimension_scores"),
        "heuristic_quality_score": row.get("quality_score"),
        "combined_quality_score": row.get("combined_quality_score"),
        "calibrated_review_score": row.get("calibrated_review_score"),
        "minimum_calibrated_score": min_score,
        "required_dimension_minimums": REQUIRED_DIMENSIONS,
        "structural_pass_required": True,
    }


def split_theme_groups(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        theme = normalize(row.get("theme"))
        key = f"theme:{theme}" if theme else f"prompt:{row.get('prompt_key')}"
        groups[key].append(row)
    if len(groups) < 3:
        raise ValueError("At least three theme groups are required")
    ordered = sorted(groups, key=lambda key: stable_int(str(seed), key))
    validation_count = max(1, round(len(ordered) * 0.10))
    test_count = max(1, round(len(ordered) * 0.10))
    train_count = len(ordered) - validation_count - test_count
    assignments = {
        key: "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        for index, key in enumerate(ordered)
    }
    splits = {"train": [], "validation": [], "test": []}
    for key, group_rows in groups.items():
        splits[assignments[key]].extend(group_rows)
    for split in splits:
        splits[split].sort(key=lambda row: stable_int(str(seed), split, candidate_id(row)))
    return splits


def chatml(prompt: str, lyrics: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{lyrics.strip()}<|im_end|>\n"
    )


def training_record(row: dict[str, Any], split: str, min_score: float) -> dict[str, Any]:
    return {
        "id": f"auto-v2-{candidate_id(row)}",
        "training_text": chatml(str(row["prompt"]), str(row["lyrics"])),
        "metadata": {
            "source": "qwen3_4b_12line_auto_calibrated_v2",
            "source_bucket": "automated_consensus_keep",
            "candidate_id": candidate_id(row),
            "prompt_key": row.get("prompt_key"),
            "theme": row.get("theme"),
            "prompt_family": row.get("prompt_family"),
            "split": split,
            "target_line_count": 12,
            "actual_line_count": 12,
            "license_scope": row["source_provenance"]["license_scope"],
            "normalized_text_sha256": row["source_provenance"]["normalized_text_sha256"],
            "source_provenance": row["source_provenance"],
            "automated_calibration": automated_calibration(row, min_score),
            "format": "qwen_chatml_training_text",
        },
    }


def preference_pairs(
    selected: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    split_by_candidate: dict[str, str],
    *,
    margin: float,
    max_rejects: int,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_prompt[str(row.get("prompt_key"))].append(row)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chosen in selected:
        rejected = sorted(
            (
                row
                for row in by_prompt[str(chosen.get("prompt_key"))]
                if candidate_id(row) != candidate_id(chosen)
                and score(chosen) - score(row) >= margin
                and len(lyric_lines(row.get("lyrics"))) == 12
                and not int((row.get("structural_metrics") or {}).get("slur_count") or 0)
                and not bool((row.get("structural_metrics") or {}).get("prompt_leakage"))
                and normalize(row.get("lyrics")) != normalize(chosen.get("lyrics"))
            ),
            key=lambda row: (score(row), candidate_id(row)),
        )[:max_rejects]
        for row in rejected:
            key = (candidate_id(chosen), candidate_id(row))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "id": f"auto-pref-v2-{key[0]}-{key[1]}",
                    "prompt": chosen.get("prompt"),
                    "chosen": chosen.get("lyrics"),
                    "rejected": row.get("lyrics"),
                    "metadata": {
                        "source": "qwen3_4b_12line_auto_calibrated_v2",
                        "label_source": "automated_consensus_score_margin",
                        "criteria_version": CRITERIA_VERSION,
                        "prompt_key": chosen.get("prompt_key"),
                        "chosen_candidate_id": key[0],
                        "rejected_candidate_id": key[1],
                        "chosen_score": score(chosen),
                        "rejected_score": score(row),
                        "score_delta": round(score(chosen) - score(row), 4),
                        "minimum_score_delta": margin,
                        "split": split_by_candidate[key[0]],
                    },
                }
            )
    return pairs


def main() -> int:
    args = parse_args()
    if not 100 <= args.min_examples <= args.max_examples <= 300:
        raise ValueError("Require 100 <= min-examples <= max-examples <= 300")
    judged = read_jsonl(args.judged)
    raw_rows = read_jsonl(args.raw_generations)
    raw_by_id = {candidate_id(row): row for row in raw_rows}
    summary = json.loads(args.generation_summary.read_text(encoding="utf-8"))
    eligible = [row for row in judged if consensus_eligible(row, args.min_calibrated_score)]
    eligible.sort(key=lambda row: (score(row), candidate_id(row)), reverse=True)
    unique: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for row in eligible:
        text_hash = normalized_sha256(row.get("lyrics"))
        if text_hash in seen_texts:
            continue
        raw = raw_by_id.get(candidate_id(row))
        if raw is None:
            raise ValueError(f"Missing raw provenance for {candidate_id(row)}")
        copied = dict(row)
        copied["source_provenance"] = source_provenance(
            row,
            raw,
            judged_path=args.judged,
            raw_path=args.raw_generations,
            summary_path=args.generation_summary,
            summary=summary,
        )
        unique.append(copied)
        seen_texts.add(text_hash)
        if len(unique) >= args.max_examples:
            break
    if len(unique) < args.min_examples:
        raise SystemExit(f"Only {len(unique)} automated-consensus examples passed; minimum is {args.min_examples}")

    splits = split_theme_groups(unique, args.seed)
    split_by_candidate = {
        candidate_id(row): split for split, rows in splits.items() for row in rows
    }
    records = {
        split: [training_record(row, split, args.min_calibrated_score) for row in rows]
        for split, rows in splits.items()
    }
    pairs = preference_pairs(
        unique,
        judged,
        split_by_candidate,
        margin=args.preference_margin,
        max_rejects=args.max_rejects_per_chosen,
    )
    if not pairs:
        raise SystemExit("No preference pairs passed the automated score-margin gate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        paths[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(paths[split], records[split])
    paths["preferences"] = args.output_dir / "preference_pairs.jsonl"
    write_jsonl(paths["preferences"], pairs)
    split_themes = {
        split: {normalize(row.get("theme")) for row in rows if row.get("theme")}
        for split, rows in splits.items()
    }
    manifest = {
        "name": "qwen3_4b_12line_auto_calibrated_v2",
        "status": "training_ready",
        "label_policy": "automated_consensus_no_human_review",
        "human_review_deprecated": True,
        "criteria_version": CRITERIA_VERSION,
        "seed": args.seed,
        "thresholds": {
            "min_calibrated_score": args.min_calibrated_score,
            "required_dimension_minimums": REQUIRED_DIMENSIONS,
            "preference_margin": args.preference_margin,
            "max_rejects_per_chosen": args.max_rejects_per_chosen,
        },
        "source_inputs": {
            "judged": {"path": str(args.judged), "sha256": sha256_file(args.judged)},
            "raw_generations": {"path": str(args.raw_generations), "sha256": sha256_file(args.raw_generations)},
            "generation_summary": {"path": str(args.generation_summary), "sha256": sha256_file(args.generation_summary)},
        },
        "counts": {
            "judged_rows": len(judged),
            "eligible_before_dedup": len(eligible),
            "automated_consensus_unique": len(unique),
            "train_rows": len(records["train"]),
            "validation_rows": len(records["validation"]),
            "test_rows": len(records["test"]),
            "preference_pairs": len(pairs),
            "theme_count": len({normalize(row.get("theme")) for row in unique}),
            "prompt_key_count": len({str(row.get("prompt_key")) for row in unique}),
        },
        "theme_split_overlap": {
            "train_validation": len(split_themes["train"] & split_themes["validation"]),
            "train_test": len(split_themes["train"] & split_themes["test"]),
            "validation_test": len(split_themes["validation"] & split_themes["test"]),
        },
        "judge_issue_counts": dict(Counter(str((row.get("judge") or {}).get("main_issue")) for row in unique)),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest["output_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
