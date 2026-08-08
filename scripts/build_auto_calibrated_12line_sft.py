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
DEFAULT_V3_OUTPUT = Path("data/training/qwen3_4b_12line_auto_calibrated_v3")
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_REVISION_SOURCE = "retrospective_single_local_hf_snapshot_created_before_source_run"
CRITERIA_VERSION = "automated_consensus_v2"
BALANCED_FAMILIES = ("melodic", "story", "technical", "clean")
DEFAULT_V3_FAMILY_QUOTA = 33
QUALITY_TIER_1 = "tier_1_imagery_and_payoff"
QUALITY_TIER_2 = "tier_2_imagery_or_payoff"
QUALITY_TIER_3 = "tier_3_consensus_eligible"
QUALITY_TIERS = (QUALITY_TIER_1, QUALITY_TIER_2, QUALITY_TIER_3)
REQUIRED_DIMENSIONS = {
    "theme_adherence": 4,
    "imagery": 3,
    "rhyme_cadence": 3,
    "originality": 3,
    "scene_coherence": 4,
    "ending_payoff": 3,
    "naturalness": 4,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    parser.add_argument("--raw-generations", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--dataset-version",
        choices=("v2", "v3"),
        default="v2",
        help="v2 preserves legacy score selection; v3 enables balanced tiered family selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the versioned qwen3_4b_12line_auto_calibrated_v2/v3 directory.",
    )
    parser.add_argument("--min-calibrated-score", type=float, default=3.5)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--preference-margin", type=float, default=0.75)
    parser.add_argument("--max-rejects-per-chosen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
    for family in BALANCED_FAMILIES:
        parser.add_argument(
            f"--{family}-quota",
            type=int,
            default=DEFAULT_V3_FAMILY_QUOTA,
            help=f"Number of {family} examples selected by --dataset-version v3.",
        )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_V3_OUTPUT if args.dataset_version == "v3" else DEFAULT_OUTPUT
    return args


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


def dimension_scores(row: dict[str, Any]) -> dict[str, Any]:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    dimensions = judge.get("dimension_scores") if isinstance(judge.get("dimension_scores"), dict) else {}
    return dimensions


def quality_tier(row: dict[str, Any]) -> str:
    """Return the v3 quality tier used for deterministic family selection."""
    dimensions = dimension_scores(row)
    imagery = int(dimensions.get("imagery") or 0)
    payoff = int(dimensions.get("ending_payoff") or 0)
    if imagery >= 4 and payoff >= 4:
        return QUALITY_TIER_1
    supporting_dimensions_pass = all(
        int(dimensions.get(name) or 0) >= REQUIRED_DIMENSIONS[name]
        for name in ("originality", "rhyme_cadence", "naturalness")
    )
    if (imagery >= 4 or payoff >= 4) and supporting_dimensions_pass:
        return QUALITY_TIER_2
    return QUALITY_TIER_3


def worst_dimension_score(row: dict[str, Any]) -> int:
    dimensions = dimension_scores(row)
    return min(int(dimensions.get(name) or 0) for name in REQUIRED_DIMENSIONS)


def tiered_quality_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    """Sort high-quality rows first with a stable candidate-id tie break."""
    return (
        QUALITY_TIERS.index(quality_tier(row)),
        -worst_dimension_score(row),
        -score(row),
        candidate_id(row),
    )


def select_balanced_examples(
    rows: list[dict[str, Any]],
    family_quotas: dict[str, int],
) -> list[dict[str, Any]]:
    """Select an exact, deterministic quota from each requested prompt family."""
    selected: list[dict[str, Any]] = []
    for family in BALANCED_FAMILIES:
        if family not in family_quotas:
            continue
        quota = family_quotas[family]
        candidates = sorted(
            (row for row in rows if normalize(row.get("prompt_family")) == family),
            key=tiered_quality_key,
        )
        if len(candidates) < quota:
            raise SystemExit(
                f"Only {len(candidates)} {family} examples passed; v3 quota is {quota}"
            )
        for rank, row in enumerate(candidates[:quota], start=1):
            copied = dict(row)
            copied["selection_metadata"] = {
                "quality_tier": quality_tier(row),
                "worst_dimension_score": worst_dimension_score(row),
                "family_rank": rank,
                "family_quota": quota,
            }
            selected.append(copied)
    ensure_unique_lyrics(selected)
    return selected


def ensure_unique_lyrics(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        text_hash = normalized_sha256(row.get("lyrics"))
        duplicate_of = seen.get(text_hash)
        if duplicate_of is not None:
            raise ValueError(
                "Duplicate normalized lyrics selected: "
                f"{candidate_id(row)} duplicates {duplicate_of} ({text_hash})"
            )
        seen[text_hash] = candidate_id(row)


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


def theme_group_key(row: dict[str, Any]) -> str:
    theme = normalize(row.get("theme"))
    return f"theme:{theme}" if theme else f"prompt:{row.get('prompt_key')}"


def split_theme_groups(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Legacy v2 split behavior, intentionally unchanged."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[theme_group_key(row)].append(row)
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


def _split_family_counts(
    values: Iterable[dict[str, Any]], families: tuple[str, ...]
) -> Counter[str]:
    expected = set(families)
    return Counter(
        family
        for row in values
        if (family := normalize(row.get("prompt_family"))) in expected
    )


def split_balanced_theme_groups(
    rows: list[dict[str, Any]],
    seed: int,
    families: tuple[str, ...] = BALANCED_FAMILIES,
) -> dict[str, list[dict[str, Any]]]:
    """Keep themes isolated while optimizing 80/10/10 per-family balance."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[theme_group_key(row)].append(row)
    if len(groups) < 3:
        raise ValueError("At least three theme groups are required")

    keys = sorted(groups, key=lambda key: (stable_int(str(seed), key), key))
    group_counts = [_split_family_counts(groups[key], families) for key in keys]
    total_counts = _split_family_counts(rows, families)
    missing = [family for family in families if not total_counts[family]]
    if missing:
        raise ValueError(f"Balanced split is missing required families: {', '.join(missing)}")
    thin = [
        family
        for family in families
        if sum(bool(counts[family]) for counts in group_counts) < 3
    ]
    if thin:
        raise ValueError(
            "Each family must occur in at least three theme groups for isolated splits: "
            + ", ".join(thin)
        )

    family_targets = {family: total_counts[family] * 0.10 for family in families}
    row_target = len(rows) * 0.10

    def subset_details(mask: int) -> tuple[Counter[str], int, int]:
        counts: Counter[str] = Counter()
        row_count = 0
        group_count = 0
        for index, values in enumerate(group_counts):
            if mask & (1 << index):
                counts.update(values)
                row_count += len(groups[keys[index]])
                group_count += 1
        return counts, row_count, group_count

    def subset_cost(counts: Counter[str], row_count: int, group_count: int) -> float:
        family_cost = sum(
            ((counts[family] - family_targets[family]) / max(1.0, family_targets[family])) ** 2
            for family in families
        )
        row_cost = ((row_count - row_target) / max(1.0, row_target)) ** 2
        return family_cost + row_cost + group_count * 1e-6

    candidate_masks: set[int] = set()
    group_total = len(keys)
    if group_total <= 18:
        for mask in range(1, (1 << group_total) - 1):
            counts, _, _ = subset_details(mask)
            if all(counts[family] for family in families):
                candidate_masks.add(mask)
    else:
        # Large prompt-only corpora use deterministic greedy starts to avoid an
        # exponential search. The real v3 source has only a dozen theme groups.
        for start in range(group_total):
            mask = 1 << start
            while True:
                counts, row_count, group_count = subset_details(mask)
                if all(counts[family] for family in families) and row_count >= row_target * 0.75:
                    candidate_masks.add(mask)
                    break
                additions: list[tuple[float, int, int]] = []
                for index in range(group_total):
                    bit = 1 << index
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    next_counts, next_rows, next_groups = subset_details(next_mask)
                    missing_count = sum(not next_counts[family] for family in families)
                    additions.append(
                        (
                            missing_count * 1_000_000.0
                            + subset_cost(next_counts, next_rows, next_groups),
                            stable_int(str(seed), str(start), keys[index]),
                            next_mask,
                        )
                    )
                if not additions:
                    break
                mask = min(additions)[2]

    candidates: list[tuple[float, int, int, Counter[str], int]] = []
    for mask in candidate_masks:
        counts, row_count, group_count = subset_details(mask)
        selected_keys = "\n".join(keys[index] for index in range(group_total) if mask & (1 << index))
        candidates.append(
            (
                subset_cost(counts, row_count, group_count),
                stable_int(str(seed), selected_keys),
                mask,
                counts,
                row_count,
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    candidates = candidates[:1024]

    best: tuple[float, int, int, int] | None = None
    all_mask = (1 << group_total) - 1
    for validation in candidates:
        for test in candidates:
            validation_mask = validation[2]
            test_mask = test[2]
            if validation_mask & test_mask:
                continue
            train_mask = all_mask ^ validation_mask ^ test_mask
            if not train_mask:
                continue
            train_counts, _, _ = subset_details(train_mask)
            if not all(train_counts[family] for family in families):
                continue
            symmetry_cost = sum(
                (
                    (validation[3][family] - test[3][family])
                    / max(1.0, family_targets[family])
                )
                ** 2
                for family in families
            )
            cost = validation[0] + test[0] + symmetry_cost * 0.25
            tie = stable_int(str(seed), str(validation_mask), str(test_mask))
            result = (cost, tie, validation_mask, test_mask)
            if best is None or result < best:
                best = result
    if best is None:
        raise ValueError("Could not create disjoint theme-isolated splits containing every family")

    assignments: dict[str, str] = {}
    validation_mask, test_mask = best[2], best[3]
    for index, key in enumerate(keys):
        bit = 1 << index
        assignments[key] = "validation" if validation_mask & bit else "test" if test_mask & bit else "train"
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for key, group_rows in groups.items():
        splits[assignments[key]].extend(group_rows)
    for split, values in splits.items():
        values.sort(key=lambda row: stable_int(str(seed), split, candidate_id(row)))
        split_counts = _split_family_counts(values, families)
        absent = [family for family in families if not split_counts[family]]
        if absent:
            raise ValueError(f"{split} split is missing required families: {', '.join(absent)}")
    return splits


def chatml(prompt: str, lyrics: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{lyrics.strip()}<|im_end|>\n"
    )


def training_record(
    row: dict[str, Any],
    split: str,
    min_score: float,
    *,
    dataset_name: str = "qwen3_4b_12line_auto_calibrated_v2",
    record_prefix: str = "auto-v2",
) -> dict[str, Any]:
    metadata = {
        "source": dataset_name,
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
    }
    if isinstance(row.get("selection_metadata"), dict):
        metadata["selection"] = row["selection_metadata"]
    return {
        "id": f"{record_prefix}-{candidate_id(row)}",
        "training_text": chatml(str(row["prompt"]), str(row["lyrics"])),
        "metadata": metadata,
    }


def preference_pairs(
    selected: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    split_by_candidate: dict[str, str],
    *,
    margin: float,
    max_rejects: int,
    dataset_name: str = "qwen3_4b_12line_auto_calibrated_v2",
    record_prefix: str = "auto-pref-v2",
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
                    "id": f"{record_prefix}-{key[0]}-{key[1]}",
                    "prompt": chosen.get("prompt"),
                    "chosen": chosen.get("lyrics"),
                    "rejected": row.get("lyrics"),
                    "metadata": {
                        "source": dataset_name,
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


def _sorted_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions: dict[str, dict[str, int]] = {}
    for name in REQUIRED_DIMENSIONS:
        dimensions[name] = _sorted_counter(
            str(int(dimension_scores(row).get(name) or 0)) for row in rows
        )

    issue_tags: list[str] = []
    for row in rows:
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        tags = judge.get("issue_tags") if isinstance(judge.get("issue_tags"), list) else []
        if tags:
            issue_tags.extend(str(tag) for tag in tags)
        else:
            issue_tags.append(str(judge.get("main_issue") or "none"))
    return {
        "rows": len(rows),
        "family": _sorted_counter(normalize(row.get("prompt_family")) or "unknown" for row in rows),
        "theme": _sorted_counter(normalize(row.get("theme")) or "unknown" for row in rows),
        "quality_tier": _sorted_counter(quality_tier(row) for row in rows),
        "worst_dimension_score": _sorted_counter(str(worst_dimension_score(row)) for row in rows),
        "dimensions": dimensions,
        "issue_tag": _sorted_counter(issue_tags),
    }


def main() -> int:
    args = parse_args()
    if not 100 <= args.min_examples <= args.max_examples <= 300:
        raise ValueError("Require 100 <= min-examples <= max-examples <= 300")
    family_quotas = {
        family: int(getattr(args, f"{family}_quota"))
        for family in BALANCED_FAMILIES
    }
    if args.dataset_version == "v3":
        if any(quota <= 0 for quota in family_quotas.values()):
            raise ValueError("All v3 family quotas must be positive")
        selected_count = sum(family_quotas.values())
        if not args.min_examples <= selected_count <= args.max_examples:
            raise ValueError(
                "The sum of v3 family quotas must fall between --min-examples and --max-examples"
            )
    judged = read_jsonl(args.judged)
    raw_rows = read_jsonl(args.raw_generations)
    raw_by_id = {candidate_id(row): row for row in raw_rows}
    summary = json.loads(args.generation_summary.read_text(encoding="utf-8"))
    eligible = [row for row in judged if consensus_eligible(row, args.min_calibrated_score)]
    duplicate_rows_removed = 0
    if args.dataset_version == "v3":
        selected = select_balanced_examples(eligible, family_quotas)
    else:
        eligible.sort(key=lambda row: (score(row), candidate_id(row)), reverse=True)
        selected = []
        seen_texts: set[str] = set()
        for row in eligible:
            text_hash = normalized_sha256(row.get("lyrics"))
            if text_hash in seen_texts:
                duplicate_rows_removed += 1
                continue
            selected.append(row)
            seen_texts.add(text_hash)
            if len(selected) >= args.max_examples:
                break
        if len(selected) < args.min_examples:
            raise SystemExit(
                f"Only {len(selected)} automated-consensus examples passed; minimum is {args.min_examples}"
            )

    unique: list[dict[str, Any]] = []
    for row in selected:
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
    ensure_unique_lyrics(unique)

    splits = (
        split_balanced_theme_groups(unique, args.seed)
        if args.dataset_version == "v3"
        else split_theme_groups(unique, args.seed)
    )
    dataset_name = f"qwen3_4b_12line_auto_calibrated_{args.dataset_version}"
    record_prefix = f"auto-{args.dataset_version}"
    split_by_candidate = {
        candidate_id(row): split for split, rows in splits.items() for row in rows
    }
    records = {
        split: [
            training_record(
                row,
                split,
                args.min_calibrated_score,
                dataset_name=dataset_name,
                record_prefix=record_prefix,
            )
            for row in rows
        ]
        for split, rows in splits.items()
    }
    pairs = preference_pairs(
        unique,
        judged,
        split_by_candidate,
        margin=args.preference_margin,
        max_rejects=args.max_rejects_per_chosen,
        dataset_name=dataset_name,
        record_prefix=f"auto-pref-{args.dataset_version}",
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
    split_families = {
        split: _split_family_counts(rows, BALANCED_FAMILIES)
        for split, rows in splits.items()
    }
    manifest = {
        "name": dataset_name,
        "status": "training_ready",
        "label_policy": "automated_consensus_no_human_review",
        "human_review_deprecated": True,
        "criteria_version": CRITERIA_VERSION,
        "dataset_version": args.dataset_version,
        "selection_policy": (
            "balanced_family_quality_tiers" if args.dataset_version == "v3" else "legacy_calibrated_score"
        ),
        "seed": args.seed,
        "thresholds": {
            "min_calibrated_score": args.min_calibrated_score,
            "required_dimension_minimums": REQUIRED_DIMENSIONS,
            "preference_margin": args.preference_margin,
            "max_rejects_per_chosen": args.max_rejects_per_chosen,
            "family_quotas": family_quotas if args.dataset_version == "v3" else None,
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
            "duplicate_rows_removed": duplicate_rows_removed,
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
        "family_presence_by_split": {
            split: {
                family: int(split_families[split][family])
                for family in BALANCED_FAMILIES
            }
            for split in ("train", "validation", "test")
        },
        "distributions": {
            "selected": distribution_summary(unique),
            "split": {
                split: distribution_summary(rows)
                for split, rows in splits.items()
            },
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
