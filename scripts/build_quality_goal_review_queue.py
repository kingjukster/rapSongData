#!/usr/bin/env python3
"""DEPRECATED: build the legacy human-review queue for historical reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.build_manual_rank_app import app_source_sha256, build_html, candidate_payload, current_commit
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_manual_rank_app import app_source_sha256, build_html, candidate_payload, current_commit


DEFAULT_JUDGE_DIR = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge")
DEFAULT_CALIBRATED_DIR = DEFAULT_JUDGE_DIR / "calibrated_quality_sets"
DEFAULT_SEED_DIR = Path("data/training/qwen3_4b_base_12line_v1_quality_sft_seed_v1")
DEFAULT_OUTPUT_DIR = Path("data/curation/qwen3_4b_12line_human_v1")
DEFAULT_SWEEP_RAW = Path("data/sweeps/qwen3_4b_base_12line_v1_quality1200_retry2/sweep_raw.jsonl")
DEFAULT_SWEEP_SUMMARY = DEFAULT_SWEEP_RAW.with_name("sweep_summary.json")
DEFAULT_SOURCE_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DEFAULT_MODEL_REVISION_SOURCE = "retrospective_single_local_hf_snapshot_created_before_source_run"
TARGET_ISSUES = {"weak_imagery", "low_rhyme", "generic", "weak_payoff", "scene_drift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    parser.add_argument("--calibrated-dir", type=Path, default=DEFAULT_CALIBRATED_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--sweep-raw", type=Path, default=DEFAULT_SWEEP_RAW)
    parser.add_argument("--sweep-summary", type=Path, default=DEFAULT_SWEEP_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--legacy-count", type=int, default=60)
    parser.add_argument("--curated-unreviewed-count", type=int, default=70)
    parser.add_argument("--per-missing-prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--rubric-version", default="rap_12line_quality_v1")
    parser.add_argument("--session-id", default="qwen3-4b-12line-human-v1-review-001")
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_SOURCE_MODEL_REVISION,
        help="Retrospectively verified Hugging Face commit for the source generation run.",
    )
    parser.add_argument("--model-revision-source", default=DEFAULT_MODEL_REVISION_SOURCE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected array at {path}")
    return [row for row in value if isinstance(row, dict)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("candidate_id") or row.get("row_id") or meta.get("candidate_id") or "")


def candidate_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [value for row in rows if (value := candidate_id(row))]


def lyric_lines(row: dict[str, Any]) -> list[str]:
    return [line.strip() for line in str(row.get("lyrics") or row.get("generated_text") or "").splitlines() if line.strip()]


def structurally_eligible(row: dict[str, Any]) -> bool:
    structural = row.get("structural_metrics") if isinstance(row.get("structural_metrics"), dict) else {}
    return (
        len(lyric_lines(row)) == 12
        and int(structural.get("target_line_count") or 12) == 12
        and int(structural.get("slur_count") or 0) == 0
        and not bool(structural.get("prompt_leakage"))
        and not bool(structural.get("high_copy_similarity"))
    )


def quality_score(row: dict[str, Any]) -> float:
    for key in ("calibrated_review_score", "combined_quality_score", "heuristic_5", "quality_score"):
        try:
            if row.get(key) is not None:
                return float(row[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def prompt_key(row: dict[str, Any]) -> str:
    return str(row.get("prompt_key") or sha256_text(str(row.get("prompt") or ""))[:16])


def legacy_review_ids(judge_dir: Path) -> list[str]:
    paths = [
        judge_dir / "manual_rank_results.json",
        judge_dir / "manual_rank_auto_keep_results.json",
    ]
    return candidate_ids(row for path in paths for row in read_json_array(path))


def curated_candidate_ids(calibrated_dir: Path, seed_dir: Path) -> list[str]:
    rows: list[dict[str, Any]] = []
    for name in ("usable_candidates.jsonl", "edit_candidates.jsonl"):
        rows.extend(read_jsonl(calibrated_dir / name))
    for name in ("train.jsonl", "validation.jsonl"):
        rows.extend(read_jsonl(seed_dir / name))
    return candidate_ids(rows)


def select_queue(args: argparse.Namespace, judged_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in judged_rows if structurally_eligible(row)]
    by_id = {candidate_id(row): row for row in eligible}
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_prompt[prompt_key(row)].append(row)
    for rows in by_prompt.values():
        rows.sort(key=lambda row: (quality_score(row), candidate_id(row)), reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any], reason: str) -> bool:
        row_id = candidate_id(row)
        if not row_id or row_id in selected_ids:
            return False
        copied = dict(row)
        copied["queue_reason"] = reason
        selected.append(copied)
        selected_ids.add(row_id)
        return True

    for row_id in legacy_review_ids(args.judge_dir)[: args.legacy_count]:
        if row_id in by_id:
            add(by_id[row_id], "legacy_blind_re_review")

    curated = [by_id[row_id] for row_id in set(curated_candidate_ids(args.calibrated_dir, args.seed_dir)) if row_id in by_id]
    curated.sort(key=lambda row: (quality_score(row), candidate_id(row)), reverse=True)
    curated_added = 0
    for row in curated:
        if add(row, "unreviewed_curated_union"):
            curated_added += 1
            if curated_added >= args.curated_unreviewed_count:
                break

    covered_prompts = {prompt_key(row) for row in selected}
    for key in sorted(set(by_prompt) - covered_prompts):
        count = 0
        for row in by_prompt[key]:
            if add(row, "missing_prompt_coverage"):
                count += 1
                if count >= args.per_missing_prompt:
                    break

    selected_prompt_counts = Counter(prompt_key(row) for row in selected)
    contrast_pool = sorted(
        (row for row in eligible if candidate_id(row) not in selected_ids),
        key=lambda row: (
            0 if str((row.get("judge") or {}).get("main_issue")) in TARGET_ISSUES else 1,
            selected_prompt_counts[prompt_key(row)],
            quality_score(row),
            candidate_id(row),
        ),
    )
    for row in contrast_pool:
        if len(selected) >= args.target_count:
            break
        add(row, "same_prompt_contrast_or_balance")

    if len(selected) < args.target_count:
        raise ValueError(f"Only selected {len(selected)} eligible candidates; requested {args.target_count}")
    return selected[: args.target_count]


def attach_provenance(
    rows: list[dict[str, Any]],
    *,
    judged_path: Path,
    sweep_raw_path: Path,
    sweep_summary_path: Path,
    generation_config_path: Path,
    model_revision: str | None,
    model_revision_source: str | None,
) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(sweep_raw_path)
    raw_by_id = {str(row.get("row_id") or row.get("candidate_id")): row for row in raw_rows}
    created_at = datetime.fromtimestamp(
        (sweep_raw_path if sweep_raw_path.exists() else judged_path).stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
    generation_config_hash = sha256_file(generation_config_path)
    judged_path_hash = sha256_file(judged_path)
    sweep_raw_hash = sha256_file(sweep_raw_path)
    sweep_summary_hash = sha256_file(sweep_summary_path)
    sweep_summary = (
        json.loads(sweep_summary_path.read_text(encoding="utf-8"))
        if sweep_summary_path.exists()
        else {}
    )
    for row in rows:
        lyrics = "\n".join(lyric_lines(row))
        raw = raw_by_id.get(candidate_id(row))
        raw_lyrics = str((raw or {}).get("generated_text") or "")
        if raw and " ".join(raw_lyrics.lower().split()) != " ".join(lyrics.lower().split()):
            raise ValueError(f"Judged and raw generation text disagree for {candidate_id(row)}")
        settings = (raw or {}).get("settings") if isinstance((raw or {}).get("settings"), dict) else {}
        row["provenance"] = {
            "candidate_id": candidate_id(row),
            "normalized_text_sha256": sha256_text(" ".join(lyrics.lower().split())),
            "prompt_key": prompt_key(row),
            "judge_source_path": str(judged_path),
            "judge_source_sha256": judged_path_hash,
            "generation_source_path": str(sweep_raw_path) if raw else None,
            "generation_source_sha256": sweep_raw_hash if raw else None,
            "generation_summary_path": str(sweep_summary_path) if sweep_summary_path.exists() else None,
            "generation_summary_sha256": sweep_summary_hash,
            "generation_run_id": "qwen3_4b_base_12line_v1_quality1200_retry2",
            "judge_run_id": "qwen3_4b_base_12line_v1_auto_quality_judge",
            "model_id": settings.get("base_model") or "Qwen/Qwen3-4B",
            "model_revision": settings.get("model_revision") or model_revision,
            "model_revision_source": (
                "source_generation_record" if settings.get("model_revision") else model_revision_source
            ),
            "adapter": settings.get("adapter_dir"),
            "decoding_reference_config_sha256": generation_config_hash,
            "generation_settings": sweep_summary.get("settings") or settings,
            "rng_provenance": generation_rng_provenance(raw, sweep_summary),
            "sample_index": row.get("sample_index"),
            "candidate_index": row.get("candidate_index"),
            "postprocess_actions": (raw or {}).get("postprocess_actions") or [],
            "created_at": created_at,
            "license_scope": "synthetic_model_generated_local_audit",
        }
    return rows


def generation_rng_provenance(
    raw: dict[str, Any] | None,
    sweep_summary: dict[str, Any],
) -> dict[str, Any] | None:
    if not raw:
        return None
    attempts = int(raw.get("generation_attempt_count") or 1)
    accepted_attempt = int(raw.get("accepted_attempt_index") or 1)
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    settings = sweep_summary.get("settings") if isinstance(sweep_summary.get("settings"), dict) else {}
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

    candidate_index = int(raw.get("candidate_index") or 0)
    batch_size = int(timing.get("batch_size") or 0)
    base_seed = settings.get("seed")
    if candidate_index <= 0 or batch_size <= 0 or not isinstance(base_seed, int):
        return {
            "protocol": "legacy_batched_shared_rng_unresolved",
            "manual_seed": None,
            "recorded_row_seed": raw.get("seed"),
            "exact_per_row_seed": False,
        }
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


def main() -> int:
    print(
        "DEPRECATED: use build_auto_calibrated_12line_sft.py; human review is retired.",
        file=sys.stderr,
    )
    args = parse_args()
    if not 100 <= args.target_count <= 300:
        raise ValueError("--target-count must be between 100 and 300")
    judged_path = args.judge_dir / "judged_candidates.jsonl"
    judged_rows = read_jsonl(judged_path)
    selected = select_queue(args, judged_rows)
    attach_provenance(
        selected,
        judged_path=judged_path,
        sweep_raw_path=args.sweep_raw,
        sweep_summary_path=args.sweep_summary,
        generation_config_path=Path("configs/evaluation/qwen3_4b_base_12line_v1.json"),
        model_revision=args.model_revision,
        model_revision_source=args.model_revision_source,
    )
    random.Random(args.seed).shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["queue_index"] = index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "review_queue.jsonl"
    manifest_path = args.output_dir / "review_queue_manifest.json"
    app_path = args.output_dir / "manual_review.html"
    write_jsonl(queue_path, selected)

    app_candidates = [candidate_payload(row, rank=index) for index, row in enumerate(selected, start=1)]
    app_path.write_text(
        build_html(
            app_candidates,
            review_set="quality_goal",
            export_prefix="qwen3_4b_12line_human_v1_reviews",
            blind=True,
            rubric_version=args.rubric_version,
            review_session_id=args.session_id,
            app_version="manual_rank_v2",
            app_commit_sha=current_commit(),
            app_source_hash=app_source_sha256(),
        ),
        encoding="utf-8",
    )

    manifest = {
        "name": "qwen3_4b_12line_human_v1_review_queue",
        "status": "awaiting_human_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queue_path": str(queue_path),
        "app_path": str(app_path),
        "target_count": args.target_count,
        "actual_count": len(selected),
        "randomization_seed": args.seed,
        "rubric_version": args.rubric_version,
        "session_id": args.session_id,
        "retrospective_model_revision": args.model_revision,
        "model_revision_source": args.model_revision_source,
        "selection_reason_counts": dict(Counter(str(row.get("queue_reason")) for row in selected)),
        "prompt_key_count": len({prompt_key(row) for row in selected}),
        "theme_count": len({str(row.get("theme") or "") for row in selected}),
        "prompt_family_counts": dict(Counter(str(row.get("prompt_family") or "unknown") for row in selected)),
        "normalized_text_sha256_count": len(
            {str((row.get("provenance") or {}).get("normalized_text_sha256")) for row in selected}
        ),
        "queue_sha256": sha256_file(queue_path),
        "provenance_gap_counts": {
            field: sum(not (row.get("provenance") or {}).get(field) for row in selected)
            for field in ("generation_source_sha256", "generation_summary_sha256", "model_revision", "rng_provenance")
        },
        "rng_protocol_counts": dict(
            Counter(str(((row.get("provenance") or {}).get("rng_provenance") or {}).get("protocol")) for row in selected)
        ),
        "exact_per_row_seed_count": sum(
            bool(((row.get("provenance") or {}).get("rng_provenance") or {}).get("exact_per_row_seed"))
            for row in selected
        ),
        "eligibility": "No row is training-eligible until exported with human_attested=true and the full rubric.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
