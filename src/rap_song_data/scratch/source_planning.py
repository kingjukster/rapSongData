from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .common import command_record, read_json, utc_now, write_json


VALID_TIERS = {"A", "B", "C", "D"}
VALID_ELIGIBILITY = {"eligible", "private_only", "conditional", "metadata_only", "excluded"}


def validate_registry(registry: dict[str, Any]) -> None:
    if int(registry.get("schema_version", 0)) != 1:
        raise ValueError("Source registry schema_version must be 1.")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source registry must contain a non-empty sources list.")
    seen: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in seen:
            raise ValueError(f"Source IDs must be non-empty and unique: {source_id!r}")
        seen.add(source_id)
        if source.get("tier") not in VALID_TIERS:
            raise ValueError(f"Invalid tier for {source_id}: {source.get('tier')!r}")
        if source.get("training_eligibility") not in VALID_ELIGIBILITY:
            raise ValueError(
                f"Invalid training_eligibility for {source_id}: "
                f"{source.get('training_eligibility')!r}"
            )
        for required in ("license_scope", "text_access", "provenance_url", "partition"):
            if not str(source.get(required) or "").strip():
                raise ValueError(f"Source {source_id} is missing {required}.")
        if source["training_eligibility"] in {"eligible", "private_only"} and not source.get(
            "full_text_available"
        ):
            raise ValueError(f"Trainable source {source_id} must have full_text_available=true.")


def model_target(parameter_count: int, current_tokens: int, tokens_per_parameter: float) -> dict[str, Any]:
    target = int(round(parameter_count * tokens_per_parameter))
    gap = max(0, target - current_tokens)
    return {
        "parameters": parameter_count,
        "tokens_per_parameter": tokens_per_parameter,
        "target_training_tokens": target,
        "current_unique_training_tokens": current_tokens,
        "token_gap": gap,
        "coverage_fraction": round(current_tokens / target, 4) if target else 0.0,
        "minimum_passes_over_current_corpus": round(target / current_tokens, 2) if current_tokens else None,
    }


def build_scale_plan(
    registry: dict[str, Any],
    tokenization_manifest: dict[str, Any],
    *,
    model_sizes: list[int],
    tokens_per_parameter: float,
) -> dict[str, Any]:
    validate_registry(registry)
    current_tokens = int(
        tokenization_manifest.get("acceptance", {}).get("unique_training_tokens", 0)
    )
    sources = registry["sources"]
    usable = [s["source_id"] for s in sources if s["training_eligibility"] == "eligible"]
    private_only = [s["source_id"] for s in sources if s["training_eligibility"] == "private_only"]
    conditional = [s["source_id"] for s in sources if s["training_eligibility"] == "conditional"]
    metadata_only = [s["source_id"] for s in sources if s["training_eligibility"] == "metadata_only"]
    excluded = [s["source_id"] for s in sources if s["training_eligibility"] == "excluded"]
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "planning_basis": {
            "token_metric": "deduplicated tokenizer input tokens",
            "tokens_per_parameter": tokens_per_parameter,
            "note": (
                "The ratio is a compute-planning baseline, not a guarantee of lyric quality. "
                "Repeated epochs increase exposure tokens but not corpus diversity."
            ),
        },
        "current_corpus": {
            "unique_training_tokens": current_tokens,
            "retained_documents": int(
                tokenization_manifest.get("splits", {}).get("base", {}).get("train", {}).get("documents", 0)
            ),
            "tokenization_manifest": str(tokenization_manifest.get("corpus_dir", "data/scratch/v1")),
        },
        "model_targets": [
            model_target(size, current_tokens, tokens_per_parameter) for size in sorted(set(model_sizes))
        ],
        "source_status": {
            "eligible_full_text": usable,
            "private_only_full_text": private_only,
            "conditional_full_text": conditional,
            "metadata_only": metadata_only,
            "excluded": excluded,
        },
        "partition_policy": registry["partition_policy"],
        "recommended_sequence": registry["recommended_sequence"],
        "release_rule": (
            "Build and hash shards separately by rights partition. Never merge Tier B/C text into "
            "a Tier A release lineage; compose mixtures from immutable partition manifests."
        ),
    }


def render_markdown(plan: dict[str, Any]) -> str:
    rows = []
    for target in plan["model_targets"]:
        rows.append(
            "| {parameters:,} | {target_training_tokens:,} | {current_unique_training_tokens:,} | "
            "{token_gap:,} | {coverage_fraction:.1%} | {minimum_passes_over_current_corpus} |".format(**target)
        )
    status = plan["source_status"]
    return "\n".join(
        [
            "# Scratch corpus scale plan",
            "",
            f"Generated: {plan['generated_at']}",
            "",
            "The controlling unit is deduplicated tokenizer input tokens, not song count.",
            "",
            "| Parameters | Planning target | Current unique tokens | Gap | Coverage | Current-corpus passes |",
            "|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Source state",
            "",
            f"- Eligible full text: {', '.join(status['eligible_full_text']) or 'none'}",
            f"- Private-only full text: {', '.join(status['private_only_full_text']) or 'none'}",
            f"- Conditional full text: {', '.join(status['conditional_full_text']) or 'none'}",
            f"- Metadata only: {', '.join(status['metadata_only']) or 'none'}",
            f"- Excluded: {', '.join(status['excluded']) or 'none'}",
            "",
            "## Acquisition order",
            "",
            *[f"{index}. {item}" for index, item in enumerate(plan["recommended_sequence"], 1)],
            "",
            "## Non-negotiable release rule",
            "",
            plan["release_rule"],
            "",
        ]
    )


def plan_corpus(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registry_path = Path(args.registry)
    manifest_path = Path(args.tokenization_manifest)
    output_dir = Path(args.output_dir)
    registry = read_json(registry_path)
    manifest = read_json(manifest_path)
    plan = build_scale_plan(
        registry,
        manifest,
        model_sizes=list(args.model_sizes),
        tokens_per_parameter=float(args.tokens_per_parameter),
    )
    plan.update(
        {
            "command": command_record(),
            "source_registry": str(registry_path),
            "tokenization_manifest_path": str(manifest_path),
            "wall_seconds": round(time.monotonic() - started, 3),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "corpus_scale_plan.json", plan)
    (output_dir / "corpus_scale_plan.md").write_text(render_markdown(plan), encoding="utf-8")
    return plan


def add_source_planning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/datasets/scratch_source_registry_v2.json")
    )
    parser.add_argument(
        "--tokenization-manifest",
        type=Path,
        default=Path("data/scratch/v1/tokenization_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2"))
    parser.add_argument(
        "--model-sizes",
        type=int,
        nargs="+",
        default=[30_000_000, 75_000_000, 150_000_000, 300_000_000],
    )
    parser.add_argument("--tokens-per-parameter", type=float, default=20.0)
