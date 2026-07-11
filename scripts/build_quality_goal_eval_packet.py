#!/usr/bin/env python3
"""Build reproducible raw-scored and blinded quality-goal evaluation packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_generation_outputs import load_train_index
from scripts.rank_qwen3_quality import score_row


SCHEMA_VERSION = 1
TARGET_ISSUE_TAGS = {
    "weak_imagery": {"weak_imagery"},
    "generic": {"generic", "generic_motivation"},
    "low_rhyme": {"low_rhyme", "low_rhyme_density"},
    "weak_payoff": {"weak_payoff"},
    "scene_drift": {"scene_drift"},
}
PROMPT_PARITY_KEYS = (
    "prompt_key",
    "prompt",
    "theme_id",
    "theme",
    "instruction_family",
    "prompt_family",
    "evaluation_split",
    "samples_per_model",
)


@dataclass(frozen=True)
class GenerationInput:
    label: str
    path: Path
    file_slug: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation",
        action="append",
        required=True,
        metavar="LABEL=JSONL",
        help="Named generation JSONL. Repeat once per model (at least two).",
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--train-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Training JSONL used for copy-risk checks. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Deterministic blinding seed.")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--max-train-records", type=int, default=5000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def safe_slug(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._").lower()
    return slug or "model"


def parse_generation_specs(specs: Iterable[str]) -> list[GenerationInput]:
    inputs: list[GenerationInput] = []
    labels: set[str] = set()
    slugs: set[str] = set()
    for spec in specs:
        label, separator, raw_path = str(spec).partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not separator or not label or not raw_path:
            raise ValueError(f"Invalid --generation {spec!r}; expected LABEL=JSONL")
        if label in labels:
            raise ValueError(f"Duplicate generation label: {label}")
        slug = safe_slug(label)
        if slug in slugs:
            raise ValueError(f"Generation labels collide as output filename slug: {slug}")
        inputs.append(GenerationInput(label=label, path=Path(raw_path), file_slug=slug))
        labels.add(label)
        slugs.add(slug)
    if len(inputs) < 2:
        raise ValueError("At least two --generation inputs are required for a comparison packet")
    return inputs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"Generation JSONL is empty: {path}")
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_bank(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--prompt-file must contain a non-empty JSON list")
    prompts: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {index} is not an object")
        prompt = str(item.get("prompt") or "").strip()
        prompt_key = str(item.get("prompt_key") or "").strip()
        if not prompt or not prompt_key:
            raise ValueError(f"Prompt entry {index} must have prompt and prompt_key")
        if prompt_key in by_key:
            raise ValueError(f"Duplicate prompt_key in prompt file: {prompt_key}")
        target = item.get("target_line_count")
        if target is None or int(target) != 12:
            raise ValueError(f"Quality-goal prompt {prompt_key} must target exactly 12 lines")
        normalized = dict(item)
        normalized["prompt"] = prompt
        normalized["prompt_key"] = prompt_key
        normalized["target_line_count"] = 12
        prompts.append(normalized)
        by_key[prompt_key] = normalized
    return prompts, by_key


def rows_by_id(path: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_index, row in enumerate(rows, start=1):
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Missing string row_id at {path}:{line_index}")
        if row_id in indexed:
            raise ValueError(f"Duplicate row_id {row_id!r} in {path}")
        if "raw_generated_text" not in row or not isinstance(row["raw_generated_text"], str):
            raise ValueError(f"Missing string raw_generated_text for row_id {row_id!r} in {path}")
        indexed[row_id] = row
    return indexed


def prompt_metadata(row: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Project a generation row onto the exact metadata defined by its prompt bank."""
    return {key: row.get(key) for key in PROMPT_PARITY_KEYS}


def validate_generation_matrix(
    generations: list[GenerationInput],
    rows_by_label: dict[str, dict[str, dict[str, Any]]],
    prompts: list[dict[str, Any]],
    prompt_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    labels = [item.label for item in generations]
    reference_ids = set(rows_by_label[labels[0]])
    for label in labels[1:]:
        actual_ids = set(rows_by_label[label])
        if actual_ids != reference_ids:
            missing = sorted(reference_ids - actual_ids)
            extra = sorted(actual_ids - reference_ids)
            raise ValueError(
                f"Generation row-id sets differ for {label}: missing={missing[:5]}, extra={extra[:5]}"
            )

    per_label_prompt_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for row_id in sorted(reference_ids):
        reference_metadata: dict[str, Any] | None = None
        for label in labels:
            row = rows_by_label[label][row_id]
            prompt_key = str(row.get("prompt_key") or "")
            expected = prompt_lookup.get(prompt_key)
            if expected is None:
                raise ValueError(f"Unknown prompt_key {prompt_key!r} for row_id {row_id!r} in {label}")
            expected_metadata = prompt_metadata(expected, expected)
            actual_metadata = prompt_metadata(row, expected)
            if actual_metadata != expected_metadata:
                mismatches = {
                    key: {"expected": value, "actual": actual_metadata.get(key)}
                    for key, value in expected_metadata.items()
                    if actual_metadata.get(key) != value
                }
                raise ValueError(
                    f"Prompt metadata mismatch for row_id {row_id!r} in {label}: {mismatches}"
                )
            if reference_metadata is None:
                reference_metadata = actual_metadata
            elif actual_metadata != reference_metadata:
                raise ValueError(f"Prompt metadata differs across models for row_id {row_id!r}")
            per_label_prompt_counts[label][prompt_key] += 1
            row_target = row.get("target_line_count")
            if row_target not in (None, 12):
                raise ValueError(
                    f"Generation row target_line_count must be null (raw mode) or 12 for {row_id!r} in {label}"
                )

    for prompt in prompts:
        prompt_key = prompt["prompt_key"]
        expected_count = int(prompt.get("samples_per_model") or 1)
        for label in labels:
            actual_count = per_label_prompt_counts[label][prompt_key]
            if actual_count != expected_count:
                raise ValueError(
                    f"Prompt {prompt_key!r} expected {expected_count} rows for {label}, got {actual_count}"
                )
    return sorted(reference_ids)


def extract_run_fingerprint(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_fingerprints: set[str] = set()
    missing_rows = 0
    for row in rows:
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        value = row.get("run_fingerprint") or settings.get("run_fingerprint")
        if isinstance(value, str) and value.strip():
            row_fingerprints.add(value.strip())
        else:
            missing_rows += 1
    if len(row_fingerprints) > 1:
        raise ValueError(f"Multiple generation run fingerprints found in {path}: {sorted(row_fingerprints)}")

    sidecar_path = path.with_name(f"{path.stem}.run_manifest.json")
    sidecar_fingerprint: str | None = None
    sidecar_sha256: str | None = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        value = sidecar.get("fingerprint") if isinstance(sidecar, dict) else None
        sidecar_fingerprint = value.strip() if isinstance(value, str) and value.strip() else None
        sidecar_sha256 = sha256_file(sidecar_path)

    row_fingerprint = next(iter(row_fingerprints), None)
    if row_fingerprint and sidecar_fingerprint and row_fingerprint != sidecar_fingerprint:
        raise ValueError(f"Generation run fingerprint conflicts with sidecar for {path}")
    return {
        "fingerprint": row_fingerprint or sidecar_fingerprint,
        "row_fingerprint": row_fingerprint,
        "rows_missing_fingerprint": missing_rows,
        "sidecar_path": str(sidecar_path.resolve()) if sidecar_path.is_file() else None,
        "sidecar_sha256": sidecar_sha256,
    }


def target_issue_flags(scored: dict[str, Any]) -> dict[str, bool]:
    structural = scored["structural_metrics"]
    tags = set(scored.get("quality_tags") or [])
    flags = {
        "exact12": bool(structural.get("line_count") == 12 and structural.get("target_line_count") == 12),
        "slur": bool(structural.get("slur_count", 0) > 0),
        "prompt_leakage": bool(structural.get("prompt_leakage")),
        "incomplete_ending": bool(structural.get("incomplete_ending")),
        "high_copy": bool(structural.get("high_copy_similarity")),
    }
    flags.update({name: bool(tags & aliases) for name, aliases in TARGET_ISSUE_TAGS.items()})
    return flags


def score_raw_rows(
    row_ids: list[str],
    rows: dict[str, dict[str, Any]],
    *,
    prompt_targets: dict[str, int],
    train_index: list[dict[str, Any]],
    ngram_size: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row_id in row_ids:
        source = rows[row_id]
        raw_text = source["raw_generated_text"]
        scoring_input = dict(source)
        # score_row normally prefers generated_text. Overwriting it is intentional:
        # the evaluation contract forbids scoring postprocessed generation text.
        scoring_input["generated_text"] = raw_text
        scored = score_row(
            scoring_input,
            prompt_targets=prompt_targets,
            train_index=train_index,
            ngram_size=ngram_size,
            similarity_threshold=similarity_threshold,
        )
        scored["score_source"] = {
            "field": "raw_generated_text",
            "postprocessing_used": False,
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        }
        scored["target_issue_flags"] = target_issue_flags(scored)
        scored_rows.append(scored)
    return scored_rows


def summarize_scored_rows(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_rows)
    flag_names = [
        "exact12",
        "slur",
        "prompt_leakage",
        "incomplete_ending",
        "high_copy",
        *TARGET_ISSUE_TAGS,
    ]
    counts = {
        name: sum(bool(row["target_issue_flags"][name]) for row in scored_rows) for name in flag_names
    }
    rates = {name: round(count / total, 4) if total else 0.0 for name, count in counts.items()}
    quality_scores = [float(row["quality_score"]) for row in scored_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "score_source_field": "raw_generated_text",
        "postprocessing_used": False,
        "rows": total,
        "counts": counts,
        "rates": rates,
        "exact_12_line_count": counts["exact12"],
        "exact_12_line_rate": rates["exact12"],
        # Keep the complete requested metric vocabulary together, even though
        # exact12 is a success rate rather than a defect rate.
        "issue_counts": counts,
        "issue_rates": rates,
        "failure_issue_counts": {name: counts[name] for name in flag_names if name != "exact12"},
        "failure_issue_rates": {name: rates[name] for name in flag_names if name != "exact12"},
        "average_quality_score": round(statistics.mean(quality_scores), 4) if quality_scores else 0.0,
        "median_quality_score": round(statistics.median(quality_scores), 4) if quality_scores else 0.0,
    }


def candidate_alias(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"C{index + 1:02d}"


def build_blinded_packet(
    generations: list[GenerationInput],
    row_ids: list[str],
    rows_by_label: dict[str, dict[str, dict[str, Any]]],
    prompt_lookup: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = sorted(item.label for item in generations)
    rng = random.Random(seed)
    comparisons: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for row_id in row_ids:
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        comparison_id = stable_hash("quality-goal-comparison", str(seed), row_id)[:20]
        first_row = rows_by_label[labels[0]][row_id]
        prompt = prompt_lookup[str(first_row["prompt_key"])]
        candidates: list[dict[str, Any]] = []
        private_candidates: list[dict[str, Any]] = []
        for index, label in enumerate(shuffled_labels):
            alias = candidate_alias(index)
            raw_text = rows_by_label[label][row_id]["raw_generated_text"]
            candidates.append({"candidate_alias": alias, "lyrics": raw_text})
            private_candidates.append({"candidate_alias": alias, "generation_label": label})
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "prompt_metadata": prompt,
                "candidates": candidates,
                "review": {"winner_alias": None, "ranking": [], "tie": False, "notes": ""},
            }
        )
        assignments.append(
            {
                "comparison_id": comparison_id,
                "source_row_id": row_id,
                "candidates": private_candidates,
            }
        )
    public_packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_type": "blinded_raw_generation_comparison",
        "score_visibility": "hidden",
        "model_identity_visibility": "hidden",
        "comparisons": comparisons,
    }
    private_key = {
        "schema_version": SCHEMA_VERSION,
        "key_type": "private_model_alias_key",
        "blinding_seed": seed,
        "assignments": assignments,
    }
    return public_packet, private_key


def build_evaluation_packet(
    generations: list[GenerationInput],
    *,
    prompt_file: Path,
    train_jsonls: list[Path],
    output_dir: Path,
    seed: int,
    ngram_size: int = 5,
    similarity_threshold: float = 0.85,
    max_train_records: int = 5000,
) -> dict[str, Any]:
    if len(generations) < 2:
        raise ValueError("At least two generation inputs are required for a comparison packet")
    labels = [item.label for item in generations]
    if len(set(labels)) != len(labels):
        raise ValueError("Generation labels must be unique")
    slugs = [item.file_slug for item in generations]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Generation output filename slugs must be unique")
    if ngram_size <= 0:
        raise ValueError("ngram_size must be > 0")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if max_train_records <= 0:
        raise ValueError("max_train_records must be > 0")
    for path in train_jsonls:
        if not path.is_file():
            raise FileNotFoundError(path)

    prompts, prompt_lookup = load_prompt_bank(prompt_file)
    raw_rows = {item.label: read_jsonl(item.path) for item in generations}
    indexed_rows = {
        item.label: rows_by_id(item.path, raw_rows[item.label]) for item in generations
    }
    row_ids = validate_generation_matrix(generations, indexed_rows, prompts, prompt_lookup)

    prompt_targets = {prompt["prompt"]: 12 for prompt in prompts}
    train_index = load_train_index(train_jsonls, ngram_size, max_train_records)
    scored_by_label: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    generation_inputs: list[dict[str, Any]] = []

    # All validation and scoring happen before the first write, so malformed or
    # incomparable inputs cannot leave a plausible-looking partial packet.
    for item in generations:
        scored = score_raw_rows(
            row_ids,
            indexed_rows[item.label],
            prompt_targets=prompt_targets,
            train_index=train_index,
            ngram_size=ngram_size,
            similarity_threshold=similarity_threshold,
        )
        for row in scored:
            row["generation_label"] = item.label
        scored_by_label[item.label] = scored
        summaries[item.label] = summarize_scored_rows(scored)
        summaries[item.label]["generation_label"] = item.label
        run_evidence = extract_run_fingerprint(item.path, raw_rows[item.label])
        if not run_evidence["fingerprint"] or run_evidence["rows_missing_fingerprint"]:
            raise ValueError(f"Generation run fingerprint is missing from one or more rows in {item.path}")
        generation_inputs.append(
            {
                "label": item.label,
                "path": str(item.path.resolve()),
                "sha256": sha256_file(item.path),
                "rows": len(raw_rows[item.label]),
                "run": run_evidence,
                "raw_scored_output": str((output_dir / "raw_scored" / f"{item.file_slug}.jsonl").resolve()),
                "summary_output": str((output_dir / "summaries" / f"{item.file_slug}.json").resolve()),
            }
        )

    public_packet, private_key = build_blinded_packet(
        generations,
        row_ids,
        indexed_rows,
        prompt_lookup,
        seed=seed,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_contract": {
            "text_field": "raw_generated_text",
            "postprocessing_used": False,
            "required_target_line_count": 12,
            "identical_row_id_sets": True,
            "identical_prompt_metadata": True,
            "blinding_seed": seed,
            "ngram_size": ngram_size,
            "similarity_threshold": similarity_threshold,
            "max_train_records": max_train_records,
        },
        "prompt_input": {
            "path": str(prompt_file.resolve()),
            "sha256": sha256_file(prompt_file),
            "prompts": len(prompts),
        },
        "training_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in train_jsonls
        ],
        "training_index_records": len(train_index),
        "generation_inputs": generation_inputs,
        "row_count_per_model": len(row_ids),
        "row_id_set_sha256": stable_hash(*row_ids),
        "outputs": {
            "comparison_packet": str((output_dir / "comparison_packet.json").resolve()),
            "private_alias_key": str((output_dir / "comparison_alias_key.private.json").resolve()),
        },
    }

    for item in generations:
        write_jsonl(output_dir / "raw_scored" / f"{item.file_slug}.jsonl", scored_by_label[item.label])
        write_json(output_dir / "summaries" / f"{item.file_slug}.json", summaries[item.label])
    write_json(output_dir / "comparison_packet.json", public_packet)
    write_json(output_dir / "comparison_alias_key.private.json", private_key)
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    generations = parse_generation_specs(args.generation)
    manifest = build_evaluation_packet(
        generations,
        prompt_file=args.prompt_file,
        train_jsonls=args.train_jsonl,
        output_dir=args.output_dir,
        seed=args.seed,
        ngram_size=args.ngram_size,
        similarity_threshold=args.similarity_threshold,
        max_train_records=args.max_train_records,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
