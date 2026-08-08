#!/usr/bin/env python3
"""Build balanced v4 SFT by replacing v3 technical/clean rows with strict mined sections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
FAMILIES = ("melodic", "story", "technical", "clean")
SPLIT_COUNTS = {"train": 26, "validation": 4, "test": 3}
PROMPTS = {
    "story": {
        "train": [
            "Write exactly 12 original rap lines that develop one concrete scene from beginning to end. Keep the details specific, the cadence natural, and the last line resolved. Lyrics only.",
            "Create a self-contained 12-line story rap excerpt. Make every line advance the same event or idea, use vivid physical detail, and finish with an earned payoff.",
            "Write 12 original narrative rap lines with clear continuity, natural rhyme, specific imagery, and a complete final line. Avoid generic filler and return only lyrics.",
        ],
        "validation": [
            "Produce exactly 12 original story-driven rap lines with a coherent sequence, concrete detail, natural phrasing, and a decisive ending. Lyrics only.",
        ],
        "test": [
            "In exactly 12 original rap lines, tell one focused scene with continuous development, vivid specificity, and a resolved closing line.",
        ],
    },
    "technical": {
        "train": [
            "Write exactly 12 lines of original technical rap lyrics. Use controlled internal and multisyllabic rhyme while keeping every line coherent. Finish with a complete payoff. Return lyrics only.",
            "Create a self-contained 12-line technical rap excerpt with dense but natural rhyme chains, steady cadence, clear meaning, and a resolved final line. Do not name or imitate artists.",
            "Write 12 original rap lines with strong internal rhyme, multisyllabic connections, semantic continuity, and an earned ending. Avoid filler and return only lyrics.",
        ],
        "validation": [
            "Produce exactly 12 original technical rap lines with disciplined rhyme density, natural phrasing, coherent development, and a decisive closing line. Lyrics only.",
        ],
        "test": [
            "In exactly 12 original rap lines, demonstrate controlled multisyllabic and internal rhyme without sacrificing clarity or ending strength. Return only the verse.",
        ],
    },
    "clean": {
        "train": [
            "Write exactly 12 clean, radio-safe original rap lines with natural cadence, concrete specificity, personality, and a complete final payoff. Avoid profanity, slurs, explicit sex, graphic violence, and drug promotion.",
            "Create a self-contained 12-line clean rap excerpt. Keep it vivid and technically competent without generic motivational filler, profanity, slurs, explicit content, or graphic violence. Lyrics only.",
            "Write 12 original family-safe rap lines with coherent development, natural phrasing, specific imagery, and an earned ending. Keep the writing expressive rather than sanitized or generic.",
        ],
        "validation": [
            "Produce exactly 12 clean original rap lines that remain vivid, specific, coherent, and rhythmically natural, ending on a complete payoff. Return lyrics only.",
        ],
        "test": [
            "In exactly 12 radio-safe rap lines, deliver personality, concrete detail, coherent meaning, competent rhyme, and a strong final line without generic filler.",
        ],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", text.lower())).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def stable_int(*parts: str) -> int:
    return int(hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16], 16)


def assistant_text(training_text: str) -> str:
    match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", training_text, re.S)
    return match.group(1).strip() if match else ""


def rank_key(row: dict[str, Any], family: str) -> tuple[Any, ...]:
    family_score = int(row["technical_rhyme"]) if family == "technical" else 6 - int(row["genericness"])
    return (
        -int(row["overall"]), -family_score, -int(row["ending_strength"]),
        -int(row["coherence"]), -int(row["thematic_specificity"]),
        -int(row["natural_phrasing_cadence"]), -float(row.get("local_score") or 0), row["section_id"],
    )


def choose_unique_sources(rows: list[dict[str, Any]], family: str, quota: int, excluded_songs: set[str]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used = set(excluded_songs)
    for row in sorted((row for row in rows if row["family"] == family), key=lambda item: rank_key(item, family)):
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


def assign_splits(rows: list[dict[str, Any]], family: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: stable_int(str(seed), family, str(row["song_key"]), row["section_id"]))
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        count = SPLIT_COUNTS[split]
        result[split] = ordered[offset:offset + count]
        offset += count
    return result


def prompt_for(row: dict[str, Any], split: str) -> str:
    prompts = PROMPTS[row["family"]][split]
    return prompts[stable_int(row["section_id"], split) % len(prompts)]


def source_training_record(
    row: dict[str, Any], split: str, accepted_path: Path, accepted_sha: str,
    summary_path: Path, summary_sha: str, seed: int,
) -> dict[str, Any]:
    text = str(row["text"]).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    prompt = prompt_for(row, split)
    candidate_id = hashlib.sha256(row["section_id"].encode()).hexdigest()[:16]
    prompt_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    family = row["family"]
    dimensions = {
        "overall": int(row["overall"]), "coherence": int(row["coherence"]),
        "thematic_specificity": int(row["thematic_specificity"]),
        "ending_strength": int(row["ending_strength"]),
        "natural_phrasing_cadence": int(row["natural_phrasing_cadence"]),
        "technical_rhyme": int(row["technical_rhyme"]), "safety": int(row["safety"]),
        "self_contained_excerpt": int(row["self_contained_excerpt"]),
        "nongenericness": 6 - int(row["genericness"]),
    }
    minimums = {
        "overall": 4, "coherence": 4, "thematic_specificity": 4, "ending_strength": 4,
        "natural_phrasing_cadence": 4, "technical_rhyme": 4 if family == "technical" else 3,
        "safety": 3 if family in {"technical", "story"} else 5, "self_contained_excerpt": 4,
    }
    if family == "clean":
        minimums["nongenericness"] = 4
    training_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{text}<|im_end|>\n"
    )
    created_at = datetime.now(timezone.utc).isoformat()
    license_scope = "source_corpus_rights_not_verified_for_local_research"
    return {
        "id": f"section-v4-{candidate_id}", "training_text": training_text,
        "metadata": {
            "source": "qwen3_4b_12line_section_mined_v4",
            "source_bucket": "gpt55_strict_section_keep", "candidate_id": candidate_id,
            "section_id": row["section_id"], "source_song_key": str(row["song_key"]),
            "source_line_range": [row["source_start_line"], row["source_end_line"]],
            "prompt_key": prompt_key, "theme": None, "prompt_family": family, "split": split,
            "target_line_count": 12, "actual_line_count": 12, "license_scope": license_scope,
            "normalized_text_sha256": text_sha256(text),
            "source_provenance": {
                "candidate_id": candidate_id, "judge_source_path": str(accepted_path),
                "judge_source_sha256": accepted_sha, "generation_source_path": str(accepted_path),
                "generation_source_sha256": accepted_sha, "generation_summary_path": str(summary_path),
                "generation_summary_sha256": summary_sha,
                "generation_run_id": "source_song_section_mining_rounds_1_2",
                "judge_run_id": "gpt55_section_quality_rounds_1_2", "model_id": "source_corpus_excerpt",
                "model_revision": accepted_sha[:16], "model_revision_source": "content_addressed_combined_accepted_pool",
                "rng_provenance": {"protocol": "sha256_source_group_split", "manual_seed": seed},
                "created_at": created_at, "license_scope": license_scope,
            },
            "automated_calibration": {
                "label_source": "automated_consensus", "human_reviewed": False,
                "criteria_version": "gpt55_section_strict_v1", "judge_provider": "openai_api",
                "judge_model": "gpt-5.5-2026-04-23", "judge_overall_quality": int(row["overall"]),
                "judge_usable_as_is": "yes", "judge_main_issue": "none",
                "judge_dimension_scores": dimensions, "calibrated_review_score": float(row["overall"]),
                "minimum_calibrated_score": 4.0, "required_dimension_minimums": minimums,
                "critical_failure_flags": row["critical_failure_flags"], "model_gate_agreed": True,
            },
            "format": "qwen_chatml_training_text",
            "selection": {"family_quota": 33, "one_section_per_source_song": True},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=Path("data/training/qwen3_4b_12line_auto_calibrated_v3"))
    parser.add_argument("--accepted", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_combined_accepted_capped.jsonl"))
    parser.add_argument("--accepted-summary", type=Path, default=Path("reports/section_quality_judge_gpt55_combined_analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v4"))
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    started = time.time()
    accepted = read_jsonl(args.accepted)
    accepted_sha = sha256_file(args.accepted)
    summary_sha = sha256_file(args.accepted_summary)

    # Select clean first because it has fewer unique source songs, then exclude those songs from technical.
    clean = choose_unique_sources(accepted, "clean", 33, set())
    clean_songs = {str(row["song_key"]) for row in clean}
    technical = choose_unique_sources(accepted, "technical", 33, clean_songs)
    source_splits = {"clean": assign_splits(clean, "clean", args.seed), "technical": assign_splits(technical, "technical", args.seed)}

    outputs: dict[str, list[dict[str, Any]]] = {}
    retained_candidate_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        v3_rows = read_jsonl(args.v3_dir / f"{split}.jsonl")
        retained = [row for row in v3_rows if row.get("metadata", {}).get("prompt_family") in {"melodic", "story"}]
        retained_candidate_ids.update(str(row.get("metadata", {}).get("candidate_id")) for row in retained)
        for row in retained:
            row["metadata"] = dict(row["metadata"])
            row["metadata"]["v4_role"] = "retained_strong_family_control"
        replacements = [
            source_training_record(row, split, args.accepted, accepted_sha, args.accepted_summary, summary_sha, args.seed)
            for family in ("technical", "clean") for row in source_splits[family][split]
        ]
        outputs[split] = sorted(retained + replacements, key=lambda row: stable_int(str(args.seed), split, row["id"]))

    all_rows = [row for split in outputs.values() for row in split]
    text_hashes = [text_sha256(assistant_text(row["training_text"])) for row in all_rows]
    if len(text_hashes) != len(set(text_hashes)):
        raise RuntimeError("Duplicate assistant text detected in v4")
    prompt_keys_by_split = {split: {row["metadata"]["prompt_key"] for row in rows} for split, rows in outputs.items()}
    source_songs_by_split = {
        split: {str(row["metadata"].get("source_song_key")) for row in rows if row["metadata"].get("source_song_key")}
        for split, rows in outputs.items()
    }
    if any(source_songs_by_split[left] & source_songs_by_split[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Source-song leakage across v4 splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, rows in outputs.items():
        paths[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(paths[split], rows)
    v3_pairs = read_jsonl(args.v3_dir / "preference_pairs.jsonl")
    pairs = [pair for pair in v3_pairs if str(pair.get("metadata", {}).get("chosen_candidate_id")) in retained_candidate_ids]
    for pair in pairs:
        pair["metadata"] = dict(pair["metadata"])
        pair["metadata"]["v4_role"] = "retained_strong_family_preference_control"
    paths["preferences"] = args.output_dir / "preference_pairs.jsonl"
    write_jsonl(paths["preferences"], pairs)

    family_by_split = {
        split: dict(sorted(Counter(row["metadata"]["prompt_family"] for row in rows).items()))
        for split, rows in outputs.items()
    }
    manifest = {
        "name": "qwen3_4b_12line_section_mined_v4", "status": "training_ready",
        "dataset_version": "v4", "selection_policy": "retain_v3_strong_families_replace_weak_families_with_gpt55_strict_sections",
        "seed": args.seed, "wall_time_seconds": round(time.time() - started, 3),
        "command": " ".join([sys.executable, *sys.argv]),
        "family_quotas": {family: 33 for family in FAMILIES},
        "split_targets_per_family": SPLIT_COUNTS,
        "source_inputs": {
            "v3_manifest": {"path": str(args.v3_dir / "manifest.json"), "sha256": sha256_file(args.v3_dir / "manifest.json")},
            "accepted_sections": {"path": str(args.accepted), "sha256": accepted_sha},
            "accepted_summary": {"path": str(args.accepted_summary), "sha256": summary_sha},
        },
        "counts": {"selected": len(all_rows), **{f"{split}_rows": len(rows) for split, rows in outputs.items()},
            "preference_pairs": len(pairs), "technical_available": sum(row["family"] == "technical" for row in accepted),
            "clean_available": sum(row["family"] == "clean" for row in accepted)},
        "family_presence_by_split": family_by_split,
        "source_song_overlap": {
            "train_validation": len(source_songs_by_split["train"] & source_songs_by_split["validation"]),
            "train_test": len(source_songs_by_split["train"] & source_songs_by_split["test"]),
            "validation_test": len(source_songs_by_split["validation"] & source_songs_by_split["test"]),
        },
        "prompt_key_overlap": {
            "train_validation": len(prompt_keys_by_split["train"] & prompt_keys_by_split["validation"]),
            "train_test": len(prompt_keys_by_split["train"] & prompt_keys_by_split["test"]),
            "validation_test": len(prompt_keys_by_split["validation"] & prompt_keys_by_split["test"]),
        },
        "duplicate_assistant_rows": len(text_hashes) - len(set(text_hashes)),
        "source_section_policy": {"one_per_source_song": True, "artist_and_title_excluded": True, "strict_gate_required": True,
            "rights_note": "Source-corpus rights were not verified; artifact is marked for local research."},
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest["output_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
