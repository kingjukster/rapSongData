#!/usr/bin/env python3
"""Fit leakage-safe local section rerankers and build an unused follow-up packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .mine_judged_song_sections import minhash_signature, minhash_similarity, normalized_line, words
except ImportError:
    from mine_judged_song_sections import minhash_signature, minhash_similarity, normalized_line, words


GENERIC_WORDS = {
    "dream", "dreams", "grind", "hustle", "life", "mind", "pain", "rise", "strong",
    "success", "top", "world", "heart", "fight", "win", "winning", "shine", "destiny",
}
FEATURE_NAMES = [
    "local_score", "word_count", "end_rhyme_density", "internal_rhyme_density",
    "multisyllabic_rhyme_count", "rhyme_chain_continuity", "lexical_diversity",
    "concrete_imagery_count", "topic_continuity", "repeated_line_ratio", "syllable_variance",
    "final_complete", "generic_phrase_count", "avg_line_words", "line_word_std",
    "short_line_ratio", "generic_word_ratio", "first_person_ratio", "numeric_token_ratio",
    "parent_overall", "parent_technical_rhyme", "parent_flow_cadence", "parent_coherence",
    "parent_thematic_depth", "parent_imagery", "parent_ending_strength",
]


def text_features(text: str) -> dict[str, float]:
    lines = [line for line in text.splitlines() if line.strip()]
    token_lines = [words(line) for line in lines]
    tokens = [token for line in token_lines for token in line]
    counts = [len(line) for line in token_lines]
    avg = sum(counts) / len(counts) if counts else 0
    std = math.sqrt(sum((value - avg) ** 2 for value in counts) / len(counts)) if counts else 0
    return {
        "avg_line_words": avg,
        "line_word_std": std,
        "short_line_ratio": sum(value < 5 for value in counts) / max(1, len(counts)),
        "generic_word_ratio": sum(token in GENERIC_WORDS for token in tokens) / max(1, len(tokens)),
        "first_person_ratio": sum(token in {"i", "i'm", "me", "my", "mine"} for token in tokens) / max(1, len(tokens)),
        "numeric_token_ratio": sum(any(char.isdigit() for char in token) for token in tokens) / max(1, len(tokens)),
    }


def feature_dict(row: dict[str, Any]) -> dict[str, float]:
    metrics = row.get("local_metrics") or {}
    parent = row.get("parent_judgment") or {}
    features = {name: float(metrics.get(name) or 0) for name in FEATURE_NAMES}
    features["local_score"] = float(row.get("local_score") or 0)
    features.update(text_features(str(row.get("text") or "")))
    for name in (
        "overall", "technical_rhyme", "flow_cadence", "coherence", "thematic_depth", "imagery", "ending_strength",
    ):
        features[f"parent_{name}"] = float(parent.get(name) or 0)
    return features


def vector(row: dict[str, Any]) -> np.ndarray:
    features = feature_dict(row)
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def validation_song(song_key: str) -> bool:
    return int(hashlib.sha1(str(song_key).encode()).hexdigest()[:8], 16) % 5 == 0


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 1800, learning_rate: float = 0.04, l2: float = 0.04) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    sample_weights = np.where(y > 0, len(y) / (2 * positives), len(y) / (2 * negatives))
    for _ in range(steps):
        probabilities = sigmoid(z @ weights + bias)
        error = (probabilities - y) * sample_weights
        weights -= learning_rate * ((z.T @ error) / len(y) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return sigmoid(z @ model["weights"] + model["bias"])


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if not len(positives) or not len(negatives):
        return 0.5
    wins = sum(float((value > negatives).sum()) + 0.5 * float((value == negatives).sum()) for value in positives)
    return wins / (len(positives) * len(negatives))


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    indexes = np.argsort(-scores)[: min(k, len(scores))]
    return float(y[indexes].mean()) if len(indexes) else 0.0


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": FEATURE_NAMES, "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
        "weights": model["weights"].tolist(), "bias": float(model["bias"]),
        "largest_absolute_coefficients": sorted(
            ({"feature": name, "coefficient": round(float(value), 5)} for name, value in zip(FEATURE_NAMES, model["weights"])),
            key=lambda item: abs(item["coefficient"]), reverse=True,
        )[:12],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration.jsonl"))
    parser.add_argument("--raw-sections", type=Path, default=Path("data/section_mining/section_mining_calibration_raw_3k.jsonl"))
    parser.add_argument("--first-packet", type=Path, default=Path("data/section_mining/section_mining_calibration_selected_800.jsonl"))
    parser.add_argument("--accepted", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_calibration_accepted_capped.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/section_mining/section_mining_calibration_round2_reranked.jsonl"))
    parser.add_argument("--model-output", type=Path, default=Path("data/section_mining/section_reranker_v1.json"))
    parser.add_argument("--summary", type=Path, default=Path("reports/section_reranker_v1_summary.json"))
    parser.add_argument("--technical-limit", type=int, default=300)
    parser.add_argument("--clean-limit", type=int, default=100)
    parser.add_argument("--story-limit", type=int, default=40)
    parser.add_argument(
        "--families", nargs="+", choices=("technical", "clean", "story"),
        default=("technical", "clean"),
    )
    parser.add_argument("--minhash-threshold", type=float, default=0.84)
    args = parser.parse_args()
    started = time.time()
    judged = read_jsonl(args.judgments)
    raw = read_jsonl(args.raw_sections)
    first = read_jsonl(args.first_packet)
    accepted = read_jsonl(args.accepted)
    models: dict[str, dict[str, Any]] = {}
    model_report: dict[str, Any] = {}

    for family in args.families:
        group = [row for row in judged if row["family"] == family]
        train = [row for row in group if not validation_song(row["song_key"])]
        validation = [row for row in group if validation_song(row["song_key"])]
        x_train = np.stack([vector(row) for row in train])
        y_train = np.asarray([int(row["computed_strict_pass"]) for row in train], dtype=np.float64)
        strict_model = fit_logistic(x_train, y_train)
        generic_model = None
        if family in {"clean", "story"}:
            generic_y = np.asarray([int(int(row["genericness"]) <= 2 and "generic_filler" not in row["critical_failure_flags"]) for row in train], dtype=np.float64)
            generic_model = fit_logistic(x_train, generic_y)
        x_val = np.stack([vector(row) for row in validation])
        y_val = np.asarray([int(row["computed_strict_pass"]) for row in validation], dtype=np.float64)
        learned = predict(strict_model, x_val)
        if generic_model is not None:
            learned = learned * (0.75 + 0.25 * predict(generic_model, x_val))
        baseline = np.asarray([float(row["local_score"]) for row in validation])
        k = max(1, int(y_val.sum()))
        model_report[family] = {
            "train_rows": len(train), "validation_rows": len(validation), "validation_positive": int(y_val.sum()),
            "learned_auc": round(auc(y_val, learned), 4), "baseline_auc": round(auc(y_val, baseline), 4),
            "learned_precision_at_positive_k": round(precision_at_k(y_val, learned, k), 4),
            "baseline_precision_at_positive_k": round(precision_at_k(y_val, baseline, k), 4),
        }
        model_report[family]["adopt_learned_reranker"] = bool(
            model_report[family]["learned_auc"] > model_report[family]["baseline_auc"]
            and model_report[family]["learned_precision_at_positive_k"] > model_report[family]["baseline_precision_at_positive_k"]
        )
        # Refit on all labels after the held-out measurement for packet scoring.
        x_all = np.stack([vector(row) for row in group])
        y_all = np.asarray([int(row["computed_strict_pass"]) for row in group], dtype=np.float64)
        full_strict = fit_logistic(x_all, y_all)
        full_generic = None
        if family in {"clean", "story"}:
            full_generic = fit_logistic(x_all, np.asarray([int(int(row["genericness"]) <= 2 and "generic_filler" not in row["critical_failure_flags"]) for row in group], dtype=np.float64))
        models[family] = {
            "strict": full_strict,
            "generic": full_generic,
            "adopt_learned": model_report[family]["adopt_learned_reranker"],
        }

    first_hashes = {row["text_hash"] for row in first}
    first_line_sets = [{normalized_line(line) for line in row["text"].splitlines()} for row in first]
    first_signatures = [minhash_signature(row["text"]) for row in first]
    accepted_counts = Counter((row["song_key"], row["family"]) for row in accepted)
    packet: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    packet_song_counts: Counter[tuple[str, str]] = Counter()
    dedupe_counts: Counter[str] = Counter()
    candidate_rows = []
    for row in raw:
        if row["text_hash"] in first_hashes:
            continue
        family = row["family"]
        if accepted_counts[(row["song_key"], family)] >= 2:
            dedupe_counts["source_already_at_accepted_cap"] += 1
            continue
        x = vector(row)[None, :]
        learned_score = float(predict(models[family]["strict"], x)[0])
        score = learned_score if models[family]["adopt_learned"] else float(row["local_score"])
        generic_probability = None
        if models[family]["generic"] is not None:
            generic_probability = float(predict(models[family]["generic"], x)[0])
            if models[family]["adopt_learned"]:
                score *= 0.75 + 0.25 * generic_probability
        candidate_rows.append({
            **row,
            "reranker_score": round(score, 6),
            "selection_score_source": "learned_reranker" if models[family]["adopt_learned"] else "validated_local_score_fallback",
            "learned_strict_probability": round(learned_score, 6),
            "nongeneric_probability": None if generic_probability is None else round(generic_probability, 6),
        })
    candidate_rows.sort(key=lambda row: (-row["reranker_score"], -row["local_score"], row["section_id"]))
    packet_signatures: list[tuple[int, ...]] = []
    packet_lines: list[set[str]] = []
    for row in candidate_rows:
        family = row["family"]
        limit = {
            "technical": args.technical_limit,
            "clean": args.clean_limit,
            "story": args.story_limit,
        }[family]
        if family_counts[family] >= limit:
            continue
        remaining_slots = 2 - accepted_counts[(row["song_key"], family)]
        pre_judge_cap = max(1, remaining_slots * 3)
        if packet_song_counts[(row["song_key"], family)] >= pre_judge_cap:
            dedupe_counts["round2_per_song_cap"] += 1
            continue
        line_set = {normalized_line(line) for line in row["text"].splitlines()}
        if any(len(line_set & existing) >= 10 for existing in first_line_sets) or any(len(line_set & existing) >= 10 for existing in packet_lines):
            dedupe_counts["line_overlap_duplicate"] += 1
            continue
        signature = minhash_signature(row["text"])
        if any(minhash_similarity(signature, existing) >= args.minhash_threshold for existing in first_signatures) or any(minhash_similarity(signature, existing) >= args.minhash_threshold for existing in packet_signatures):
            dedupe_counts["minhash_duplicate"] += 1
            continue
        packet.append(row)
        packet_signatures.append(signature)
        packet_lines.append(line_set)
        packet_song_counts[(row["song_key"], family)] += 1
        family_counts[family] += 1
    write_jsonl(args.output, packet)
    model_payload = {
        family: {
            "strict": serializable_model(model["strict"]),
            "generic": serializable_model(model["generic"]) if model["generic"] is not None else None,
            "adopt_learned": model["adopt_learned"],
        } for family, model in models.items()
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1, "wall_time_seconds": round(time.time() - started, 3),
        "command": " ".join([sys.executable, *sys.argv]), "model_validation": model_report,
        "round2_packet": {"rows": len(packet), "family_counts": dict(family_counts),
            "unique_songs": {family: len({row['song_key'] for row in packet if row['family'] == family}) for family in args.families},
            "unique_hashes": len({row["text_hash"] for row in packet}), "dedupe_counts": dict(dedupe_counts)},
        "outputs": {"packet": str(args.output), "model": str(args.model_output)},
        "paid_api_work_submitted": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
