#!/usr/bin/env python3
"""Rank structurally valid Qwen3 rap generations with local quality heuristics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_generation_outputs import (
    SLUR_RE,
    copy_similarity,
    has_incomplete_ending,
    has_prompt_leakage,
    load_prompt_targets,
    load_train_index,
    pct,
    repeated_line_ratio,
    requested_line_count,
    words,
)


STOPWORDS = {
    "a",
    "after",
    "an",
    "and",
    "about",
    "as",
    "at",
    "but",
    "by",
    "during",
    "for",
    "from",
    "in",
    "into",
    "it",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "while",
    "with",
}

ABSTRACT_WORDS = {
    "ambition",
    "believe",
    "dream",
    "dreams",
    "faith",
    "fear",
    "feeling",
    "feelings",
    "grind",
    "heart",
    "hope",
    "hustle",
    "life",
    "mind",
    "pain",
    "pressure",
    "pride",
    "promise",
    "promises",
    "soul",
    "struggle",
    "success",
    "temptation",
    "trust",
}

CONCRETE_WORDS = {
    "alley",
    "apartment",
    "bag",
    "bench",
    "block",
    "booth",
    "brake",
    "brick",
    "bus",
    "cable",
    "cash",
    "ceiling",
    "chain",
    "chair",
    "city",
    "clock",
    "coat",
    "concrete",
    "corner",
    "crowd",
    "curb",
    "desk",
    "door",
    "elevator",
    "floor",
    "glass",
    "hall",
    "jacket",
    "kitchen",
    "lamp",
    "light",
    "lights",
    "mic",
    "mirror",
    "notebook",
    "pavement",
    "pen",
    "phone",
    "platform",
    "pocket",
    "porch",
    "rain",
    "rail",
    "room",
    "roof",
    "shoes",
    "sidewalk",
    "speaker",
    "speakers",
    "stage",
    "station",
    "steps",
    "store",
    "street",
    "table",
    "tile",
    "track",
    "tracks",
    "train",
    "turnstile",
    "wall",
    "window",
}

ACTION_WORDS = {
    "bend",
    "build",
    "carry",
    "choose",
    "close",
    "count",
    "drag",
    "fold",
    "hold",
    "keep",
    "lift",
    "lock",
    "move",
    "open",
    "pace",
    "press",
    "pull",
    "push",
    "repair",
    "rise",
    "run",
    "shape",
    "step",
    "stitch",
    "turn",
    "walk",
    "write",
}

GENERIC_MOTIVATION_RE = re.compile(
    r"\b(?:"
    r"believe in (?:myself|yourself)|"
    r"chase (?:my |the )?dreams?|"
    r"follow (?:my |your )?dreams?|"
    r"from the bottom to the top|"
    r"keep (?:on )?(?:going|grinding|pushing)|"
    r"never give up|"
    r"nothing can stop me|"
    r"rise above|"
    r"stay strong|"
    r"work hard|"
    r"make it through|"
    r"shine bright|"
    r"reach the sky"
    r")\b",
    re.I,
)

AWKWARD_RE = re.compile(
    r"\b(?:"
    r"who'main|"
    r"that rejections|"
    r"this promises|"
    r"these pressure|"
    r"those loyalty|"
    r"my are|"
    r"i is|"
    r"we was"
    r")\b",
    re.I,
)

LINE_LABEL_RE = re.compile(r"^\s*(?:verse|hook|chorus|bridge|intro|outro)\s*:", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")

WEIGHTS = {
    "theme_adherence": 0.20,
    "scene_coherence": 0.15,
    "ending_payoff": 0.15,
    "specific_imagery": 0.15,
    "rhyme_density": 0.10,
    "cadence_line_shape": 0.10,
    "originality_low_cliche": 0.10,
    "grammar_naturalness": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Sweep JSONL to rank.")
    parser.add_argument("--prompts", type=Path, default=None, help="Prompt JSON/TXT with target line counts.")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown review queue path.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Full ranked JSONL path.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Quality ranker summary JSON path.")
    parser.add_argument(
        "--train-corpus",
        type=Path,
        action="append",
        default=[],
        help="Optional train JSONL/corpus for copy-risk checks. Can be repeated.",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--max-train-records", type=int, default=5000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def generated_text(row: dict[str, Any]) -> str:
    return str(row.get("generated_text") or row.get("completion") or row.get("text") or "")


def normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(text or ""))]


def content_terms(text: str) -> set[str]:
    return {token for token in normalized_tokens(text) if len(token) > 2 and token not in STOPWORDS}


def extract_theme(prompt: str) -> str:
    match = re.search(r"\babout\s+(.+?)(?:\.|$)", prompt, re.I)
    return match.group(1).strip() if match else ""


def prompt_family(prompt: str) -> str:
    if re.search(r"\bradio-safe|clean\b", prompt, re.I):
        return "clean"
    if re.search(r"\binternal rhymes?\b", prompt, re.I):
        return "technical"
    if re.search(r"\bmove the same scene forward\b", prompt, re.I):
        return "story"
    if re.search(r"\bmelodic\b", prompt, re.I):
        return "melodic"
    return "unknown"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def rhyme_key(word: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", word.lower())
    if len(cleaned) <= 3:
        return cleaned
    vowels = "aeiouy"
    for index in range(len(cleaned) - 2, -1, -1):
        if cleaned[index] in vowels:
            return cleaned[index:]
    return cleaned[-4:]


def theme_adherence(prompt: str, text: str) -> tuple[float, dict[str, Any], list[str]]:
    theme = extract_theme(prompt)
    terms = sorted(content_terms(theme))
    if not terms:
        return 0.75, {"theme": theme, "theme_terms": [], "matched_theme_terms": []}, []
    output_terms = content_terms(text)
    matched = sorted(term for term in terms if term in output_terms)
    score = clamp(len(matched) / max(1, len(terms)))

    lines = output_lines(text)
    tags: list[str] = []
    if len(lines) >= 4:
        midpoint = max(1, len(lines) // 2)
        first_hits = sum(1 for term in terms if term in content_terms(" ".join(lines[:midpoint])))
        second_hits = sum(1 for term in terms if term in content_terms(" ".join(lines[midpoint:])))
        if first_hits > 0 and second_hits == 0 and score < 0.8:
            tags.append("topic_thinning")
            score = min(score, 0.62)
    return score, {"theme": theme, "theme_terms": terms, "matched_theme_terms": matched}, tags


def imagery_score(text: str) -> tuple[float, dict[str, Any], list[str]]:
    tokens = normalized_tokens(text)
    concrete = [token for token in tokens if token in CONCRETE_WORDS]
    actions = [token for token in tokens if token in ACTION_WORDS]
    abstract = [token for token in tokens if token in ABSTRACT_WORDS]
    line_count = max(1, len(output_lines(text)))
    concrete_per_line = len(concrete) / line_count
    action_per_line = len(actions) / line_count
    abstract_ratio = len(abstract) / max(1, len(tokens))
    score = clamp((concrete_per_line / 0.55) * 0.65 + (action_per_line / 0.45) * 0.35)
    tags: list[str] = []
    if len(concrete) < 3 or score < 0.35:
        tags.append("weak_imagery")
    if abstract_ratio > 0.12 and len(concrete) < 6:
        tags.append("abstract_overload")
        score = min(score, 0.55)
    return (
        score,
        {
            "concrete_terms": sorted(set(concrete)),
            "action_terms": sorted(set(actions)),
            "abstract_terms": sorted(set(abstract)),
            "concrete_count": len(concrete),
            "action_count": len(actions),
            "abstract_count": len(abstract),
            "abstract_ratio": round(abstract_ratio, 4),
        },
        tags,
    )


def rhyme_density(text: str) -> tuple[float, dict[str, Any], list[str]]:
    lines = output_lines(text)
    end_keys: list[str] = []
    internal_hits = 0
    internal_slots = 0
    for line in lines:
        line_words = [word for word in normalized_tokens(line) if len(word) >= 3]
        if line_words:
            end_keys.append(rhyme_key(line_words[-1]))
        keys = [rhyme_key(word) for word in line_words if len(word) >= 4]
        counts = Counter(key for key in keys if key)
        internal_hits += sum(count - 1 for count in counts.values() if count > 1)
        internal_slots += max(1, len(keys))
    grouped_end_lines = sum(count for count in Counter(end_keys).values() if count > 1)
    end_rhyme_rate = grouped_end_lines / max(1, len(lines))
    internal_rate = internal_hits / max(1, internal_slots)
    score = clamp(0.72 * end_rhyme_rate + 0.28 * min(1.0, internal_rate * 5.0))
    tags = ["low_rhyme_density"] if score < 0.25 else []
    return (
        score,
        {
            "end_rhyme_rate": round(end_rhyme_rate, 4),
            "internal_rhyme_rate": round(internal_rate, 4),
            "end_rhyme_keys": end_keys,
        },
        tags,
    )


def cadence_score(text: str) -> tuple[float, dict[str, Any], list[str]]:
    lines = output_lines(text)
    counts = [len(normalized_tokens(line)) for line in lines]
    if not counts:
        return 0.0, {"line_word_counts": [], "avg_line_words": 0.0, "max_line_words": 0}, ["prose_line_shape"]
    avg_words = statistics.mean(counts)
    max_words = max(counts)
    stdev_words = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    too_long_rate = sum(1 for count in counts if count > 18) / len(counts)
    avg_penalty = max(0.0, avg_words - 13.5) / 8.0 + max(0.0, 6.0 - avg_words) / 6.0
    max_penalty = max(0.0, max_words - 20.0) / 10.0
    variance_penalty = max(0.0, stdev_words - 4.5) / 6.0
    score = clamp(1.0 - avg_penalty - max_penalty - variance_penalty - too_long_rate * 0.35)
    tags = ["prose_line_shape"] if avg_words > 14.5 or max_words > 22 or too_long_rate > 0.25 else []
    return (
        score,
        {
            "line_word_counts": counts,
            "avg_line_words": round(avg_words, 2),
            "max_line_words": max_words,
            "line_word_stdev": round(stdev_words, 2),
            "too_long_line_rate": round(too_long_rate, 4),
        },
        tags,
    )


def ending_score(prompt: str, text: str, theme_terms: list[str]) -> tuple[float, dict[str, Any], list[str]]:
    lines = output_lines(text)
    if not lines:
        return 0.0, {"ending_line": "", "ending_complete": False}, ["weak_payoff"]
    last = lines[-1]
    last_tokens = content_terms(last)
    concrete_or_action = any(token in CONCRETE_WORDS or token in ACTION_WORDS for token in last_tokens)
    theme_hit = any(term in last_tokens for term in theme_terms)
    cliche_hit = bool(GENERIC_MOTIVATION_RE.search(last))
    complete = not has_incomplete_ending([last])
    score = 0.0
    score += 0.42 if complete else 0.0
    score += 0.24 if concrete_or_action else 0.0
    score += 0.20 if theme_hit else 0.0
    score += 0.14 if not cliche_hit else 0.0
    tags = ["weak_payoff"] if score < 0.55 else []
    return (
        clamp(score),
        {
            "ending_line": last,
            "ending_complete": complete,
            "ending_has_concrete_or_action": concrete_or_action,
            "ending_theme_hit": theme_hit,
            "ending_cliche_hit": cliche_hit,
        },
        tags,
    )


def originality_score(text: str) -> tuple[float, dict[str, Any], list[str]]:
    matches = GENERIC_MOTIVATION_RE.findall(text)
    tokens = normalized_tokens(text)
    generic_words = [token for token in tokens if token in {"dream", "dreams", "grind", "shine", "strong", "hustle"}]
    penalty = min(0.75, 0.22 * len(matches) + 0.04 * len(generic_words))
    score = clamp(1.0 - penalty)
    tags = ["generic_motivation"] if matches or len(generic_words) >= 5 else []
    return score, {"generic_phrase_hits": len(matches), "generic_word_hits": len(generic_words)}, tags


def grammar_score(text: str) -> tuple[float, dict[str, Any], list[str]]:
    repeated_words = re.findall(r"\b([A-Za-z]{3,})\s+\1\b", text, flags=re.I)
    awkward = AWKWARD_RE.findall(text)
    labels = [line for line in output_lines(text) if LINE_LABEL_RE.search(line)]
    unmatched = sum(text.count(left) != text.count(right) for left, right in [("(", ")"), ("[", "]")])
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    penalty = (
        0.18 * len(repeated_words)
        + 0.25 * len(awkward)
        + 0.18 * len(labels)
        + 0.2 * unmatched
        + (0.35 if non_ascii_count else 0.0)
    )
    score = clamp(1.0 - min(0.8, penalty))
    tags = ["awkward_phrase"] if penalty > 0 else []
    return (
        score,
        {
            "repeated_word_count": len(repeated_words),
            "awkward_phrase_count": len(awkward),
            "label_line_count": len(labels),
            "unmatched_bracket_count": unmatched,
            "non_ascii_count": non_ascii_count,
        },
        tags,
    )


def scene_score(prompt: str, text: str, theme_score: float, imagery: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    lines = output_lines(text)
    if not lines:
        return 0.0, {"anchor_line_rate": 0.0}, ["scene_drift"]
    concrete_terms = set(imagery.get("concrete_terms") or [])
    theme_terms = content_terms(extract_theme(prompt))
    anchor_terms = concrete_terms | theme_terms
    anchor_line_count = 0
    for line in lines:
        line_terms = content_terms(line)
        if line_terms & anchor_terms:
            anchor_line_count += 1
    anchor_rate = anchor_line_count / len(lines)
    story_prompt = prompt_family(prompt) == "story"
    score = clamp(0.55 * anchor_rate + 0.30 * theme_score + 0.15 * min(1.0, len(concrete_terms) / 5.0))
    tags: list[str] = []
    if story_prompt and score < 0.55:
        tags.append("scene_drift")
    elif theme_score < 0.45 and anchor_rate < 0.45:
        tags.append("scene_drift")
    return score, {"anchor_line_rate": round(anchor_rate, 4), "story_prompt": story_prompt}, tags


def structural_metrics(
    row: dict[str, Any],
    *,
    prompt_targets: dict[str, int],
    train_index: list[dict[str, Any]],
    ngram_size: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    text = generated_text(row)
    lines = output_lines(text)
    target = row.get("target_line_count") or prompt_targets.get(prompt) or requested_line_count(prompt)
    target = int(target) if target is not None else None
    slur_terms = sorted(set(match.group(0).lower() for match in SLUR_RE.finditer(text)))
    similarity, neighbor = copy_similarity(text, train_index, ngram_size)
    exact = target is not None and len(lines) == target
    incomplete = has_incomplete_ending(lines)
    prompt_leakage = has_prompt_leakage(text)
    high_copy = similarity >= similarity_threshold
    structural_pass = bool(
        exact
        and not slur_terms
        and not prompt_leakage
        and not incomplete
        and not high_copy
        and not bool(row.get("hit_token_cap"))
    )
    return {
        "target_line_count": target,
        "line_count": len(lines),
        "exact_line_match": exact,
        "line_count_delta": (len(lines) - target) if target is not None else None,
        "word_count": len(words(text)),
        "repeated_line_ratio": round(repeated_line_ratio(lines), 4),
        "slur_terms": slur_terms,
        "slur_count": len(slur_terms),
        "prompt_leakage": prompt_leakage,
        "incomplete_ending": incomplete,
        "hit_token_cap": bool(row.get("hit_token_cap")),
        "finish_reason": row.get("finish_reason"),
        "copy_similarity": round(similarity, 4),
        "high_copy_similarity": high_copy,
        "nearest_neighbor": {
            "path": neighbor["path"],
            "excerpt": str(neighbor["text"])[:240],
        }
        if neighbor and high_copy
        else None,
        "structural_pass": structural_pass,
    }


def score_row(
    row: dict[str, Any],
    *,
    prompt_targets: dict[str, int],
    train_index: list[dict[str, Any]],
    ngram_size: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    text = generated_text(row)
    structural = structural_metrics(
        row,
        prompt_targets=prompt_targets,
        train_index=train_index,
        ngram_size=ngram_size,
        similarity_threshold=similarity_threshold,
    )
    theme_score, theme_metrics, theme_tags = theme_adherence(prompt, text)
    imagery, imagery_metrics, imagery_tags = imagery_score(text)
    scene, scene_metrics, scene_tags = scene_score(prompt, text, theme_score, imagery_metrics)
    ending, ending_metrics, ending_tags = ending_score(prompt, text, theme_metrics["theme_terms"])
    rhyme, rhyme_metrics, rhyme_tags = rhyme_density(text)
    cadence, line_stats, cadence_tags = cadence_score(text)
    originality, originality_metrics, originality_tags = originality_score(text)
    grammar, grammar_metrics, grammar_tags = grammar_score(text)

    dimensions = {
        "theme_adherence": theme_score,
        "scene_coherence": scene,
        "ending_payoff": ending,
        "specific_imagery": imagery,
        "rhyme_density": rhyme,
        "cadence_line_shape": cadence,
        "originality_low_cliche": originality,
        "grammar_naturalness": grammar,
    }
    weighted_score = sum(dimensions[name] * WEIGHTS[name] for name in WEIGHTS)
    structural_penalty = 0.0 if structural["structural_pass"] else 0.25
    quality_score = round(clamp(weighted_score - structural_penalty), 4)
    tags = sorted(
        set(
            theme_tags
            + imagery_tags
            + scene_tags
            + ending_tags
            + rhyme_tags
            + cadence_tags
            + originality_tags
            + grammar_tags
        )
    )
    return {
        "candidate_id": row.get("row_id") or row.get("id") or row.get("candidate_index"),
        "row_id": row.get("row_id"),
        "candidate_index": row.get("candidate_index"),
        "sample_index": row.get("sample_index"),
        "prompt_key": row.get("prompt_key"),
        "prompt_family": row.get("prompt_family") or prompt_family(prompt),
        "theme": row.get("theme") or theme_metrics["theme"],
        "style": row.get("style"),
        "prompt": prompt,
        "lyrics": text,
        "quality_score": quality_score,
        "quality_tags": tags,
        "quality_dimensions": {key: round(value, 4) for key, value in dimensions.items()},
        "structural_metrics": structural,
        "rhyme_metrics": rhyme_metrics,
        "line_length_stats": line_stats,
        "theme_metrics": theme_metrics,
        "imagery_metrics": imagery_metrics,
        "scene_metrics": scene_metrics,
        "ending_metrics": ending_metrics,
        "originality_metrics": originality_metrics,
        "grammar_metrics": grammar_metrics,
        "retry_metrics": {
            "generation_attempt_count": row.get("generation_attempt_count"),
            "underlength_retry_count": row.get("underlength_retry_count"),
            "accepted_attempt_index": row.get("accepted_attempt_index"),
        },
        "human_quality_score": None,
        "human_notes": "",
    }


def summarize(ranked: list[dict[str, Any]], *, top_n: int) -> dict[str, Any]:
    structural_pass = [row for row in ranked if row["structural_metrics"]["structural_pass"]]
    scores = [row["quality_score"] for row in ranked]
    top = ranked[:top_n]
    top50 = ranked[:50]
    tag_counts = Counter(tag for row in ranked for tag in row["quality_tags"])
    top50_tags = Counter(tag for row in top50 for tag in row["quality_tags"])
    return {
        "rows": len(ranked),
        "structural_pass_count": len(structural_pass),
        "structural_pass_rate": pct(len(structural_pass), len(ranked)),
        "avg_quality_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "median_quality_score": round(statistics.median(scores), 4) if scores else 0.0,
        "top_n": len(top),
        "top_n_avg_quality_score": round(statistics.mean([row["quality_score"] for row in top]), 4) if top else 0.0,
        "top50_generic_motivation_rate": pct(top50_tags.get("generic_motivation", 0), len(top50)),
        "top50_scene_drift_rate": pct(top50_tags.get("scene_drift", 0), len(top50)),
        "top50_prose_line_shape_rate": pct(top50_tags.get("prose_line_shape", 0), len(top50)),
        "quality_tag_counts": dict(tag_counts.most_common()),
        "top50_quality_tag_counts": dict(top50_tags.most_common()),
    }


def write_review_queue(path: Path, rows: list[dict[str, Any]], *, source: Path) -> None:
    lines_out = [
        "# Qwen3-4B Base 12-Line Quality Review Queue",
        "",
        f"- source: `{source}`",
        f"- candidates: {len(rows)}",
        "- human label scale: 5 strong / 4 good / 3 bland / 2 weak / 1 reject",
        "",
    ]
    for rank, row in enumerate(rows, start=1):
        structural = row["structural_metrics"]
        tags = ", ".join(row["quality_tags"]) or "none"
        dimensions = ", ".join(f"{key}={value}" for key, value in row["quality_dimensions"].items())
        lines_out.extend(
            [
                f"## {rank}. {row['candidate_id']}",
                "",
                f"- human_label: ",
                f"- quality_score: `{row['quality_score']}`",
                f"- quality_tags: `{tags}`",
                f"- prompt_family: `{row.get('prompt_family')}`",
                f"- structural: lines `{structural['line_count']}/{structural['target_line_count']}`, "
                f"slurs `{structural['slur_count']}`, leakage `{structural['prompt_leakage']}`, "
                f"incomplete `{structural['incomplete_ending']}`, copy `{structural['copy_similarity']}`",
                f"- rhyme: end `{row['rhyme_metrics']['end_rhyme_rate']}`, "
                f"internal `{row['rhyme_metrics']['internal_rhyme_rate']}`",
                f"- line_shape: avg_words `{row['line_length_stats']['avg_line_words']}`, "
                f"max_words `{row['line_length_stats']['max_line_words']}`",
                f"- dimensions: {dimensions}",
                f"- prompt: {row['prompt']}",
                "",
                "```text",
                row["lyrics"].strip(),
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines_out), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")
    prompt_targets = load_prompt_targets(args.prompts)
    train_index = load_train_index(args.train_corpus, args.ngram_size, args.max_train_records)
    ranked = [
        score_row(
            row,
            prompt_targets=prompt_targets,
            train_index=train_index,
            ngram_size=args.ngram_size,
            similarity_threshold=args.similarity_threshold,
        )
        for row in read_jsonl(args.input)
    ]
    ranked.sort(
        key=lambda row: (
            row["structural_metrics"]["structural_pass"],
            row["quality_score"],
            -len(row["quality_tags"]),
            row["candidate_id"] or "",
        ),
        reverse=True,
    )
    summary = {
        "input": str(args.input),
        "output_md": str(args.output_md),
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "ranker": "qwen3_4b_base_12line_v1_quality_ranker_v1",
        "local_only": True,
        "weights": WEIGHTS,
        "train_index_records": len(train_index),
        **summarize(ranked, top_n=args.top_n),
    }
    write_jsonl(args.output_jsonl, ranked)
    write_review_queue(args.output_md, ranked[: args.top_n], source=args.input)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
