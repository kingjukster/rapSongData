from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


BAD_WORD_PATTERNS = [
    "nigga",
    "niggas",
    "haha",
    "ha!",
    "lol",
    "lmao",
    "uh-huh",
    "oh yeah",
    "bro",
    "yeah",
    "man",
]

L1_CATEGORIES = {
    "reject_dialogue_drift": [
        r"\".*\"",
        r"\b(yo|bro|man|girl|dude)\b",
        r"\b(uh[- ]?huh|hey|yo)\b",
    ],
    "reject_laughter_drift": [r"\b(haha|ha+|lol|lmao|augh?)\b", r"!!!+"],
    "reject_violent_derailment": [r"\b(blood|shoot|murder|kill|gang|stab|weapon)\b"],
    "reject_off_topic": [r"\b(stage|scene|camera|director|cut|shot)\b"],
    "reject_prose_paragraph": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO control pairs for rap SFT quality probes."
    )
    parser.add_argument(
        "--good-jsonl",
        action="append",
        default=[
            "reports/mixed_sft_breakthrough_generation.jsonl",
            "reports/mixed_sft_filtered_500_generation.jsonl",
            "reports/mixed_sft_sweep_v2_step500.jsonl",
        ],
        help="Generation jsonl files with high quality lyric outputs.",
    )
    parser.add_argument(
        "--bad-jsonl",
        action="append",
        default=[
            "reports/mixed_sft_sweep_v2_step1800.jsonl",
            "reports/mixed_sft_sweep_v2_step2000.jsonl",
            "reports/mixed_sft_2000_generation.jsonl",
        ],
        help="Generation jsonl files with drifty / incomplete outputs.",
    )
    parser.add_argument(
        "--quality-pair-jsonl",
        default="data/preferences/rap_quality_pairs.jsonl",
        help="Existing quality pair file to seed strong control examples.",
    )
    parser.add_argument(
        "--output-path",
        default="data/preferences/rap_dpo_control_pairs.jsonl",
        help="Output JSONL for DPO control pairs.",
    )
    parser.add_argument(
        "--summary-path",
        default="data/preferences/rap_dpo_control_pairs_summary.json",
        help="Validation + tag distribution summary.",
    )
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=3)
    parser.add_argument("--min-verse-lines", type=int, default=8)
    parser.add_argument("--max-verse-lines", type=int, default=30)
    return parser.parse_args()


@dataclass(frozen=True)
class PromptPair:
    prompt: str
    chosen: str
    rejected: str
    tags: tuple
    bad_score: float
    source: str


def _safe_get_text(item: Any, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _extract_messages(item: Dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    prompt = _safe_get_text(item, ["prompt", "input", "query"])
    if prompt:
        # prefer explicit prompt over message-derived prompt
        pass
    completion = _safe_get_text(item, ["completion", "output", "text", "response", "result"])
    source = _safe_get_text(item, ["source", "tag", "type"])

    messages = item.get("messages")
    if isinstance(messages, list):
        user_lines: List[str] = []
        assistant_lines: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "").strip().lower()
            content = _safe_get_text(message, ["content"])
            if not content:
                continue
            if role == "user":
                user_lines.append(content.strip())
            elif role == "assistant":
                assistant_lines.append(content.strip())
        if not prompt and user_lines:
            prompt = "\n".join(user_lines)
        if not completion and assistant_lines:
            completion = "\n".join(assistant_lines)
    return prompt, completion, source


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def _tokenize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _line_stats(text: str) -> Dict[str, float]:
    lines = _tokenize_lines(text)
    words = [len(line.split()) for line in lines]
    return {
        "line_count": len(lines),
        "avg_line_words": mean(words) if words else 0.0,
        "max_line_words": max(words) if words else 0,
        "ends_punctuated": bool(lines and lines[-1].rstrip().endswith((".", "!", "?"))),
    }


def _find_tags(text: str) -> tuple[tuple[str, ...], float]:
    lower = text.lower()
    tags: List[str] = []
    score = 0.0
    for tag, patterns in L1_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                if tag not in tags:
                    tags.append(tag)
                score += 1.0
                break

    lines = _tokenize_lines(text)
    stats = _line_stats(text)
    if stats["line_count"] < 6:
        tags.append("reject_line_count_failure")
        score += 1.0
    if stats["line_count"] < 8:
        tags.append("reject_short_output")
        score += 0.75
    if stats["avg_line_words"] > 30:
        tags.append("reject_overlong_lines")
        score += 0.75
    if stats["max_line_words"] > 35:
        tags.append("reject_overlong_lines")
        score += 0.75
    if not stats["ends_punctuated"]:
        tags.append("reject_incomplete_ending")
        score += 0.8
    if stats["line_count"] >= 1 and len(lines) >= 2 and sum(len(ln.split()) for ln in lines) > 60 and len(set([ln[-1] for ln in lines])) < 4:
        tags.append("reject_prose_paragraph")
        score += 0.6

    if sum(pattern in lower for pattern in BAD_WORD_PATTERNS) > 6:
        tags.append("reject_overrun")
        score += 0.5

    return tuple(sorted(set(tags))), score


def _score_goodness(text: str) -> float:
    stats = _line_stats(text)
    tags, _ = _find_tags(text)
    score = 0.0
    if 8 <= stats["line_count"] <= 26:
        score += 2.0
    if stats["ends_punctuated"]:
        score += 0.5
    if "reject_incomplete_ending" in tags:
        score -= 1.0
    if "reject_dialogue_drift" in tags:
        score -= 0.5
    if "reject_laughter_drift" in tags:
        score -= 0.5
    if stats["avg_line_words"] > 0:
        score += min(stats["avg_line_words"], 20) / 20.0
    return score


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSONL at {path}:{i}")
    return rows


def _load_candidate_records(paths: List[str], kind: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        for item in _load_jsonl_records(path):
            prompt, completion, source = _extract_messages(item)
            if prompt is None or completion is None:
                continue
            if kind == "quality_pair":
                # already paired records; handled separately
                continue
            tags, bad_score = _find_tags(completion)
            records.append(
                {
                    "prompt": prompt.strip(),
                    "text": completion.strip(),
                    "source": source or path.name,
                    "tags": tags,
                    "bad_score": bad_score,
                }
            )
    return records


def _load_quality_pairs(path: Path) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for item in _load_jsonl_records(path):
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        if not (isinstance(chosen, str) and isinstance(rejected, str)):
            continue
        chosen, rejected = chosen.strip(), rejected.strip()
        if not chosen or not rejected:
            continue
        pairs.append(
            {
                "prompt": prompt.strip(),
                "chosen": chosen,
                "rejected": rejected,
                "source": item.get("source") or path.name,
                "tags": tuple(item.get("tags") or []),
            }
        )
    return pairs


def _select_good_candidates(
    good_records: List[Dict[str, Any]], min_lines: int, max_lines: int
) -> Dict[str, List[Dict[str, Any]]]:
    by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    loose: List[Dict[str, Any]] = []
    for record in good_records:
        stats = _line_stats(record["text"])
        if not (min_lines <= stats["line_count"] <= max_lines):
            continue
        if any(t.startswith("reject_") for t in record["tags"]):
            continue
        score = _score_goodness(record["text"])
        candidate = {
            "prompt": record["prompt"],
            "text": record["text"],
            "source": record["source"],
            "score": score,
        }
        norm = _normalize_prompt(record["prompt"])
        by_prompt[norm].append(candidate)
        loose.append(candidate)
    return {
        "prompted": by_prompt,
        "loose": sorted(loose, key=lambda x: x["score"], reverse=True),
    }


def _build_pairs(
    good_candidates: Dict[str, List[Dict[str, Any]]],
    bad_records: List[Dict[str, Any]],
    seed: int,
    max_pairs: int,
    max_pairs_per_prompt: int,
    explicit_quality_pairs: List[PromptPair],
    source_prefix: str = "generated",
) -> List[PromptPair]:
    rng = random.Random(seed)
    pairs: List[PromptPair] = list(explicit_quality_pairs)
    used_chosen: set[str] = set()
    per_prompt_used: Counter[str] = Counter()

    for pair in explicit_quality_pairs:
        if pair.prompt:
            per_prompt_used[_normalize_prompt(pair.prompt)] += 1

    bad_records = sorted(bad_records, key=lambda x: x["bad_score"], reverse=True)
    for bad in bad_records:
        if len(pairs) >= max_pairs:
            break
        prompt = bad["prompt"]
        norm_prompt = _normalize_prompt(prompt)
        if per_prompt_used[norm_prompt] >= max_pairs_per_prompt:
            continue

        chosen_list = good_candidates["prompted"].get(norm_prompt, good_candidates["loose"])
        if not chosen_list:
            continue

        chosen_idx = 0
        while chosen_idx < len(chosen_list) and chosen_list[chosen_idx]["text"] in used_chosen:
            chosen_idx += 1
        if chosen_idx >= len(chosen_list):
            chosen_idx = rng.randrange(len(chosen_list))

        chosen = chosen_list[chosen_idx]
        used_chosen.add(chosen["text"])
        per_prompt_used[norm_prompt] += 1

        tags = tuple(sorted(set(bad["tags"])))
        if not tags:
            tags = ("reject_control_noise",)

        pairs.append(
            PromptPair(
                prompt=prompt.strip(),
                chosen=chosen["text"],
                rejected=bad["text"],
                tags=tags,
                bad_score=bad["bad_score"],
                source=f"{source_prefix}:{bad['source']}",
            )
        )

    return pairs[:max_pairs]


def _coerce_pair(item: Dict[str, Any]) -> Optional[PromptPair]:
    prompt = item.get("prompt")
    chosen = item.get("chosen")
    rejected = item.get("rejected")
    if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
        return None
    prompt = prompt.strip()
    chosen = chosen.strip()
    rejected = rejected.strip()
    if not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None
    raw_tags = item.get("tags", [])
    tags = tuple(sorted({str(t).strip() for t in raw_tags if isinstance(t, str)})) or ("reject_control_noise",)
    return PromptPair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        tags=tags,
        bad_score=1.0,
        source=str(item.get("source") or "quality_pairs"),
    )


def _pair_write(path: Path, pairs: List[PromptPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                        "tags": list(pair.tags),
                        "source": pair.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_summary(path: Path, pairs: List[PromptPair], args: argparse.Namespace) -> None:
    tag_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    line_counts = []
    quality_scores = []
    for pair in pairs:
        source_counter[pair.source] += 1
        for tag in pair.tags:
            tag_counter[tag] += 1
        line_counts.append(_line_stats(pair.rejected)["line_count"])
        quality_scores.append(pair.bad_score)

    summary = {
        "num_pairs": len(pairs),
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "min_verse_lines": args.min_verse_lines,
        "max_verse_lines": args.max_verse_lines,
        "pair_source_counts": dict(sorted(source_counter.items())),
        "tag_distribution": dict(sorted(tag_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "rejected_line_stats": {
            "avg": round(mean(line_counts), 2) if line_counts else 0.0,
            "min": min(line_counts) if line_counts else 0,
            "max": max(line_counts) if line_counts else 0,
        },
        "badness_stats": {
            "avg": round(mean(quality_scores), 3) if quality_scores else 0.0,
            "min": min(quality_scores) if quality_scores else 0.0,
            "max": max(quality_scores) if quality_scores else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    good_records = _load_candidate_records(args.good_jsonl, kind="good")
    bad_records = _load_candidate_records(args.bad_jsonl, kind="bad")
    quality_pairs = [
        pair
        for pair in [_coerce_pair(item) for item in _load_quality_pairs(Path(args.quality_pair_jsonl))]
        if pair is not None
    ]

    candidates = _select_good_candidates(good_records, args.min_verse_lines, args.max_verse_lines)
    pairs = _build_pairs(
        candidates,
        bad_records,
        seed=args.seed,
        max_pairs=args.max_pairs,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        explicit_quality_pairs=quality_pairs,
    )

    output_path = Path(args.output_path)
    _pair_write(output_path, pairs)
    summary_path = Path(args.summary_path)
    _write_summary(summary_path, pairs, args)

    print(f"WROTE {len(pairs)} pairs -> {output_path}")
    print(f"Summary -> {summary_path}")
    pair_tags = Counter(tag for pair in pairs for tag in pair.tags)
    top_tags = pair_tags.most_common(12)
    print(f"Top tags: {top_tags}")


if __name__ == "__main__":
    main()
