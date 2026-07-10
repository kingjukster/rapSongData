#!/usr/bin/env python3
"""Evaluate local rap generation sweep outputs with deterministic metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SLUR_RE = re.compile(
    r"\b(?:nigga|niggas|nigger|niggers|fag|faggot|faggots|dyke|kike|spic|chink|gook|tranny)\b",
    re.I,
)
PROMPT_LEAKAGE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"as an ai\b|"
    r"i\s+(?:cannot|can't)\s+(?:comply|fulfill|provide|write|assist)\b|"
    r"i\s+will\s+write\b|"
    r"here\s+(?:are|is)\b|"
    r"(?:lyrics|verse|hook|chorus)\s*:"
    r")",
    re.I,
)
INCOMPLETE_ENDING_RE = re.compile(r"\b(?:and|but|or|so|cause|because|with|without|to|for|from|that|where|when|while)$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rap generation sweep JSONL outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Sweep JSONL to evaluate.")
    parser.add_argument("--prompts", type=Path, default=None, help="Optional prompt JSON/TXT with target line counts.")
    parser.add_argument(
        "--train-corpus",
        type=Path,
        action="append",
        default=[],
        help="Optional train JSONL/corpus file for nearest-neighbor copy-risk checks. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output metrics JSON path.")
    parser.add_argument("--sample-md", type=Path, default=None, help="Optional issue sample Markdown path.")
    parser.add_argument("--target-line-count", type=int, default=None, help="Fallback target line count.")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--max-train-records", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=20)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(value: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(value or ""))


def lines(value: Any) -> list[str]:
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def requested_line_count(prompt: str) -> int | None:
    match = re.search(r"\b(?:exactly\s+)?(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, re.I)
    if match:
        return int(match.group(1))
    return None


def repeated_line_ratio(output_lines: list[str]) -> float:
    normalized = [normalize_text(line) for line in output_lines]
    normalized = [line for line in normalized if line]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(normalized)


def has_incomplete_ending(output_lines: list[str]) -> bool:
    if not output_lines:
        return True
    last = output_lines[-1].strip()
    if not last:
        return True
    if last.endswith("?"):
        return True
    last_words = words(last)
    if len(last_words) <= 3 and not re.search(r"[.!]$", last):
        return True
    return bool(INCOMPLETE_ENDING_RE.search(last))


def has_prompt_leakage(value: Any) -> bool:
    """Detect assistant preambles or labels without flagging normal lyric phrases."""
    for line in str(value or "").splitlines():
        if PROMPT_LEAKAGE_LINE_RE.search(line):
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def load_prompt_targets(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    targets: dict[str, int] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Prompt JSON must contain a list.")
        for item in payload:
            if isinstance(item, str):
                prompt = item
                target = requested_line_count(prompt)
            elif isinstance(item, dict):
                prompt = str(item.get("prompt") or item.get("instruction") or "")
                raw_target = item.get("target_line_count") or item.get("line_count")
                target = int(raw_target) if raw_target is not None else requested_line_count(prompt)
            else:
                continue
            if prompt and target:
                targets[prompt] = target
        return targets
    for line in text.splitlines():
        prompt = line.strip()
        target = requested_line_count(prompt)
        if prompt and target:
            targets[prompt] = target
    return targets


def generated_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "completion", "assistant", "output", "text", "raw_generated_text"):
        if row.get(key):
            return str(row[key])
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return ""


def extract_training_target(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    training_text = str(row.get("training_text") or "")
    chatml_matches = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", training_text, re.S)
    if chatml_matches:
        return chatml_matches[-1].strip()
    if "Assistant:" in training_text:
        return training_text.rsplit("Assistant:", 1)[-1].strip()
    return str(row.get("target_completion") or row.get("completion") or "")


def token_ngrams(text: str, ngram_size: int) -> set[tuple[str, ...]]:
    tokens = words(normalize_text(text))
    if not tokens:
        return set()
    if len(tokens) < ngram_size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + ngram_size]) for index in range(0, len(tokens) - ngram_size + 1)}


def load_train_index(paths: list[Path], ngram_size: int, max_records: int) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
            for row in rows:
                text = extract_training_target(row)
                grams = token_ngrams(text, ngram_size)
                if grams:
                    index.append({"path": str(path), "text": text, "ngrams": grams})
                if len(index) >= max_records:
                    return index
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            for block in re.split(r"\n\s*\n", text):
                grams = token_ngrams(block, ngram_size)
                if grams:
                    index.append({"path": str(path), "text": block, "ngrams": grams})
                if len(index) >= max_records:
                    return index
    return index


def copy_similarity(text: str, train_index: list[dict[str, Any]], ngram_size: int) -> tuple[float, dict[str, Any] | None]:
    grams = token_ngrams(text, ngram_size)
    if not grams or not train_index:
        return 0.0, None
    best_score = 0.0
    best_item: dict[str, Any] | None = None
    for item in train_index:
        overlap = len(grams & item["ngrams"])
        if not overlap:
            continue
        score = overlap / max(1, len(grams))
        if score > best_score:
            best_score = score
            best_item = item
    return best_score, best_item


def pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.input)
    prompt_targets = load_prompt_targets(args.prompts)
    train_index = load_train_index(args.train_corpus, args.ngram_size, args.max_train_records)

    per_row: list[dict[str, Any]] = []
    per_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_counter: Counter[str] = Counter()

    for index, row in enumerate(rows):
        prompt = str(row.get("prompt") or "")
        text = generated_text(row)
        output_lines = lines(text)
        target = prompt_targets.get(prompt) or requested_line_count(prompt) or args.target_line_count
        line_count = len(output_lines)
        exact_line_match = target is not None and line_count == target
        slur_terms = sorted(set(match.group(0).lower() for match in SLUR_RE.finditer(text)))
        repeated_ratio = repeated_line_ratio(output_lines)
        similarity, neighbor = copy_similarity(text, train_index, args.ngram_size)
        duplicate_key = normalize_text(text)
        if duplicate_key:
            duplicate_counter[duplicate_key] += 1
        metrics = {
            "row_index": index,
            "row_id": row.get("row_id") or row.get("id"),
            "prompt_key": row.get("prompt_key"),
            "prompt": prompt,
            "target_line_count": target,
            "line_count": line_count,
            "line_count_delta": (line_count - target) if target is not None else None,
            "exact_line_match": exact_line_match if target is not None else None,
            "word_count": len(words(text)),
            "repeated_line_ratio": round(repeated_ratio, 4),
            "slur_count": len(slur_terms),
            "slur_terms": slur_terms,
            "hit_token_cap": bool(row.get("hit_token_cap")),
            "finish_reason": row.get("finish_reason"),
            "generation_attempt_count": int(row.get("generation_attempt_count") or 1),
            "underlength_retry_count": int(row.get("underlength_retry_count") or 0),
            "failure_tags": row.get("failure_tags") if isinstance(row.get("failure_tags"), list) else [],
            "prompt_leakage": has_prompt_leakage(text),
            "incomplete_ending": has_incomplete_ending(output_lines),
            "copy_similarity": round(similarity, 4),
            "nearest_neighbor": {
                "path": neighbor["path"],
                "excerpt": str(neighbor["text"])[:240],
            }
            if neighbor and similarity >= args.similarity_threshold
            else None,
        }
        per_row.append(metrics)
        per_prompt[prompt or str(row.get("prompt_key") or index)].append(metrics)

    duplicate_instances = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    line_targets = [item for item in per_row if item["target_line_count"] is not None]
    exact_line_count = sum(1 for item in line_targets if item["exact_line_match"])
    underlength_count = sum(1 for item in line_targets if item["line_count"] < item["target_line_count"])
    overlength_count = sum(1 for item in line_targets if item["line_count"] > item["target_line_count"])
    retry_triggered_count = sum(1 for item in per_row if item["underlength_retry_count"] > 0)
    retry_generation_count = sum(item["underlength_retry_count"] for item in per_row)
    retry_resolved_count = sum(1 for item in line_targets if item["underlength_retry_count"] > 0 and item["exact_line_match"])
    retry_exhausted_count = sum(
        1 for item in line_targets if item["underlength_retry_count"] > 0 and item["line_count"] < item["target_line_count"]
    )
    slur_prompt_count = sum(1 for item in per_row if item["slur_count"] > 0)
    cap_count = sum(1 for item in per_row if item["hit_token_cap"])
    prompt_leakage_count = sum(1 for item in per_row if item["prompt_leakage"])
    incomplete_count = sum(1 for item in per_row if item["incomplete_ending"])
    high_similarity_count = sum(1 for item in per_row if item["copy_similarity"] >= args.similarity_threshold)
    repeated_line_outputs = sum(1 for item in per_row if item["repeated_line_ratio"] > 0.0)

    line_counts = [item["line_count"] for item in per_row]
    word_counts = [item["word_count"] for item in per_row]
    copy_scores = [item["copy_similarity"] for item in per_row]
    repeated_ratios = [item["repeated_line_ratio"] for item in per_row]

    prompt_summaries = []
    for prompt, items in per_prompt.items():
        prompt_targets_local = [item for item in items if item["target_line_count"] is not None]
        prompt_summaries.append(
            {
                "prompt": prompt,
                "rows": len(items),
                "target_line_count": prompt_targets_local[0]["target_line_count"] if prompt_targets_local else None,
                "exact_line_match_rate": pct(
                    sum(1 for item in prompt_targets_local if item["exact_line_match"]), len(prompt_targets_local)
                ),
                "slur_rate": pct(sum(1 for item in items if item["slur_count"] > 0), len(items)),
                "avg_line_count": round(statistics.mean([item["line_count"] for item in items]), 4),
                "avg_copy_similarity": round(statistics.mean([item["copy_similarity"] for item in items]), 4),
            }
        )
    prompt_summaries.sort(key=lambda item: (item["exact_line_match_rate"], -item["slur_rate"], item["prompt"]))

    return {
        "input": str(args.input),
        "prompt_file": str(args.prompts) if args.prompts else None,
        "train_corpus": [str(path) for path in args.train_corpus],
        "config": {
            "similarity_threshold": args.similarity_threshold,
            "ngram_size": args.ngram_size,
            "train_index_records": len(train_index),
        },
        "summary": {
            "row_count": len(per_row),
            "targeted_row_count": len(line_targets),
            "exact_line_match_count": exact_line_count,
            "exact_line_match_rate": pct(exact_line_count, len(line_targets)),
            "underlength_miss_count": underlength_count,
            "underlength_miss_rate": pct(underlength_count, len(line_targets)),
            "overlength_miss_count": overlength_count,
            "overlength_miss_rate": pct(overlength_count, len(line_targets)),
            "underlength_retry_triggered_count": retry_triggered_count,
            "underlength_retry_generation_count": retry_generation_count,
            "underlength_retry_resolved_count": retry_resolved_count,
            "underlength_retry_exhausted_count": retry_exhausted_count,
            "avg_line_count": round(statistics.mean(line_counts), 4) if line_counts else 0.0,
            "avg_word_count": round(statistics.mean(word_counts), 4) if word_counts else 0.0,
            "avg_repeated_line_ratio": round(statistics.mean(repeated_ratios), 4) if repeated_ratios else 0.0,
            "repeated_line_output_count": repeated_line_outputs,
            "repeated_line_output_rate": pct(repeated_line_outputs, len(per_row)),
            "duplicate_generation_instances": duplicate_instances,
            "duplicate_generation_rate": pct(duplicate_instances, len(per_row)),
            "slur_violation_prompt_count": slur_prompt_count,
            "slur_violation_rate": pct(slur_prompt_count, len(per_row)),
            "hit_token_cap_count": cap_count,
            "hit_token_cap_rate": pct(cap_count, len(per_row)),
            "prompt_leakage_count": prompt_leakage_count,
            "prompt_leakage_rate": pct(prompt_leakage_count, len(per_row)),
            "incomplete_ending_count": incomplete_count,
            "incomplete_ending_rate": pct(incomplete_count, len(per_row)),
            "avg_copy_similarity": round(statistics.mean(copy_scores), 4) if copy_scores else 0.0,
            "high_copy_similarity_count": high_similarity_count,
            "high_copy_similarity_rate": pct(high_similarity_count, len(per_row)),
        },
        "per_prompt": prompt_summaries,
        "rows": per_row,
    }


def issue_samples(report: dict[str, Any], max_samples: int) -> str:
    rows = report["rows"]
    buckets = {
        "Line Count Misses": [
            row for row in rows if row["target_line_count"] is not None and row["exact_line_match"] is False
        ],
        "Slur Flags": [row for row in rows if row["slur_count"] > 0],
        "Token Cap Hits": [row for row in rows if row["hit_token_cap"]],
        "Prompt Leakage": [row for row in rows if row["prompt_leakage"]],
        "High Copy Similarity": [row for row in rows if row["nearest_neighbor"]],
    }
    lines_out = [
        "# Generation Evaluation Samples",
        "",
        f"- input: `{report['input']}`",
        f"- row_count: {report['summary']['row_count']}",
        f"- exact_line_match_rate: {report['summary']['exact_line_match_rate']}",
        f"- slur_violation_rate: {report['summary']['slur_violation_rate']}",
        f"- high_copy_similarity_rate: {report['summary']['high_copy_similarity_rate']}",
        "",
    ]
    for title, items in buckets.items():
        lines_out.extend([f"## {title}", ""])
        if not items:
            lines_out.extend(["- none", ""])
            continue
        for item in items[:max_samples]:
            lines_out.append(
                "- "
                + json.dumps(
                    {
                        "row_id": item["row_id"],
                        "prompt_key": item["prompt_key"],
                        "target_line_count": item["target_line_count"],
                        "line_count": item["line_count"],
                        "underlength_retry_count": item["underlength_retry_count"],
                        "slur_terms": item["slur_terms"],
                        "copy_similarity": item["copy_similarity"],
                    },
                    ensure_ascii=False,
                )
            )
        lines_out.append("")
    return "\n".join(lines_out).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.sample_md:
        args.sample_md.parent.mkdir(parents=True, exist_ok=True)
        args.sample_md.write_text(issue_samples(report, args.max_samples), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
