#!/usr/bin/env python3
"""Build a strict 12-line quality SFT seed from judged base-model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge/judged_candidates.jsonl")
DEFAULT_OUTPUT = Path("data/training/qwen3_4b_base_12line_v1_quality_sft_seed_v1")
SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-judged", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-lines", type=int, default=12)
    parser.add_argument("--min-calibrated-score", type=float, default=3.70)
    parser.add_argument("--min-pair-delta", type=float, default=0.75)
    parser.add_argument("--max-pairs-per-chosen", type=int, default=3)
    parser.add_argument("--max-hard-negatives", type=int, default=500)
    parser.add_argument("--max-copy-similarity", type=float, default=0.85)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--min-training-ready-sft", type=int, default=500)
    parser.add_argument("--min-training-ready-pairs", type=int, default=1000)
    parser.add_argument(
        "--disallowed-hard-issue",
        action="append",
        default=["weak_imagery", "scene_drift"],
        help="Judge issue or quality tag that excludes a row from strict SFT. Can be repeated.",
    )
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


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]


def stable_int(*parts: str) -> int:
    return int(hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


def clean_content(value: Any) -> str:
    text = str(value or "")
    for token in CHATML_CONTROL_TOKENS:
        text = text.replace(token, "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def lyric_lines(text: Any) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def word_count(text: Any) -> int:
    return len(WORD_RE.findall(str(text or "")))


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", str(text or "").lower())).strip()


def chatml(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = clean_content(message.get("role"))
        content = clean_content(message.get("content"))
        if role and content:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks) + "\n"


def judge(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("judge")
    return value if isinstance(value, dict) else {}


def score(row: dict[str, Any]) -> float:
    for key in ("calibrated_review_score", "combined_quality_score"):
        try:
            value = row.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def heuristic_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("quality_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def judge_quality(row: dict[str, Any]) -> int | None:
    value = row.get("judge_quality", judge(row).get("overall_quality"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def judge_usable(row: dict[str, Any]) -> str:
    return str(row.get("judge_usable_as_is", judge(row).get("usable_as_is") or "")).lower()


def judge_issue(row: dict[str, Any]) -> str:
    return str(row.get("judge_issue", judge(row).get("main_issue") or "")).lower()


def quality_tags(row: dict[str, Any]) -> set[str]:
    tags = row.get("quality_tags") or []
    return {str(tag).lower() for tag in tags}


def structural_metrics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("structural_metrics")
    return value if isinstance(value, dict) else {}


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("row_id") or stable_id(str(row.get("prompt")), str(row.get("lyrics"))))


def prompt_key(row: dict[str, Any]) -> str:
    return str(row.get("prompt_key") or stable_id(str(row.get("prompt"))))


def structural_ok(row: dict[str, Any], *, target_lines: int, max_copy_similarity: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    metrics = structural_metrics(row)
    lines = lyric_lines(row.get("lyrics"))
    if len(lines) != target_lines:
        reasons.append("line_count")
    if metrics and metrics.get("structural_pass") is False:
        reasons.append("structural_fail")
    if metrics.get("exact_line_match") is False:
        reasons.append("not_exact_line_match")
    if metrics.get("prompt_leakage"):
        reasons.append("prompt_leakage")
    if metrics.get("incomplete_ending"):
        reasons.append("incomplete_ending")
    if metrics.get("high_copy_similarity"):
        reasons.append("high_copy_similarity")
    try:
        if float(metrics.get("copy_similarity") or 0.0) >= max_copy_similarity:
            reasons.append("copy_similarity")
    except (TypeError, ValueError):
        pass
    if metrics.get("slur_count"):
        reasons.append("slur_violation")
    if not clean_content(row.get("prompt")) or not clean_content(row.get("lyrics")):
        reasons.append("missing_prompt_or_lyrics")
    return not reasons, reasons


def strict_sft_ok(
    row: dict[str, Any],
    *,
    target_lines: int,
    min_score: float,
    max_copy_similarity: float,
    disallowed_issues: set[str],
) -> tuple[bool, list[str]]:
    ok, reasons = structural_ok(row, target_lines=target_lines, max_copy_similarity=max_copy_similarity)
    if not ok:
        return False, reasons
    if judge_usable(row) != "yes":
        reasons.append("judge_not_usable")
    if score(row) < min_score:
        reasons.append("score_below_min")
    issue = judge_issue(row)
    if issue in disallowed_issues:
        reasons.append(f"judge_issue_{issue}")
    blocked_tags = quality_tags(row) & disallowed_issues
    for tag in sorted(blocked_tags):
        reasons.append(f"quality_tag_{tag}")
    return not reasons, reasons


def sft_record(row: dict[str, Any], *, split: str, target_lines: int) -> dict[str, Any]:
    prompt = clean_content(row.get("prompt"))
    lyrics = clean_content(row.get("lyrics"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": lyrics},
    ]
    return {
        "id": f"seed-sft-{candidate_id(row)}",
        "training_text": chatml(messages),
        "metadata": {
            "source": "qwen3_4b_base_12line_v1_quality_sft_seed_v1",
            "source_model": "qwen3_4b_base_12line_v1",
            "candidate_id": candidate_id(row),
            "row_id": row.get("row_id"),
            "prompt_key": prompt_key(row),
            "split": split,
            "target_line_count": target_lines,
            "actual_line_count": len(lyric_lines(lyrics)),
            "assistant_word_count": word_count(lyrics),
            "calibrated_review_score": score(row),
            "combined_quality_score": row.get("combined_quality_score"),
            "quality_score": row.get("quality_score"),
            "judge_quality": judge_quality(row),
            "judge_issue": judge_issue(row),
            "quality_tags": sorted(quality_tags(row)),
            "license_scope": "synthetic_model_generated_local_audit",
            "selection_rule": "judge_usable_score_copy_structure_no_weak_imagery_or_scene_drift",
        },
    }


def split_prompt_keys(rows: list[dict[str, Any]], *, validation_ratio: float) -> set[str]:
    keys = sorted({prompt_key(row) for row in rows}, key=lambda key: stable_int(key))
    if len(keys) <= 1:
        return set(keys[:1])
    target = max(1, round(len(keys) * min(max(validation_ratio, 0.01), 0.5)))
    return set(keys[:target])


def build_preference_pairs(
    strict_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    validation_prompt_keys: set[str],
    *,
    target_lines: int,
    max_copy_similarity: float,
    min_pair_delta: float,
    max_pairs_per_chosen: int,
) -> list[dict[str, Any]]:
    rows_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        ok, _ = structural_ok(row, target_lines=target_lines, max_copy_similarity=max_copy_similarity)
        if ok:
            rows_by_prompt[prompt_key(row)].append(row)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chosen in sorted(strict_rows, key=lambda row: (-score(row), candidate_id(row))):
        rejected_count = 0
        chosen_quality = judge_quality(chosen)
        rejected_pool = sorted(rows_by_prompt.get(prompt_key(chosen), []), key=lambda row: (score(row), heuristic_score(row)))
        for rejected in rejected_pool:
            if candidate_id(rejected) == candidate_id(chosen):
                continue
            if normalize_text(rejected.get("lyrics")) == normalize_text(chosen.get("lyrics")):
                continue
            score_delta = score(chosen) - score(rejected)
            rejected_quality = judge_quality(rejected)
            judge_delta = (
                chosen_quality - rejected_quality
                if chosen_quality is not None and rejected_quality is not None
                else None
            )
            if score_delta < min_pair_delta and not (judge_delta is not None and judge_delta >= 2):
                continue
            key = (candidate_id(chosen), candidate_id(rejected))
            if key in seen:
                continue
            seen.add(key)
            split = "validation" if prompt_key(chosen) in validation_prompt_keys else "train"
            pairs.append(
                {
                    "id": f"seed-pref-{candidate_id(chosen)}-{candidate_id(rejected)}",
                    "prompt": clean_content(chosen.get("prompt")),
                    "chosen": clean_content(chosen.get("lyrics")),
                    "rejected": clean_content(rejected.get("lyrics")),
                    "metadata": {
                        "source": "qwen3_4b_base_12line_v1_quality_sft_seed_v1",
                        "split": split,
                        "prompt_key": prompt_key(chosen),
                        "chosen_candidate_id": candidate_id(chosen),
                        "rejected_candidate_id": candidate_id(rejected),
                        "chosen_score": round(score(chosen), 4),
                        "rejected_score": round(score(rejected), 4),
                        "score_delta": round(score_delta, 4),
                        "chosen_judge_quality": chosen_quality,
                        "rejected_judge_quality": rejected_quality,
                        "judge_quality_delta": judge_delta,
                        "chosen_issue": judge_issue(chosen),
                        "rejected_issue": judge_issue(rejected),
                    },
                }
            )
            rejected_count += 1
            if rejected_count >= max(1, max_pairs_per_chosen):
                break
    return pairs


def build_hard_negatives(
    strict_ids: set[str],
    all_rows: list[dict[str, Any]],
    *,
    target_lines: int,
    max_copy_similarity: float,
    max_rows: int,
) -> list[dict[str, Any]]:
    negatives: list[dict[str, Any]] = []
    for row in all_rows:
        if candidate_id(row) in strict_ids:
            continue
        ok, _ = structural_ok(row, target_lines=target_lines, max_copy_similarity=max_copy_similarity)
        if not ok:
            continue
        issue = judge_issue(row)
        tags = sorted(quality_tags(row))
        if judge_usable(row) == "yes" and issue not in {"weak_imagery", "scene_drift", "generic", "low_rhyme"}:
            continue
        negatives.append(
            {
                "id": f"seed-hard-negative-{candidate_id(row)}",
                "prompt": clean_content(row.get("prompt")),
                "rejected": clean_content(row.get("lyrics")),
                "metadata": {
                    "source": "qwen3_4b_base_12line_v1_quality_sft_seed_v1",
                    "candidate_id": candidate_id(row),
                    "prompt_key": prompt_key(row),
                    "calibrated_review_score": score(row),
                    "quality_score": row.get("quality_score"),
                    "judge_quality": judge_quality(row),
                    "judge_usable_as_is": judge_usable(row),
                    "judge_issue": issue,
                    "quality_tags": tags,
                },
            }
        )
    negatives.sort(key=lambda row: (float(row["metadata"]["calibrated_review_score"]), row["id"]))
    return negatives[: max(0, max_rows)]


def write_preview(path: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    lines = ["# Qwen3 4B 12-Line Quality SFT Seed Preview", ""]
    lines.extend(["## SFT Examples", ""])
    for row in (train + validation)[:10]:
        metadata = row["metadata"]
        assistant = row["training_text"].rsplit("<|im_start|>assistant\n", 1)[-1].rsplit("<|im_end|>", 1)[0]
        lines.extend(
            [
                f"### {metadata['candidate_id']}",
                "",
                f"- split: `{metadata['split']}`",
                f"- score: `{metadata['calibrated_review_score']}`",
                f"- judge_issue: `{metadata['judge_issue']}`",
                "",
                "```text",
                assistant.strip(),
                "```",
                "",
            ]
        )
    lines.extend(["## Preference Pairs", ""])
    for pair in pairs[:5]:
        metadata = pair["metadata"]
        lines.extend(
            [
                f"### {pair['id']}",
                "",
                f"- split: `{metadata['split']}`",
                f"- score_delta: `{metadata['score_delta']}`",
                f"- rejected_issue: `{metadata['rejected_issue']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    disallowed_issues = {str(issue).lower() for issue in args.disallowed_hard_issue}
    rows = read_jsonl(args.input_judged)
    exclusion_counts: Counter[str] = Counter()
    strict_rows: list[dict[str, Any]] = []
    seen_lyrics: set[tuple[str, str]] = set()
    for row in rows:
        ok, reasons = strict_sft_ok(
            row,
            target_lines=args.target_lines,
            min_score=args.min_calibrated_score,
            max_copy_similarity=args.max_copy_similarity,
            disallowed_issues=disallowed_issues,
        )
        if not ok:
            exclusion_counts.update(reasons)
            continue
        key = (prompt_key(row), normalize_text(row.get("lyrics")))
        if key in seen_lyrics:
            exclusion_counts["duplicate_prompt_lyrics"] += 1
            continue
        seen_lyrics.add(key)
        strict_rows.append(row)

    strict_rows.sort(key=lambda row: (-score(row), candidate_id(row)))
    validation_prompt_keys = split_prompt_keys(strict_rows, validation_ratio=args.validation_ratio)
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for row in strict_rows:
        split = "validation" if prompt_key(row) in validation_prompt_keys else "train"
        record = sft_record(row, split=split, target_lines=args.target_lines)
        if split == "validation":
            validation_records.append(record)
        else:
            train_records.append(record)

    preference_pairs = build_preference_pairs(
        strict_rows,
        rows,
        validation_prompt_keys,
        target_lines=args.target_lines,
        max_copy_similarity=args.max_copy_similarity,
        min_pair_delta=args.min_pair_delta,
        max_pairs_per_chosen=args.max_pairs_per_chosen,
    )
    hard_negatives = build_hard_negatives(
        {candidate_id(row) for row in strict_rows},
        rows,
        target_lines=args.target_lines,
        max_copy_similarity=args.max_copy_similarity,
        max_rows=args.max_hard_negatives,
    )

    prompt_split_leaks = sorted(
        {
            row["metadata"]["prompt_key"]
            for row in train_records
            if row["metadata"]["prompt_key"] in {item["metadata"]["prompt_key"] for item in validation_records}
        }
    )
    if prompt_split_leaks:
        raise SystemExit(f"Prompt split leakage: {prompt_split_leaks[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    pairs_path = args.output_dir / "preference_pairs.jsonl"
    negatives_path = args.output_dir / "hard_negatives.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    preview_path = args.output_dir / "preview.md"

    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    write_jsonl(pairs_path, preference_pairs)
    write_jsonl(negatives_path, hard_negatives)
    write_preview(preview_path, train_records, validation_records, preference_pairs)

    sft_total = len(train_records) + len(validation_records)
    training_ready = sft_total >= args.min_training_ready_sft and len(preference_pairs) >= args.min_training_ready_pairs
    manifest = {
        "name": "qwen3_4b_base_12line_v1_quality_sft_seed_v1",
        "status": "training_ready" if training_ready else "seed_only_not_training_ready",
        "reason": None
        if training_ready
        else "Strict judged 12-line pool is below minimum target counts for main training.",
        "input_judged": str(args.input_judged),
        "output_dir": str(args.output_dir),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "preference_pairs_path": str(pairs_path),
        "hard_negatives_path": str(negatives_path),
        "preview_path": str(preview_path),
        "system_prompt": SYSTEM_PROMPT,
        "selection": {
            "target_lines": args.target_lines,
            "min_calibrated_score": args.min_calibrated_score,
            "min_pair_delta": args.min_pair_delta,
            "max_copy_similarity": args.max_copy_similarity,
            "disallowed_hard_issues": sorted(disallowed_issues),
        },
        "minimum_training_ready_counts": {
            "sft_examples": args.min_training_ready_sft,
            "preference_pairs": args.min_training_ready_pairs,
        },
        "counts": {
            "input_judged_rows": len(rows),
            "strict_sft_examples": sft_total,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
            "strict_prompt_count": len({row["metadata"]["prompt_key"] for row in train_records + validation_records}),
            "preference_pairs": len(preference_pairs),
            "hard_negatives": len(hard_negatives),
            "exclusion_counts": dict(exclusion_counts),
            "judge_issue_counts_in_strict_sft": dict(Counter(row["metadata"]["judge_issue"] for row in train_records + validation_records)),
            "quality_tag_counts_in_strict_sft": dict(
                Counter(tag for row in train_records + validation_records for tag in row["metadata"]["quality_tags"])
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
