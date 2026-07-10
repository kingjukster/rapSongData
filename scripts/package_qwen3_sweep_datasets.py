"""Package local Qwen3-4B sweep curation outputs into next-stage datasets.

This script is intentionally local-only. It reads JSONL files produced by a
generation sweep / curation pass and writes training-ready buckets:

* positive SFT examples from strong raw keepers
* repair-pair SFT examples from raw -> postprocessed generations
* DPO-style chosen/rejected preference pairs
* a manual repair queue for high-scoring fixable outputs

It does not call OpenAI, Hugging Face, or any external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "Generate original rap lyrics. Do not copy existing songs."
REPAIR_SYSTEM_PROMPT = (
    "Repair rap generations. Preserve the prompt intent and usable lines, "
    "but remove artifacts, weak endings, prompt drift, and formatting problems."
)

TEXT_FIELDS = (
    "generated_text",
    "postprocessed_text",
    "cleaned_text",
    "clean_generation",
    "completion",
    "output",
    "text",
)
RAW_TEXT_FIELDS = (
    "raw_generated_text",
    "raw_generation",
    "raw_text",
    "raw_output",
    "model_output",
)
PROMPT_FIELDS = (
    "prompt",
    "instruction",
    "input_prompt",
    "generation_prompt",
    "source_prompt",
    "request",
)

POSITIVE_BLOCK_TAGS = {
    "copied_source",
    "dialogue_drift",
    "dialogue_like",
    "empty_output",
    "hate_speech",
    "line_count_miss",
    "no_slur_fail",
    "non_ascii",
    "non_lyric",
    "policy_fail",
    "question_drift",
    "question_ending",
    "sexual_threat",
    "source_artifact",
    "too_short",
    "violent_derailment",
    "weak_ending",
}

REPAIRABLE_TAGS = {
    "artifact_cleanup",
    "clean_model_artifacts",
    "drop_dangling_final_line",
    "dangling_ending",
    "line_count_miss",
    "question_ending",
    "weak_ending",
}

SLUR_RE = re.compile(
    r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Annotated sweep JSONL. May be passed more than once.",
    )
    parser.add_argument("--keepers", default="", help="Optional keepers JSONL.")
    parser.add_argument("--fixable", default="", help="Optional fixable JSONL.")
    parser.add_argument("--rejects", default="", help="Optional rejects JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/packaged/qwen3_4b_sweep"))
    parser.add_argument("--positive-min-score", type=float, default=0.70)
    parser.add_argument("--manual-fixable-min-score", type=float, default=0.65)
    parser.add_argument("--max-positive-postprocess-actions", type=int, default=0)
    parser.add_argument("--max-dpo-pairs", type=int, default=5000)
    parser.add_argument("--max-dpo-pairs-per-prompt", type=int, default=4)
    parser.add_argument(
        "--allow-slur-positive",
        action="store_true",
        help="Allow positive SFT examples containing detected slur terms.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, *, decision_hint: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            payload["_source_file"] = str(path)
            if decision_hint and not any(
                key in payload for key in ("decision_label", "decision", "label")
            ):
                payload["_decision_hint"] = decision_hint
            rows.append(payload)
    return rows


def first_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for field in fields:
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        items = re.split(r"[,;|]", value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(normalize_tag(text))
    return out


def normalize_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_id(parts: list[str]) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = first_text(row, PROMPT_FIELDS)
    generated_text = first_text(row, TEXT_FIELDS)
    raw_text = first_text(row, RAW_TEXT_FIELDS) or generated_text
    decision = str(
        row.get("decision_label")
        or row.get("decision")
        or row.get("label")
        or row.get("_decision_hint")
        or ""
    ).strip().lower()
    score = as_float(row.get("score") or row.get("quality_score") or row.get("ranker_score"))
    failure_tags = as_list(row.get("failure_tags") or row.get("failures"))
    strength_tags = as_list(row.get("strength_tags") or row.get("strengths"))
    postprocess_actions = as_list(
        row.get("postprocess_actions")
        or row.get("postprocessing_actions")
        or row.get("cleanup_actions")
    )
    postprocess_applied = (
        as_bool(row.get("postprocess_applied"))
        or bool(postprocess_actions)
        or (bool(raw_text and generated_text) and raw_text.strip() != generated_text.strip())
    )
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    row_id = str(
        row.get("id")
        or row.get("row_id")
        or row.get("sample_id")
        or metadata.get("id")
        or stable_id([prompt, raw_text, str(index)])
    )
    prompt_key = str(
        row.get("prompt_id")
        or row.get("prompt_key")
        or metadata.get("prompt_id")
        or stable_id([prompt])
    )
    slur_present = as_bool(row.get("slur_present")) or bool(SLUR_RE.search(raw_text))
    hard_reject_slur = as_bool(row.get("hard_reject_slur")) or "no_slur_fail" in failure_tags
    return {
        "row_id": row_id,
        "prompt_key": prompt_key,
        "prompt": prompt,
        "raw_generated_text": raw_text,
        "generated_text": generated_text,
        "decision_label": decision,
        "score": score,
        "failure_tags": failure_tags,
        "strength_tags": strength_tags,
        "postprocess_actions": postprocess_actions,
        "postprocess_applied": postprocess_applied,
        "raw_line_count": row.get("raw_line_count"),
        "postprocessed_line_count": row.get("postprocessed_line_count"),
        "slur_present": slur_present,
        "hard_reject_slur": hard_reject_slur,
        "source_file": row.get("_source_file"),
        "metadata": metadata,
    }


def has_text(row: dict[str, Any]) -> bool:
    return bool(row["prompt"].strip() and row["generated_text"].strip())


def has_bad_output(row: dict[str, Any]) -> bool:
    return row["hard_reject_slur"] or bool(SLUR_RE.search(row["generated_text"]))


def positive_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["generated_text"]},
        ],
        "metadata": {
            "source": "qwen3_4b_generation_sweep",
            "row_id": row["row_id"],
            "prompt_key": row["prompt_key"],
            "decision_label": row["decision_label"],
            "score": row["score"],
            "failure_tags": row["failure_tags"],
            "strength_tags": row["strength_tags"],
            "postprocess_actions": row["postprocess_actions"],
        },
    }


def repair_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    user = (
        "Original prompt:\n"
        f"{row['prompt']}\n\n"
        "Raw model output:\n"
        f"{row['raw_generated_text']}\n\n"
        "Return only the repaired rap lyrics."
    )
    return {
        "messages": [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["generated_text"]},
        ],
        "metadata": {
            "source": "qwen3_4b_generation_sweep_repair_pair",
            "row_id": row["row_id"],
            "prompt_key": row["prompt_key"],
            "decision_label": row["decision_label"],
            "score": row["score"],
            "failure_tags": row["failure_tags"],
            "postprocess_actions": row["postprocess_actions"],
            "raw_line_count": row["raw_line_count"],
            "postprocessed_line_count": row["postprocessed_line_count"],
        },
    }


def dpo_pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    chosen_row: dict[str, Any],
    rejected_row: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "source": "qwen3_4b_generation_sweep",
            "reason": reason,
            "prompt_key": chosen_row["prompt_key"],
            "chosen_row_id": chosen_row["row_id"],
            "rejected_row_id": rejected_row["row_id"],
            "chosen_decision": chosen_row["decision_label"],
            "rejected_decision": rejected_row["decision_label"],
            "chosen_score": chosen_row["score"],
            "rejected_score": rejected_row["score"],
            "chosen_failure_tags": chosen_row["failure_tags"],
            "rejected_failure_tags": rejected_row["failure_tags"],
        },
    }


def build_positive_sft(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["decision_label"] != "keeper" or not has_text(row):
            continue
        if row["score"] is not None and row["score"] < args.positive_min_score:
            continue
        if len(row["postprocess_actions"]) > args.max_positive_postprocess_actions:
            continue
        if set(row["failure_tags"]) & POSITIVE_BLOCK_TAGS:
            continue
        if not args.allow_slur_positive and row["slur_present"]:
            continue
        out.append(positive_sft_row(row))
    return out


def build_repair_sft(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["decision_label"] not in {"keeper", "fixable"} or not has_text(row):
            continue
        if not row["postprocess_applied"]:
            continue
        if row["raw_generated_text"].strip() == row["generated_text"].strip():
            continue
        if has_bad_output(row):
            continue
        out.append(repair_sft_row(row))
    return out


def build_manual_queue(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["decision_label"] != "fixable" or not has_text(row):
            continue
        if row["score"] is not None and row["score"] < args.manual_fixable_min_score:
            continue
        if has_bad_output(row):
            continue
        repairable = not row["failure_tags"] or bool(set(row["failure_tags"]) & REPAIRABLE_TAGS)
        if not repairable:
            continue
        out.append(
            {
                "row_id": row["row_id"],
                "prompt_key": row["prompt_key"],
                "prompt": row["prompt"],
                "raw_generated_text": row["raw_generated_text"],
                "postprocessed_text": row["generated_text"],
                "score": row["score"],
                "failure_tags": row["failure_tags"],
                "postprocess_actions": row["postprocess_actions"],
                "repair_notes": "",
                "manual_repaired_text": "",
            }
        )
    return sorted(out, key=lambda item: item["score"] if item["score"] is not None else -1.0, reverse=True)


def build_dpo_pairs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(pair: dict[str, Any]) -> None:
        key = stable_id([pair["prompt"], pair["chosen"], pair["rejected"], pair["metadata"]["reason"]])
        if key in seen or pair["chosen"].strip() == pair["rejected"].strip():
            return
        seen.add(key)
        out.append(pair)

    for row in rows:
        if len(out) >= args.max_dpo_pairs:
            break
        if row["decision_label"] not in {"keeper", "fixable"} or not has_text(row):
            continue
        if not row["postprocess_applied"]:
            continue
        if row["raw_generated_text"].strip() == row["generated_text"].strip():
            continue
        if has_bad_output(row):
            continue
        add(
            dpo_pair(
                prompt=row["prompt"],
                chosen=row["generated_text"],
                rejected=row["raw_generated_text"],
                chosen_row=row,
                rejected_row=row,
                reason="postprocess_repair_preference",
            )
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_key"]].append(row)

    for group_rows in grouped.values():
        if len(out) >= args.max_dpo_pairs:
            break
        chosen_rows = [
            row
            for row in group_rows
            if row["decision_label"] == "keeper"
            and has_text(row)
            and not has_bad_output(row)
            and not (set(row["failure_tags"]) & POSITIVE_BLOCK_TAGS)
        ]
        rejected_rows = [
            row
            for row in group_rows
            if row["decision_label"] in {"reject", "rejected", "fixable"}
            and row["raw_generated_text"].strip()
            and (
                row["decision_label"] in {"reject", "rejected"}
                or row["hard_reject_slur"]
                or bool(set(row["failure_tags"]) & POSITIVE_BLOCK_TAGS)
            )
        ]
        chosen_rows.sort(key=lambda item: item["score"] if item["score"] is not None else 0.0, reverse=True)
        rejected_rows.sort(key=lambda item: item["score"] if item["score"] is not None else 0.0)
        pairs_for_prompt = 0
        for chosen_row in chosen_rows:
            for rejected_row in rejected_rows:
                if len(out) >= args.max_dpo_pairs or pairs_for_prompt >= args.max_dpo_pairs_per_prompt:
                    break
                add(
                    dpo_pair(
                        prompt=chosen_row["prompt"],
                        chosen=chosen_row["generated_text"],
                        rejected=rejected_row["raw_generated_text"],
                        chosen_row=chosen_row,
                        rejected_row=rejected_row,
                        reason="keeper_over_reject",
                    )
                )
                pairs_for_prompt += 1
            if pairs_for_prompt >= args.max_dpo_pairs_per_prompt:
                break
    return out[: args.max_dpo_pairs]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    positive: list[dict[str, Any]],
    repair: list[dict[str, Any]],
    manual: list[dict[str, Any]],
    dpo: list[dict[str, Any]],
) -> None:
    labels = Counter(row["decision_label"] or "unlabeled" for row in rows)
    failure_tags = Counter(tag for row in rows for tag in row["failure_tags"])
    actions = Counter(action for row in rows for action in row["postprocess_actions"])
    lines = [
        "# Qwen3-4B Sweep Dataset Packaging",
        "",
        "Local-only packaging run. No OpenAI or network calls are used.",
        "",
        "## Input",
        "",
        f"- Rows: {len(rows)}",
        f"- Decisions: {dict(sorted(labels.items()))}",
        f"- Slur-present rows: {sum(1 for row in rows if row['slur_present'])}",
        f"- Postprocessed rows: {sum(1 for row in rows if row['postprocess_applied'])}",
        "",
        "## Output",
        "",
        f"- Positive SFT examples: {len(positive)}",
        f"- Repair-pair SFT examples: {len(repair)}",
        f"- Manual repair queue rows: {len(manual)}",
        f"- DPO preference pairs: {len(dpo)}",
        "",
        "## Top Failure Tags",
        "",
    ]
    if failure_tags:
        lines.extend(f"- {tag}: {count}" for tag, count in failure_tags.most_common(20))
    else:
        lines.append("- none")
    lines.extend(["", "## Top Postprocess Actions", ""])
    if actions:
        lines.extend(f"- {action}: {count}" for action, count in actions.most_common(20))
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    sources: list[tuple[Path, str | None]] = []
    sources.extend((Path(path), None) for path in args.input)
    if args.keepers:
        sources.append((Path(args.keepers), "keeper"))
    if args.fixable:
        sources.append((Path(args.fixable), "fixable"))
    if args.rejects:
        sources.append((Path(args.rejects), "reject"))
    if not sources:
        raise SystemExit(
            "No sweep inputs supplied. Pass --input annotated_sweep.jsonl or "
            "--keepers/--fixable/--rejects JSONL files."
        )
    raw_rows: list[dict[str, Any]] = []
    for path, decision_hint in sources:
        raw_rows.extend(read_jsonl(path, decision_hint=decision_hint))
    return [normalize_row(row, index) for index, row in enumerate(raw_rows, start=1)]


def main() -> None:
    args = parse_args()
    rows = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positive = build_positive_sft(rows, args)
    repair = build_repair_sft(rows)
    manual = build_manual_queue(rows, args)
    dpo = build_dpo_pairs(rows, args)

    positive_path = args.output_dir / "positive_sft.jsonl"
    repair_path = args.output_dir / "repair_pairs_sft.jsonl"
    manual_path = args.output_dir / "manual_repair_queue.jsonl"
    dpo_path = args.output_dir / "dpo_pairs.jsonl"
    summary_path = args.output_dir / "packaging_summary.json"
    report_path = args.output_dir / "packaging_report.md"

    write_jsonl(positive_path, positive)
    write_jsonl(repair_path, repair)
    write_jsonl(manual_path, manual)
    write_jsonl(dpo_path, dpo)
    summary = {
        "base_model_scope": "Qwen/Qwen3-4B",
        "local_only": True,
        "input_rows": len(rows),
        "positive_sft_rows": len(positive),
        "repair_pair_sft_rows": len(repair),
        "manual_repair_rows": len(manual),
        "dpo_pairs": len(dpo),
        "outputs": {
            "positive_sft": str(positive_path),
            "repair_pairs_sft": str(repair_path),
            "manual_repair_queue": str(manual_path),
            "dpo_pairs": str(dpo_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(
        path=report_path,
        rows=rows,
        positive=positive,
        repair=repair,
        manual=manual,
        dpo=dpo,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
