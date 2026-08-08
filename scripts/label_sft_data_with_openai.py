"""Label rap SFT records with OpenAI and build a balanced training mix.

This is different from the binary bad-data reviewer:
- The reviewer asks "is this bad enough to remove?"
- This labeler asks "what behavior would this row teach?"

The output labels are meant for dataset composition: complete verses, hooks,
short fragments, mutation records, prose drift, weak endings, etc. Raw input
files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"

ARTIFACT_RE = re.compile(
    r"(?:download the full version|itunes|genius\.com|lyrics taken from|you might also like|embed|"
    r"https?://|www\.|Ã|Â|â€™|â€œ|â€|庭|旁|asarificing|fogggy)",
    re.IGNORECASE,
)
REPETITION_RE = re.compile(r"\b(\w{3,})\b(?:\W+\1\b){3,}", re.IGNORECASE)


LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "record_id": {"type": "string"},
                    "training_bucket": {
                        "type": "string",
                        "enum": [
                            "complete_verse",
                            "hook_or_chorus",
                            "short_bar_fragment",
                            "mutation_rewrite",
                            "repair_or_control",
                            "prose_or_dialogue",
                            "artifact_or_corrupt",
                            "unsafe_or_off_prompt",
                            "other",
                        ],
                    },
                    "keep_for_sft": {"type": "boolean"},
                    "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "creativity_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "control_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "line_structure": {
                        "type": "string",
                        "enum": ["strong", "acceptable", "weak", "paragraph", "fragment"],
                    },
                    "completion_shape": {
                        "type": "string",
                        "enum": ["complete", "usable_partial", "too_short", "rambling", "broken"],
                    },
                    "prompt_adherence": {
                        "type": "string",
                        "enum": ["strong", "acceptable", "weak", "off_prompt"],
                    },
                    "clean_ending": {"type": "boolean"},
                    "lyric_only": {"type": "boolean"},
                    "estimated_lines": {"type": "integer", "minimum": 0, "maximum": 128},
                    "failure_tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "encoding_corruption",
                                "metadata_or_web_artifact",
                                "mixed_language_garbage",
                                "dialogue_or_stage_chatter",
                                "giant_paragraph",
                                "too_short",
                                "bad_ending",
                                "off_prompt",
                                "unsafe_derailment",
                                "excessive_repetition",
                                "mutation_overreach",
                                "generic_or_boring",
                                "copied_artist_leak",
                                "none",
                            ],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "record_id",
                    "training_bucket",
                    "keep_for_sft",
                    "quality_score",
                    "creativity_score",
                    "control_score",
                    "line_structure",
                    "completion_shape",
                    "prompt_adherence",
                    "clean_ending",
                    "lyric_only",
                    "estimated_lines",
                    "failure_tags",
                    "notes",
                ],
            },
        }
    },
    "required": ["labels"],
}


@dataclass(frozen=True)
class DatasetRecord:
    index: int
    record_id: str
    record: dict[str, Any]
    prompt: str
    assistant: str
    source: str
    local_shape: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_record_id(index: int, record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for key in ["bar_id", "section_id", "song_id"]:
        value = metadata.get(key)
        if value:
            return f"{key}:{value}:line:{index}"
    digest = hashlib.sha1(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"line:{index}:sha1:{digest}"


def message_text(record: dict[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == role
    ).strip()


def local_shape(prompt: str, assistant: str) -> dict[str, Any]:
    lines = [line.strip() for line in assistant.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", assistant)
    flags: list[str] = []
    if ARTIFACT_RE.search(assistant):
        flags.append("artifact_or_encoding")
    if len(lines) <= 1 and len(words) > 28:
        flags.append("paragraph")
    if len(words) < 4:
        flags.append("too_short")
    if REPETITION_RE.search(assistant):
        flags.append("repetition")
    return {
        "line_count": len(lines),
        "word_count": len(words),
        "prompt_mentions_hook": "hook" in prompt.lower() or "chorus" in prompt.lower(),
        "prompt_mentions_verse": "verse" in prompt.lower(),
        "prompt_mentions_mutation": "rewrite" in prompt.lower() or "mutation" in prompt.lower(),
        "flags": flags,
    }


def extract_record(index: int, record: dict[str, Any]) -> DatasetRecord:
    prompt = message_text(record, "user")
    assistant = message_text(record, "assistant")
    source = str((record.get("metadata") or {}).get("sft_source") or "unknown")
    if not assistant and "output_bars" in record:
        assistant = "\n".join(str(item) for item in record.get("output_bars") or [])
        prompt = "\n".join(str(item) for item in record.get("input_bars") or [])
        source = "mutation"
    return DatasetRecord(
        index=index,
        record_id=stable_record_id(index, record),
        record=record,
        prompt=prompt,
        assistant=assistant,
        source=source,
        local_shape=local_shape(prompt, assistant),
    )


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload.get("record_id"), str):
                done.add(payload["record_id"])
    return done


def compact_payload(records: list[DatasetRecord], *, max_chars: int) -> str:
    payload = [
        {
            "record_id": item.record_id,
            "source": item.source,
            "local_shape": item.local_shape,
            "prompt": item.prompt[:max_chars],
            "assistant": item.assistant[:max_chars],
        }
        for item in records
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_label_request(records: list[DatasetRecord], *, model: str, max_chars: int) -> dict[str, Any]:
    instructions = (
        "You are labeling rap SFT training records for dataset composition. "
        "Judge what behavior each record teaches, not whether the language is polite. "
        "Profanity, rap slang, dark imagery, street vocabulary, and the word nigga are not automatic failures. "
        "Prefer records that teach clean lyric-only prompt completion: coherent lines, useful cadence, clear ending, "
        "on-topic response, and no scrape/artifact contamination. Penalize records that teach prose rambling, dialogue, "
        "metadata/web artifacts, bad mutations, off-prompt drift, or broken endings. "
        "For single-bar fragments, keep_for_sft can be true only if they are clean and useful as a deliberately capped bucket."
    )
    user = (
        "Label every record_id exactly once. Return structured labels only.\n\n"
        f"{compact_payload(records, max_chars=max_chars)}"
    )
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rap_sft_behavior_labels",
                "schema": LABEL_SCHEMA,
                "strict": True,
            }
        },
    }


def parse_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    return json.loads(text)
    raise ValueError("Could not find JSON text in OpenAI response")


def post_with_retries(payload: dict[str, Any], *, api_key: str, timeout: int, retries: int) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"retryable OpenAI API status {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60, 2**attempt + random.random()))
    raise RuntimeError(f"OpenAI API request failed after {retries + 1} attempts: {last_error}") from last_error


def select_records(records: list[DatasetRecord], args: argparse.Namespace) -> list[DatasetRecord]:
    selected = list(records)
    rng = random.Random(args.seed)
    if args.candidate_mode == "random":
        rng.shuffle(selected)
    elif args.candidate_mode == "train-subset":
        rng.shuffle(selected)
        selected = selected[: args.train_subset_size]
    elif args.candidate_mode == "shape-risk":
        selected = [
            item
            for item in selected
            if item.local_shape["flags"]
            or item.local_shape["line_count"] <= 2
            or item.source == "mutation"
        ]
        rng.shuffle(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def cmd_label(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is required. Add it to .env or the environment, or use --dry-run.")
    raw = read_jsonl(args.input)
    records = [extract_record(index, record) for index, record in enumerate(raw)]
    selected = select_records(records, args)
    done = load_done_ids(args.output_labels) if args.resume else set()
    todo = [item for item in selected if item.record_id not in done]
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "records_loaded": len(raw),
                    "records_selected": len(selected),
                    "records_remaining": len(todo),
                    "candidate_mode": args.candidate_mode,
                    "batch_size": args.batch_size,
                    "model": args.model,
                    "first_batch_payload": json.loads(compact_payload(todo[: args.batch_size], max_chars=args.max_chars)),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    with args.output_labels.open("a", encoding="utf-8") as handle:
        for batch_number, start in enumerate(range(0, len(todo), args.batch_size), start=1):
            batch = todo[start : start + args.batch_size]
            request_payload = build_label_request(batch, model=args.model, max_chars=args.max_chars)
            started = time.perf_counter()
            response_payload = post_with_retries(
                request_payload,
                api_key=api_key or "",
                timeout=args.timeout,
                retries=args.retries,
            )
            elapsed = round(time.perf_counter() - started, 3)
            parsed = parse_response_json(response_payload)
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise RuntimeError("Structured response did not contain labels array")
            by_id = {item.record_id: item for item in batch}
            written = 0
            for label in labels:
                item = by_id.get(label.get("record_id"))
                if item is None:
                    continue
                output = {
                    "record_id": item.record_id,
                    "line_index": item.index,
                    "source": item.source,
                    "local_shape": item.local_shape,
                    **label,
                    "model": args.model,
                    "label_elapsed_seconds": elapsed,
                    "response_id": response_payload.get("id"),
                    "usage": response_payload.get("usage"),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            print(
                json.dumps(
                    {
                        "batch": batch_number,
                        "written": written,
                        "remaining": max(0, len(todo) - start - args.batch_size),
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                )
            )


def read_labels(path: Path) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("line_index"), int):
                labels[row["line_index"]] = row
    return labels


def cmd_build(args: argparse.Namespace) -> None:
    raw = read_jsonl(args.input)
    labels = read_labels(args.labels)
    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []

    for index, record in enumerate(raw):
        label = labels.get(index)
        if not label:
            continue
        keep = bool(label.get("keep_for_sft"))
        quality = int(label.get("quality_score") or 0)
        control = int(label.get("control_score") or 0)
        shape = str(label.get("completion_shape") or "")
        tags = {str(tag) for tag in label.get("failure_tags", [])}
        line_structure = str(label.get("line_structure") or "")
        prompt_adherence = str(label.get("prompt_adherence") or "")
        clean_ending = bool(label.get("clean_ending"))
        lyric_only = bool(label.get("lyric_only"))
        if (
            not keep
            or quality < args.min_quality
            or control < args.min_control
            or shape in {"broken", "rambling"}
            or tags & set(args.disallow_failure_tag)
            or (args.require_clean_ending and not clean_ending)
            or (args.require_lyric_only and not lyric_only)
            or (args.require_strong_line_structure and line_structure != "strong")
            or (args.require_prompt_adherence and prompt_adherence not in {"strong", "acceptable"})
        ):
            rejected.append({"record": record, "label": label})
            continue
        bucket = str(label.get("training_bucket") or "other")
        enriched = dict(record)
        enriched.setdefault("metadata", {})["openai_label_bucket"] = bucket
        enriched["metadata"]["openai_quality_score"] = quality
        enriched["metadata"]["openai_control_score"] = control
        buckets[bucket].append(enriched)

    target_counts = {
        "complete_verse": args.max_complete_verse if args.max_complete_verse is not None else int(args.max_records * 0.45),
        "hook_or_chorus": args.max_hook_or_chorus if args.max_hook_or_chorus is not None else int(args.max_records * 0.15),
        "short_bar_fragment": args.max_short_bar_fragment if args.max_short_bar_fragment is not None else int(args.max_records * 0.15),
        "mutation_rewrite": args.max_mutation_rewrite if args.max_mutation_rewrite is not None else int(args.max_records * 0.15),
        "repair_or_control": args.max_repair_or_control if args.max_repair_or_control is not None else int(args.max_records * 0.05),
        "other": args.max_other if args.max_other is not None else int(args.max_records * 0.05),
    }
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for bucket, limit in target_counts.items():
        rows = list(buckets.get(bucket, []))
        rng.shuffle(rows)
        take = rows[:limit]
        selected.extend(take)
        selected_counts[bucket] += len(take)

    if len(selected) < args.max_records and args.allow_fallback_fill:
        fallback: list[tuple[str, dict[str, Any]]] = []
        for bucket, rows in buckets.items():
            already = selected_counts[bucket]
            extras = list(rows)
            rng.shuffle(extras)
            for row in extras[already:]:
                fallback.append((bucket, row))
        rng.shuffle(fallback)
        for bucket, row in fallback:
            if len(selected) >= args.max_records:
                break
            selected.append(row)
            selected_counts[bucket] += 1

    rng.shuffle(selected)
    validation_count = min(args.validation_records, max(0, len(selected) // 5))
    validation = selected[:validation_count]
    train = selected[validation_count:]
    for split_name, rows in [("validation", validation), ("train", train)]:
        for row in rows:
            row.setdefault("metadata", {})["split"] = split_name

    write_jsonl(args.output_train, train)
    write_jsonl(args.output_validation, validation)
    summary = {
        "input": str(args.input),
        "labels": str(args.labels),
        "outputs": {
            "train": str(args.output_train),
            "validation": str(args.output_validation),
            "summary": str(args.summary_output),
        },
        "seed": args.seed,
        "criteria": {
            "max_records": args.max_records,
            "validation_records": validation_count,
            "min_quality": args.min_quality,
            "min_control": args.min_control,
            "allow_fallback_fill": args.allow_fallback_fill,
            "require_clean_ending": args.require_clean_ending,
            "require_lyric_only": args.require_lyric_only,
            "require_strong_line_structure": args.require_strong_line_structure,
            "require_prompt_adherence": args.require_prompt_adherence,
            "disallow_failure_tag": args.disallow_failure_tag,
            "target_counts": target_counts,
        },
        "records": {
            "input": len(raw),
            "labeled": len(labels),
            "eligible": sum(len(rows) for rows in buckets.values()),
            "rejected_labeled": len(rejected),
            "selected": len(selected),
            "train": len(train),
            "validation": len(validation),
        },
        "selected_buckets": dict(selected_counts),
        "available_buckets": {bucket: len(rows) for bucket, rows in sorted(buckets.items())},
        "label_buckets": dict(Counter(str(label.get("training_bucket")) for label in labels.values())),
        "failure_tags": dict(Counter(tag for label in labels.values() for tag in label.get("failure_tags", []))),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser("label", help="Label records with OpenAI for dataset composition.")
    label.add_argument("--input", type=Path, default=Path("data/sft/rap_mixed_sft_openai_filtered_train.jsonl"))
    label.add_argument("--output-labels", type=Path, default=Path("data/labels/rap_mixed_sft_behavior_labels.jsonl"))
    label.add_argument("--model", default=os.environ.get("OPENAI_LABEL_MODEL", DEFAULT_MODEL))
    label.add_argument("--candidate-mode", choices=["all", "random", "train-subset", "shape-risk"], default="train-subset")
    label.add_argument("--train-subset-size", type=int, default=15000)
    label.add_argument("--limit", type=int, default=None)
    label.add_argument("--batch-size", type=int, default=16)
    label.add_argument("--max-chars", type=int, default=1000)
    label.add_argument("--seed", type=int, default=20260626)
    label.add_argument("--timeout", type=int, default=180)
    label.add_argument("--retries", type=int, default=4)
    label.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    label.add_argument("--dry-run", action="store_true")
    label.set_defaults(func=cmd_label)

    build = subparsers.add_parser("build", help="Build a balanced SFT dataset from behavior labels.")
    build.add_argument("--input", type=Path, default=Path("data/sft/rap_mixed_sft_openai_filtered_train.jsonl"))
    build.add_argument("--labels", type=Path, default=Path("data/labels/rap_mixed_sft_behavior_labels.jsonl"))
    build.add_argument("--output-train", type=Path, default=Path("data/sft/rap_mixed_sft_ai_labeled_balanced_train.jsonl"))
    build.add_argument("--output-validation", type=Path, default=Path("data/sft/rap_mixed_sft_ai_labeled_balanced_validation.jsonl"))
    build.add_argument("--summary-output", type=Path, default=Path("data/labels/rap_mixed_sft_ai_labeled_balanced_summary.json"))
    build.add_argument("--max-records", type=int, default=12000)
    build.add_argument("--validation-records", type=int, default=500)
    build.add_argument("--min-quality", type=int, default=3)
    build.add_argument("--min-control", type=int, default=3)
    build.add_argument("--allow-fallback-fill", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--require-clean-ending", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-lyric-only", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-strong-line-structure", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-prompt-adherence", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--disallow-failure-tag", action="append", default=[])
    build.add_argument("--max-complete-verse", type=int, default=None)
    build.add_argument("--max-hook-or-chorus", type=int, default=None)
    build.add_argument("--max-short-bar-fragment", type=int, default=None)
    build.add_argument("--max-mutation-rewrite", type=int, default=None)
    build.add_argument("--max-repair-or-control", type=int, default=None)
    build.add_argument("--max-other", type=int, default=None)
    build.add_argument("--seed", type=int, default=20260626)
    build.set_defaults(func=cmd_build)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
