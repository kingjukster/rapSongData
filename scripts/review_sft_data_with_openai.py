"""Review SFT JSONL records with OpenAI and filter bad training data.

The script is intentionally conservative:
- Raw input files are never modified.
- Reviews are written as JSONL and can be resumed.
- Filtering is a separate step driven by saved review decisions.
- Slurs and rap slang are not automatic drop reasons; the review focuses on
  artifact leakage, encoding damage, prompt mismatch, structure failures, and
  corpus-contamination patterns that hurt SFT.
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
PROSE_RE = re.compile(r"[.!?][^\n]{120,}[.!?]")
REPETITION_RE = re.compile(r"\b(\w{3,})\b(?:\W+\1\b){3,}", re.IGNORECASE)


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "record_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["keep", "drop", "borderline"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "severity": {"type": "string", "enum": ["none", "low", "medium", "high"]},
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "good",
                                "encoding_corruption",
                                "metadata_artifact",
                                "download_or_web_artifact",
                                "mixed_language_drift",
                                "prose_or_dialogue_drift",
                                "incomplete_or_too_short",
                                "giant_paragraph",
                                "excessive_repetition",
                                "off_prompt",
                                "unsafe_derailment",
                                "copied_or_specific_artist_leak",
                                "bad_mutation",
                                "other_bad_data",
                            ],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["record_id", "decision", "confidence", "severity", "categories", "notes"],
            },
        }
    },
    "required": ["reviews"],
}


@dataclass(frozen=True)
class DatasetRecord:
    index: int
    record_id: str
    record: dict[str, Any]
    prompt: str
    assistant: str
    source: str
    heuristic_flags: list[str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    chunks = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == role
    ]
    return "\n".join(chunks).strip()


def extract_record(index: int, record: dict[str, Any]) -> DatasetRecord:
    prompt = message_text(record, "user")
    assistant = message_text(record, "assistant")
    source = str((record.get("metadata") or {}).get("sft_source") or "unknown")
    if not assistant and "output_bars" in record:
        assistant = "\n".join(str(item) for item in record.get("output_bars") or [])
        prompt = "\n".join(str(item) for item in record.get("input_bars") or [])
        source = "mutation"
    flags = heuristic_flags(prompt=prompt, assistant=assistant)
    return DatasetRecord(
        index=index,
        record_id=stable_record_id(index, record),
        record=record,
        prompt=prompt,
        assistant=assistant,
        source=source,
        heuristic_flags=flags,
    )


def heuristic_flags(*, prompt: str, assistant: str) -> list[str]:
    flags: list[str] = []
    text = assistant.strip()
    if ARTIFACT_RE.search(text):
        flags.append("artifact_or_encoding")
    if len([line for line in text.splitlines() if line.strip()]) <= 1 and len(text.split()) > 28:
        flags.append("paragraph_not_lyrics")
    if PROSE_RE.search(text):
        flags.append("long_prose_sentence")
    if REPETITION_RE.search(text):
        flags.append("repetition")
    if len(text.split()) < 4:
        flags.append("too_short")
    if "target output bars" in prompt.lower():
        match = re.search(r"target output bars:\s*(\d+)", prompt, flags=re.IGNORECASE)
        if match:
            requested = int(match.group(1))
            lines = len([line for line in text.splitlines() if line.strip()])
            if abs(lines - requested) > max(2, requested // 2):
                flags.append("mutation_line_count_mismatch")
    return flags


def batched(items: list[DatasetRecord], size: int) -> list[list[DatasetRecord]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_reviewed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    reviewed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record_id = payload.get("record_id")
            if isinstance(record_id, str):
                reviewed.add(record_id)
    return reviewed


def compact_payload(records: list[DatasetRecord], *, max_chars: int) -> str:
    payload = []
    for item in records:
        payload.append(
            {
                "record_id": item.record_id,
                "source": item.source,
                "heuristic_flags": item.heuristic_flags,
                "prompt": item.prompt[:max_chars],
                "assistant": item.assistant[:max_chars],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_request(records: list[DatasetRecord], *, model: str, max_chars: int) -> dict[str, Any]:
    instructions = (
        "You are reviewing rap SFT training records for data quality. "
        "Mark records as drop only when they would teach bad model behavior: encoding corruption, web/download/metadata "
        "artifacts, mixed-language garbage, prose/dialogue drift, incomplete fragments, giant paragraphs, excessive "
        "repetition, off-prompt content, unsafe derailment unrelated to prompt, copied-artist leakage, or bad mutation "
        "outputs. Do not drop solely for profanity, rap slang, dark imagery, street vocabulary, or the word nigga. "
        "Prefer borderline when the record has some useful lyric signal but visible flaws."
    )
    user = (
        "Review these JSON records. Return one review for every record_id, preserving record_id exactly.\n\n"
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
                "name": "rap_sft_quality_reviews",
                "schema": REVIEW_SCHEMA,
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


def post_with_retries(request_payload: dict[str, Any], *, api_key: str, timeout: int, retries: int) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(RESPONSES_URL, headers=headers, json=request_payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"retryable OpenAI API status {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retain context in JSONL-friendly logs.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60, 2**attempt + random.random()))
    raise RuntimeError(f"OpenAI API request failed after {retries + 1} attempts: {last_error}") from last_error


def cmd_review(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is required. Add it to .env or the environment, or use --dry-run.")

    raw_records = read_jsonl(args.input)
    extracted = [extract_record(index, record) for index, record in enumerate(raw_records)]
    if args.candidate_mode == "heuristic":
        extracted = [item for item in extracted if item.heuristic_flags]
    elif args.candidate_mode == "random":
        rng = random.Random(args.seed)
        rng.shuffle(extracted)

    if args.limit is not None:
        extracted = extracted[: args.limit]

    reviewed_ids = load_reviewed_ids(args.output_reviews) if args.resume else set()
    todo = [item for item in extracted if item.record_id not in reviewed_ids]
    args.output_reviews.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview = {
            "input": str(args.input),
            "candidate_mode": args.candidate_mode,
            "records_loaded": len(raw_records),
            "records_selected": len(extracted),
            "records_remaining": len(todo),
            "batch_size": args.batch_size,
            "model": args.model,
            "first_batch_payload": json.loads(compact_payload(todo[: args.batch_size], max_chars=args.max_chars)),
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return

    with args.output_reviews.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batched(todo, args.batch_size), start=1):
            request_payload = build_request(batch, model=args.model, max_chars=args.max_chars)
            started = time.perf_counter()
            response_payload = post_with_retries(
                request_payload,
                api_key=api_key or "",
                timeout=args.timeout,
                retries=args.retries,
            )
            elapsed = round(time.perf_counter() - started, 3)
            parsed = parse_response_json(response_payload)
            reviews = parsed.get("reviews")
            if not isinstance(reviews, list):
                raise RuntimeError("Structured response did not contain reviews array")
            batch_by_id = {item.record_id: item for item in batch}
            for review in reviews:
                record_id = review.get("record_id")
                source_record = batch_by_id.get(record_id)
                if source_record is None:
                    continue
                output = {
                    "record_id": record_id,
                    "line_index": source_record.index,
                    "source": source_record.source,
                    "heuristic_flags": source_record.heuristic_flags,
                    "decision": review.get("decision"),
                    "confidence": review.get("confidence"),
                    "severity": review.get("severity"),
                    "categories": review.get("categories"),
                    "notes": review.get("notes"),
                    "model": args.model,
                    "review_elapsed_seconds": elapsed,
                    "response_id": response_payload.get("id"),
                    "usage": response_payload.get("usage"),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "reviewed": len(reviews),
                        "remaining": max(0, len(todo) - batch_index * args.batch_size),
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                )
            )


def read_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record_id = payload.get("record_id")
            if isinstance(record_id, str):
                reviews[record_id] = payload
    return reviews


def cmd_apply(args: argparse.Namespace) -> None:
    raw_records = read_jsonl(args.input)
    reviews = read_reviews(args.reviews)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for index, record in enumerate(raw_records):
        item = extract_record(index, record)
        review = reviews.get(item.record_id)
        decision = str(review.get("decision") if review else "unreviewed")
        counts[decision] = counts.get(decision, 0) + 1
        for category in review.get("categories", []) if review else []:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
        should_drop = decision == "drop" or (args.drop_borderline and decision == "borderline")
        if should_drop:
            enriched = dict(record)
            enriched["_openai_review"] = review
            dropped.append(enriched)
        else:
            kept.append(record)

    write_jsonl(args.output_kept, kept)
    if args.output_dropped:
        write_jsonl(args.output_dropped, dropped)
    summary = {
        "input": str(args.input),
        "reviews": str(args.reviews),
        "output_kept": str(args.output_kept),
        "output_dropped": str(args.output_dropped) if args.output_dropped else None,
        "drop_borderline": bool(args.drop_borderline),
        "records": {
            "input": len(raw_records),
            "reviewed": len(reviews),
            "kept": len(kept),
            "dropped": len(dropped),
        },
        "decision_counts": counts,
        "category_counts": dict(sorted(category_counts.items())),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("review", help="Call OpenAI in batches and save per-record reviews.")
    review.add_argument("--input", type=Path, default=Path("data/sft/rap_mixed_sft_train.jsonl"))
    review.add_argument("--output-reviews", type=Path, default=Path("data/reviews/rap_mixed_sft_openai_reviews.jsonl"))
    review.add_argument("--model", default=os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_MODEL))
    review.add_argument("--batch-size", type=int, default=8)
    review.add_argument("--max-chars", type=int, default=1400)
    review.add_argument("--limit", type=int, default=None)
    review.add_argument("--candidate-mode", choices=["all", "heuristic", "random"], default="heuristic")
    review.add_argument("--seed", type=int, default=20260625)
    review.add_argument("--timeout", type=int, default=120)
    review.add_argument("--retries", type=int, default=4)
    review.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    review.add_argument("--dry-run", action="store_true")
    review.set_defaults(func=cmd_review)

    apply = subparsers.add_parser("apply", help="Filter a JSONL file using saved OpenAI reviews.")
    apply.add_argument("--input", type=Path, default=Path("data/sft/rap_mixed_sft_train.jsonl"))
    apply.add_argument("--reviews", type=Path, default=Path("data/reviews/rap_mixed_sft_openai_reviews.jsonl"))
    apply.add_argument("--output-kept", type=Path, default=Path("data/sft/rap_mixed_sft_openai_filtered_train.jsonl"))
    apply.add_argument("--output-dropped", type=Path, default=Path("data/reviews/rap_mixed_sft_openai_dropped.jsonl"))
    apply.add_argument("--summary-output", type=Path, default=Path("data/reviews/rap_mixed_sft_openai_filter_summary.json"))
    apply.add_argument("--drop-borderline", action="store_true")
    apply.set_defaults(func=cmd_apply)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
