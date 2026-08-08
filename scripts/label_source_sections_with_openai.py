"""OpenAI-label source rap sections and build instruction SFT data.

The input is produced by ``build_source_section_candidates.py`` from the raw
song corpus. This keeps OpenAI review close to the original songs instead of
only reviewing already-derived SFT mixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"

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
                    "keep_for_sft": {"type": "boolean"},
                    "section_role": {"type": "string", "enum": ["verse", "hook", "bridge", "fragment", "drop"]},
                    "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "creativity_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "control_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "line_structure": {"type": "string", "enum": ["strong", "acceptable", "weak", "paragraph"]},
                    "completion_shape": {"type": "string", "enum": ["complete", "usable_partial", "too_short", "rambling", "broken"]},
                    "clean_ending": {"type": "boolean"},
                    "lyric_only": {"type": "boolean"},
                    "promptable_theme": {"type": "string"},
                    "failure_tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "metadata_or_web_artifact",
                                "encoding_corruption",
                                "dialogue_or_stage_chatter",
                                "giant_paragraph",
                                "too_short",
                                "bad_ending",
                                "excessive_repetition",
                                "generic_or_boring",
                                "copied_artist_leak",
                                "unsafe_derailment",
                                "none",
                            ],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "record_id",
                    "keep_for_sft",
                    "section_role",
                    "quality_score",
                    "creativity_score",
                    "control_score",
                    "line_structure",
                    "completion_shape",
                    "clean_ending",
                    "lyric_only",
                    "promptable_theme",
                    "failure_tags",
                    "notes",
                ],
            },
        }
    },
    "required": ["labels"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("record_id"), str):
                done.add(row["record_id"])
    return done


def compact_payload(records: list[dict[str, Any]], *, max_chars: int) -> str:
    payload = []
    for row in records:
        payload.append(
            {
                "record_id": row["record_id"],
                "section_kind": row.get("section_kind"),
                "section_label": row.get("section_label"),
                "title": row.get("title"),
                "artist": row.get("artist"),
                "year": row.get("year"),
                "shape": row.get("shape"),
                "text": str(row.get("text") or "")[:max_chars],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def request_payload(records: list[dict[str, Any]], *, model: str, max_chars: int) -> dict[str, Any]:
    instructions = (
        "You are labeling rap lyric sections extracted from raw songs for local SFT training. "
        "Do not reject solely for profanity, rap slang, dark imagery, street vocabulary, or the word nigga. "
        "Keep vivid authentic rap sections. Reject or down-score sections that teach bad assistant behavior: "
        "metadata, dialogue/stage chatter, web artifacts, giant prose paragraphs, severe repetition, copied artist/source leakage, "
        "broken endings, or sections too fragmentary to answer an instruction prompt. "
        "A good kept section should be lyric-only, complete-looking, line-broken, promptable, and useful as a model answer."
    )
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "Label every source section below, preserving record_id exactly.\n\n"
                + compact_payload(records, max_chars=max_chars),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rap_source_section_labels",
                "schema": LABEL_SCHEMA,
                "strict": True,
            }
        },
    }


def parse_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return json.loads(content["text"])
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
            time.sleep(min(90, 2**attempt + random.random()))
    raise RuntimeError(f"OpenAI API request failed after {retries + 1} attempts: {last_error}") from last_error


def cmd_label(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    done = load_done_ids(args.output_labels) if args.resume else set()
    todo = [row for row in rows if row["record_id"] not in done]
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "loaded": len(rows),
                    "remaining": len(todo),
                    "first_batch": json.loads(compact_payload(todo[: args.batch_size], max_chars=args.max_chars)),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    with args.output_labels.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            started = time.perf_counter()
            response = post_with_retries(
                request_payload(batch, model=args.model, max_chars=args.max_chars),
                api_key=api_key,
                timeout=args.timeout,
                retries=args.retries,
            )
            elapsed = round(time.perf_counter() - started, 3)
            parsed = parse_response_json(response)
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise RuntimeError("Structured response did not contain labels array")
            by_id = {row["record_id"]: row for row in batch}
            written = 0
            for label in labels:
                source = by_id.get(label.get("record_id"))
                if source is None:
                    continue
                output = {**source, **label, "model": args.model, "label_elapsed_seconds": elapsed, "response_id": response.get("id"), "usage": response.get("usage")}
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            print(
                json.dumps(
                    {
                        "batch": start // args.batch_size + 1,
                        "written": written,
                        "remaining": max(0, len(todo) - start - len(batch)),
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                )
            )


def prompt_for(row: dict[str, Any]) -> str:
    role = row.get("section_role") or row.get("section_kind")
    line_count = int((row.get("shape") or {}).get("line_count") or 0)
    theme = str(row.get("promptable_theme") or "").strip().rstrip(".")
    if role == "hook":
        count = min(max(line_count, 4), 8)
        return f"Write exactly {count} lines of a catchy rap hook about {theme or 'pressure and loyalty'}. Return only lyrics, one line per bar."
    count = min(max(line_count, 8), 20)
    return f"Write exactly {count} lines of a rap verse about {theme or 'ambition, pressure, and survival'}. Return only lyrics, one bar per line."


def cmd_build(args: argparse.Namespace) -> None:
    labels = read_jsonl(args.labels)
    disallowed = set(args.disallow_failure_tag)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for row in labels:
        if not row.get("keep_for_sft"):
            rejected.append(row)
            continue
        if int(row.get("quality_score") or 0) < args.min_quality or int(row.get("control_score") or 0) < args.min_control:
            rejected.append(row)
            continue
        if row.get("completion_shape") not in {"complete", "usable_partial"}:
            rejected.append(row)
            continue
        if args.require_clean_ending and not row.get("clean_ending"):
            rejected.append(row)
            continue
        if args.require_lyric_only and not row.get("lyric_only"):
            rejected.append(row)
            continue
        if set(row.get("failure_tags", [])) & disallowed:
            rejected.append(row)
            continue
        buckets[str(row.get("section_role") or row.get("section_kind") or "other")].append(row)

    caps = {"verse": args.max_verse, "hook": args.max_hook, "bridge": args.max_bridge, "fragment": args.max_fragment}
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for bucket, cap in caps.items():
        rows = list(buckets.get(bucket, []))
        rng.shuffle(rows)
        take = rows[:cap]
        selected.extend(take)
        selected_counts[bucket] += len(take)
    rng.shuffle(selected)
    if args.max_records is not None:
        selected = selected[: args.max_records]

    records = []
    for row in selected:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        records.append(
            {
                "messages": [
                    {"role": "system", "content": "Write only original rap lyrics. Keep line breaks. Do not explain."},
                    {"role": "user", "content": prompt_for(row)},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {
                    "sft_source": "raw_song_source_openai_labeled_section",
                    "record_id": row.get("record_id"),
                    "song_id": row.get("song_id"),
                    "title": row.get("title"),
                    "artist": row.get("artist"),
                    "year": row.get("year"),
                    "section_role": row.get("section_role"),
                    "quality_score": row.get("quality_score"),
                    "control_score": row.get("control_score"),
                    "creativity_score": row.get("creativity_score"),
                },
            }
        )

    val_count = min(args.validation_records, max(0, len(records) // 10))
    validation = records[:val_count]
    train = records[val_count:]
    write_jsonl(args.output_train, train)
    write_jsonl(args.output_validation, validation)
    summary = {
        "labels": str(args.labels),
        "records": {
            "labeled": len(labels),
            "eligible": sum(len(rows) for rows in buckets.values()),
            "rejected": len(rejected),
            "selected": len(records),
            "train": len(train),
            "validation": len(validation),
        },
        "selected_roles": dict(Counter(row["metadata"]["section_role"] for row in records)),
        "available_roles": {bucket: len(rows) for bucket, rows in sorted(buckets.items())},
        "criteria": vars(args),
        "outputs": {"train": str(args.output_train), "validation": str(args.output_validation), "summary": str(args.summary_output)},
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    label = subparsers.add_parser("label")
    label.add_argument("--input", type=Path, default=Path("data/source_sections/raw_rap_source_section_candidates.jsonl"))
    label.add_argument("--output-labels", type=Path, default=Path("data/labels/raw_source_sections_openai_labels.jsonl"))
    label.add_argument("--model", default=os.environ.get("OPENAI_SOURCE_LABEL_MODEL", DEFAULT_MODEL))
    label.add_argument("--limit", type=int, default=None)
    label.add_argument("--batch-size", type=int, default=12)
    label.add_argument("--max-chars", type=int, default=1800)
    label.add_argument("--timeout", type=int, default=180)
    label.add_argument("--retries", type=int, default=4)
    label.add_argument("--sleep-seconds", type=float, default=0.0)
    label.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    label.add_argument("--dry-run", action="store_true")
    label.set_defaults(func=cmd_label)

    build = subparsers.add_parser("build")
    build.add_argument("--labels", type=Path, default=Path("data/labels/raw_source_sections_openai_labels.jsonl"))
    build.add_argument("--output-train", type=Path, default=Path("data/sft/rap_source_sections_openai_sft_train.jsonl"))
    build.add_argument("--output-validation", type=Path, default=Path("data/sft/rap_source_sections_openai_sft_validation.jsonl"))
    build.add_argument("--summary-output", type=Path, default=Path("data/labels/rap_source_sections_openai_sft_summary.json"))
    build.add_argument("--min-quality", type=int, default=4)
    build.add_argument("--min-control", type=int, default=3)
    build.add_argument("--require-clean-ending", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--require-lyric-only", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--disallow-failure-tag", action="append", default=[])
    build.add_argument("--max-verse", type=int, default=10000)
    build.add_argument("--max-hook", type=int, default=6000)
    build.add_argument("--max-bridge", type=int, default=500)
    build.add_argument("--max-fragment", type=int, default=0)
    build.add_argument("--max-records", type=int, default=None)
    build.add_argument("--validation-records", type=int, default=500)
    build.add_argument("--seed", type=int, default=20260628)
    build.set_defaults(func=cmd_build)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
