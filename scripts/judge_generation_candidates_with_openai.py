"""Judge generated rap candidates with OpenAI and produce a curated shortlist.

This script is for generated outputs, not training rows. It uses a local
structural rank as a prefilter, then asks OpenAI to score the shortlist for
creative usefulness, control, prompt fit, and editability.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["keeper", "fixable", "reject"],
                    },
                    "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "creative_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "control_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "prompt_fit_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "line_structure_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "ending_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "strong_voice",
                                "vivid_imagery",
                                "good_hook",
                                "good_cadence",
                                "good_rhyme_potential",
                                "emotionally_coherent",
                                "authentic_rap_texture",
                                "needs_line_breaks",
                                "too_short",
                                "too_long",
                                "unfinished",
                                "prose_drift",
                                "off_prompt",
                                "generic",
                                "awkward_phrasing",
                                "dialogue_drift",
                                "unsafe_derailment",
                                "artist_name_leak",
                                "keep_as_seed",
                                "none",
                            ],
                        },
                    },
                    "best_lines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 4,
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "decision",
                    "overall_score",
                    "creative_score",
                    "control_score",
                    "prompt_fit_score",
                    "line_structure_score",
                    "ending_score",
                    "tags",
                    "best_lines",
                    "notes",
                ],
            },
        }
    },
    "required": ["judgments"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def candidate_id(record: dict[str, Any]) -> str:
    prompt_index = record.get("prompt_index", record.get("index", "x"))
    sample_index = record.get("sample_index", record.get("index", "x"))
    return f"p{prompt_index}_s{sample_index}"


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for row in read_jsonl(path):
        value = row.get("candidate_id")
        if isinstance(value, str):
            seen.add(value)
    return seen


def shortlist(records: list[dict[str, Any]], *, top_per_prompt: int, limit: int | None) -> list[dict[str, Any]]:
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_prompt[int(record.get("prompt_index") or 0)].append(record)
    selected: list[dict[str, Any]] = []
    for prompt_index in sorted(by_prompt):
        bucket = sorted(
            by_prompt[prompt_index],
            key=lambda item: float(item.get("candidate_score") or 0),
            reverse=True,
        )
        selected.extend(bucket[:top_per_prompt])
    selected.sort(key=lambda item: (int(item.get("prompt_index") or 0), -float(item.get("candidate_score") or 0)))
    if limit is not None:
        selected = selected[:limit]
    return selected


def compact_payload(records: list[dict[str, Any]], *, max_chars: int) -> str:
    payload = []
    for record in records:
        text = str(record.get("generated_text") or "")
        payload.append(
            {
                "candidate_id": candidate_id(record),
                "prompt_index": record.get("prompt_index"),
                "sample_index": record.get("sample_index"),
                "local_score": record.get("candidate_score"),
                "local_flags": record.get("candidate_flags"),
                "analysis": record.get("analysis"),
                "prompt": str(record.get("prompt") or "")[:max_chars],
                "generated_text": text[:max_chars],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_request(records: list[dict[str, Any]], *, model: str, max_chars: int) -> dict[str, Any]:
    instructions = (
        "You are judging generated rap lyrics for creative selection and dataset improvement. "
        "Prefer outputs that are usable or promising as rap lyrics: vivid, coherent, line-broken, emotionally or stylistically "
        "interesting, and aligned with the prompt. Do not reject solely for profanity, dark imagery, rap slang, or the word "
        "nigga. Reject or mark fixable for unfinished endings, prose drift, awkward non-rap phrasing, off-prompt content, "
        "generic motivational text, dialogue drift, artist-name leakage, or unsafe derailment that overwhelms the prompt. "
        "Use 'keeper' only when the output is directly useful or very close. Use 'fixable' when it has strong material but "
        "needs editing. Include 0-4 best lines worth preserving."
    )
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    "Judge these candidates. Return one judgment for every candidate_id, preserving candidate_id exactly.\n\n"
                    f"{compact_payload(records, max_chars=max_chars)}"
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rap_generation_candidate_judgments",
                "schema": JUDGE_SCHEMA,
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60, 2**attempt + random.random()))
    raise RuntimeError(f"OpenAI API request failed after {retries + 1} attempts: {last_error}") from last_error


def cmd_judge(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    records = shortlist(read_jsonl(args.input_jsonl), top_per_prompt=args.top_per_prompt, limit=args.limit)
    for record in records:
        record["candidate_id"] = candidate_id(record)
    reviewed_ids = read_existing_ids(args.output_judgments) if args.resume else set()
    todo = [record for record in records if record["candidate_id"] not in reviewed_ids]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input_jsonl),
                    "selected": len(records),
                    "remaining": len(todo),
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "first_batch": json.loads(compact_payload(todo[: args.batch_size], max_chars=args.max_chars)),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required. Add it to .env or the environment, or use --dry-run.")

    args.output_judgments.parent.mkdir(parents=True, exist_ok=True)
    by_id = {record["candidate_id"]: record for record in records}
    with args.output_judgments.open("a", encoding="utf-8") as handle:
        for batch_index in range(0, len(todo), args.batch_size):
            batch = todo[batch_index : batch_index + args.batch_size]
            started = time.perf_counter()
            response_payload = post_with_retries(
                build_request(batch, model=args.model, max_chars=args.max_chars),
                api_key=api_key,
                timeout=args.timeout,
                retries=args.retries,
            )
            elapsed = round(time.perf_counter() - started, 3)
            parsed = parse_response_json(response_payload)
            judgments = parsed.get("judgments")
            if not isinstance(judgments, list):
                raise RuntimeError("Structured response did not contain judgments array")
            for judgment in judgments:
                source = by_id.get(judgment.get("candidate_id"))
                if not source:
                    continue
                output = {
                    **judgment,
                    "prompt_index": source.get("prompt_index"),
                    "sample_index": source.get("sample_index"),
                    "prompt": source.get("prompt"),
                    "generated_text": source.get("generated_text"),
                    "analysis": source.get("analysis"),
                    "local_score": source.get("candidate_score"),
                    "local_flags": source.get("candidate_flags"),
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
                        "batch": batch_index // args.batch_size + 1,
                        "judged": len(judgments),
                        "remaining": max(0, len(todo) - batch_index - args.batch_size),
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                )
            )


def cmd_report(args: argparse.Namespace) -> None:
    judgments = read_jsonl(args.judgments)
    judgments.sort(
        key=lambda item: (
            int(item.get("prompt_index") or 0),
            -int(item.get("overall_score") or 0),
            -int(item.get("creative_score") or 0),
            -int(item.get("control_score") or 0),
        )
    )
    decision_counts = Counter(str(item.get("decision")) for item in judgments)
    tag_counts = Counter(tag for item in judgments for tag in item.get("tags", []) if tag != "none")
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in judgments:
        by_prompt[int(item.get("prompt_index") or 0)].append(item)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, judgments)

    md = ["# OpenAI-Judged Generation Candidates", ""]
    md.append(f"- Judgments: `{len(judgments)}`")
    md.append(f"- Decisions: `{dict(decision_counts)}`")
    md.append(f"- Top tags: `{dict(tag_counts.most_common(15))}`")
    md.append("")
    for prompt_index in sorted(by_prompt):
        rows = by_prompt[prompt_index]
        md.extend([f"## Prompt {prompt_index}", "", str(rows[0].get("prompt") or ""), ""])
        keepers = [row for row in rows if row.get("decision") == "keeper"]
        fixable = [row for row in rows if row.get("decision") == "fixable"]
        display = (keepers + fixable + rows)[: args.top_per_prompt]
        seen = set()
        unique_display = []
        for row in display:
            cid = row.get("candidate_id")
            if cid in seen:
                continue
            seen.add(cid)
            unique_display.append(row)
            if len(unique_display) >= args.top_per_prompt:
                break
        for rank, row in enumerate(unique_display, start=1):
            analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
            md.extend(
                [
                    (
                        f"### #{rank} {row.get('decision')} score={row.get('overall_score')} "
                        f"creative={row.get('creative_score')} control={row.get('control_score')} "
                        f"id={row.get('candidate_id')}"
                    ),
                    "",
                    f"- tags: `{', '.join(row.get('tags') or [])}`",
                    f"- lines: `{analysis.get('line_count')}` words: `{analysis.get('word_count')}` slurs: `{analysis.get('slur_count')}`",
                    f"- notes: {row.get('notes')}",
                ]
            )
            best_lines = row.get("best_lines") or []
            if best_lines:
                md.append(f"- best lines: `{'; '.join(best_lines)}`")
            md.extend(["", "```text", str(row.get("generated_text") or ""), "```", ""])

    args.output_md.write_text("\n".join(md), encoding="utf-8")
    summary = {
        "judgments": len(judgments),
        "decision_counts": dict(decision_counts),
        "tag_counts": dict(tag_counts.most_common()),
        "output_md": str(args.output_md),
        "output_jsonl": str(args.output_jsonl),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="Call OpenAI to judge a locally ranked candidate shortlist.")
    judge.add_argument("--input-jsonl", type=Path, required=True)
    judge.add_argument("--output-judgments", type=Path, required=True)
    judge.add_argument("--model", default=os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_MODEL))
    judge.add_argument("--top-per-prompt", type=int, default=20)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--batch-size", type=int, default=6)
    judge.add_argument("--max-chars", type=int, default=1700)
    judge.add_argument("--timeout", type=int, default=160)
    judge.add_argument("--retries", type=int, default=4)
    judge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    judge.add_argument("--dry-run", action="store_true")
    judge.set_defaults(func=cmd_judge)

    report = subparsers.add_parser("report", help="Build a Markdown shortlist from saved OpenAI judgments.")
    report.add_argument("--judgments", type=Path, required=True)
    report.add_argument("--output-md", type=Path, required=True)
    report.add_argument("--output-jsonl", type=Path, required=True)
    report.add_argument("--top-per-prompt", type=int, default=8)
    report.set_defaults(func=cmd_report)
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
