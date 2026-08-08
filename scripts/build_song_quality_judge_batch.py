#!/usr/bin/env python3
"""Build and materialize a blind, strict OpenAI Batch quality calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_INPUT = Path("data/reviews/song_triage_candidate_pool_v1.parquet")
DEFAULT_OUTPUT_DIR = Path("data/openai_batch/song_quality_judge_gpt55_calibration_1k")
DEFAULT_MODEL = "gpt-5.5"

SCORE = {"type": "integer", "minimum": 1, "maximum": 5}
JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "song_key": {"type": "string"},
        "overall": SCORE,
        "technical_rhyme": SCORE,
        "flow_cadence": SCORE,
        "coherence": SCORE,
        "thematic_depth": SCORE,
        "imagery": SCORE,
        "ending_strength": SCORE,
        "family_compliance": SCORE,
        "cleanliness": SCORE,
        "genericness": SCORE,
        "internal_rhyme_count": {"type": "integer", "minimum": 0},
        "multisyllabic_rhyme_count": {"type": "integer", "minimum": 0},
        "repeated_ending_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "syllable_variance": {"type": "number", "minimum": 0},
        "unresolved_final_fragment": {"type": "boolean"},
        "critical_failure_flags": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "string",
                "enum": [
                    "scrape_artifact", "non_lyric", "broken_structure", "incoherent",
                    "generic_filler", "repetition_collapse", "weak_ending",
                    "unsafe_for_clean", "artist_or_metadata_leak", "none",
                ],
            },
        },
        "strict_pass": {"type": "boolean"},
        "notes": {"type": "string", "maxLength": 240},
    },
    "required": [
        "song_key", "overall", "technical_rhyme", "flow_cadence", "coherence",
        "thematic_depth", "imagery", "ending_strength", "family_compliance",
        "cleanliness", "genericness", "internal_rhyme_count",
        "multisyllabic_rhyme_count", "repeated_ending_ratio", "syllable_variance",
        "unresolved_final_fragment", "critical_failure_flags", "strict_pass", "notes",
    ],
}


def compact_lyrics(text: str, max_chars: int) -> tuple[str, bool]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept), len(kept) < len(lines)


def select_balanced(path: Path, per_family: int) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["song_key", "lyrics", "candidate_families"])
    buckets: dict[str, list[dict[str, Any]]] = {"technical": [], "clean": []}
    for row in table.to_pylist():
        families = set(row.get("candidate_families") or [])
        for family in buckets:
            if family in families:
                buckets[family].append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for family in ("technical", "clean"):
        for row in buckets[family]:
            key = str(row.get("song_key") or "")
            if not key or key in used:
                continue
            selected.append({**row, "judge_family": family})
            used.add(key)
            if sum(item["judge_family"] == family for item in selected) >= per_family:
                break
        if sum(item["judge_family"] == family for item in selected) < per_family:
            raise RuntimeError(f"Not enough unique {family} candidates for {per_family}")
    return selected


def request_body(row: dict[str, Any], model: str, max_chars: int, max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    lyrics, truncated = compact_lyrics(str(row["lyrics"]), max_chars)
    family = row["judge_family"]
    system = (
        "You are an independent rap-corpus quality judge. Score the supplied lyrics blindly and strictly. "
        "Do not infer quality from fame, metadata, or prior selection. Scores use 1=poor through 5=excellent. "
        "Estimate structural metrics from visible lines; repeated_ending_ratio is the fraction of line endings "
        "that substantially repeat, and syllable_variance is the approximate standard deviation of syllables "
        "per nonempty line. For technical candidates, demand controlled internal and multisyllabic rhyme, stable "
        "cadence, semantic continuity, resolved phrasing, and a strong ending. For clean candidates, demand natural "
        "language, no profanity/slurs/explicit sex/graphic violence, retained tension/personality/imagery, and craft "
        "comparable to unrestricted lyrics. strict_pass is true only when overall, technical_rhyme, flow_cadence, "
        "coherence, thematic_depth, ending_strength are all >=4; family_compliance=5; there are no critical failures; "
        "and for clean, cleanliness=5 and genericness<=2. Use ['none'] when there are no critical flags."
    )
    payload = {"song_key": str(row["song_key"]), "target_family": family, "lyrics": lyrics}
    body = {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Judge this candidate. Preserve song_key exactly.\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "max_output_tokens": max_output_tokens,
        "text": {
            "verbosity": "low",
            "format": {"type": "json_schema", "name": "song_quality_judgment", "schema": JUDGMENT_SCHEMA, "strict": True},
        },
    }
    source = {
        "custom_id": f"songquality:{row['song_key']}", "song_key": str(row["song_key"]),
        "judge_family": family, "input_chars": len(lyrics), "truncated": truncated,
    }
    return body, source


def response_json(body: dict[str, Any]) -> dict[str, Any]:
    if isinstance(body.get("output_text"), str):
        return json.loads(body["output_text"])
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return json.loads(content["text"])
    raise ValueError("response has no JSON text")


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def cmd_build(args: argparse.Namespace) -> None:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_balanced(args.input, args.per_family)
    request_path = args.output_dir / f"{args.name}_requests.jsonl"
    source_path = args.output_dir / f"{args.name}_source_map.jsonl"
    input_tokens = 0
    with request_path.open("w", encoding="utf-8", newline="\n") as requests, source_path.open("w", encoding="utf-8", newline="\n") as sources:
        for row in selected:
            body, source = request_body(row, args.model, args.max_chars, args.max_output_tokens)
            batch_row = {"custom_id": source["custom_id"], "method": "POST", "url": "/v1/responses", "body": body}
            requests.write(json.dumps(batch_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            sources.write(json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n")
            input_tokens += max(math.ceil(source["input_chars"] / 3.6), 1) + 300
    summary = {
        "name": args.name, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": round(time.time() - started, 3), "command": " ".join([sys.executable, *sys.argv]),
        "input": str(args.input), "model": args.model, "reasoning_effort": "low",
        "requests": len(selected), "family_counts": dict(Counter(row["judge_family"] for row in selected)),
        "max_chars": args.max_chars, "max_output_tokens": args.max_output_tokens,
        "estimated_input_tokens": input_tokens,
        "request_bytes": request_path.stat().st_size,
        "source_map": str(source_path),
        "shards": [{"path": str(request_path), "requests": len(selected), "bytes": request_path.stat().st_size}],
    }
    summary_path = args.output_dir / f"{args.name}_batch_request_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_materialize(args: argparse.Namespace) -> None:
    sources = {row["custom_id"]: row for row in iter_jsonl([args.source_map])}
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    usage = Counter()
    for row in iter_jsonl(sorted(args.batch_output_dir.glob("*_output.jsonl"))):
        custom_id = str(row.get("custom_id") or "")
        response = row.get("response") or {}
        body = response.get("body") if isinstance(response, dict) else None
        if not isinstance(body, dict) or int(response.get("status_code") or 0) != 200:
            errors.append({"custom_id": custom_id, "error": "response_error"})
            continue
        try:
            judgment = response_json(body)
            source = sources[custom_id]
            if str(judgment.get("song_key")) != str(source["song_key"]):
                raise ValueError("song_key mismatch")
            output_rows.append({**source, **judgment, "response_id": body.get("id")})
            for key, value in (body.get("usage") or {}).items():
                if isinstance(value, int): usage[key] += value
        except Exception as exc:  # noqa: BLE001
            errors.append({"custom_id": custom_id, "error": str(exc)})
    output_rows.sort(key=lambda row: (row["judge_family"], row["song_key"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    family_pass = Counter((row["judge_family"], bool(row["strict_pass"])) for row in output_rows)
    summary = {
        "responses": len(output_rows), "errors": len(errors), "usage": dict(usage),
        "family_strict_pass": {f"{family}:{passed}": count for (family, passed), count in sorted(family_pass.items())},
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if errors:
        args.errors.write_text("\n".join(json.dumps(row) for row in errors) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--name", default="song_quality_judge_gpt55_calibration_1k")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--per-family", type=int, default=500)
    build.add_argument("--max-chars", type=int, default=6000)
    build.add_argument("--max-output-tokens", type=int, default=900)
    build.set_defaults(func=cmd_build)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--batch-output-dir", type=Path, required=True)
    materialize.add_argument("--source-map", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--summary", type=Path, required=True)
    materialize.add_argument("--errors", type=Path, required=True)
    materialize.set_defaults(func=cmd_materialize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
