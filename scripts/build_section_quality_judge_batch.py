#!/usr/bin/env python3
"""Build a section-specific GPT-5.5 Batch calibration without submitting it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/section_mining/section_mining_calibration_selected_800.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/openai_batch/section_quality_judge_gpt55_calibration")
SCORE = {"type": "integer", "minimum": 1, "maximum": 5}
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "section_id": {"type": "string"}, "overall": SCORE, "coherence": SCORE,
        "thematic_specificity": SCORE, "ending_strength": SCORE,
        "natural_phrasing_cadence": SCORE, "technical_rhyme": SCORE,
        "genericness": SCORE, "safety": SCORE, "self_contained_excerpt": SCORE,
        "critical_failure_flags": {
            "type": "array", "maxItems": 5,
            "items": {"type": "string", "enum": [
                "unsafe", "generic_filler", "incoherent", "weak_ending", "awkward_cadence",
                "weak_technical_rhyme", "repetition_collapse", "incomplete_fragment",
                "scrape_artifact", "artist_or_metadata_leak", "none",
            ]},
        },
        "strict_pass": {"type": "boolean"},
        "notes": {"type": "string", "maxLength": 220},
    },
    "required": [
        "section_id", "overall", "coherence", "thematic_specificity", "ending_strength",
        "natural_phrasing_cadence", "technical_rhyme", "genericness", "safety",
        "self_contained_excerpt", "critical_failure_flags", "strict_pass", "notes",
    ],
}


def computed_strict_pass(row: dict[str, Any]) -> bool:
    base = all(int(row.get(field) or 0) >= 4 for field in (
        "overall", "coherence", "thematic_specificity", "ending_strength",
        "natural_phrasing_cadence", "self_contained_excerpt",
    ))
    flags = set(row.get("critical_failure_flags") or []) - {"none"}
    if row.get("family") == "technical":
        return base and int(row.get("technical_rhyme") or 0) >= 4 and int(row.get("safety") or 0) >= 3 and not flags
    if row.get("family") == "clean":
        return base and int(row.get("technical_rhyme") or 0) >= 3 and int(row.get("safety") or 0) == 5 and int(row.get("genericness") or 6) <= 2 and not flags
    if row.get("family") == "story":
        return base and int(row.get("technical_rhyme") or 0) >= 3 and int(row.get("safety") or 0) >= 3 and int(row.get("genericness") or 6) <= 2 and not flags
    return False


def request_body(row: dict[str, Any], model: str, max_output_tokens: int) -> dict[str, Any]:
    family = row["family"]
    system = (
        "You are independently judging one exactly 12-line rap excerpt as a potential individual training example. "
        "It is an excerpt, not a complete song: do not require album-scale narrative development, a hook, intro, or outro. "
        "Demand a coherent and self-contained thought, thematic specificity, natural cadence, and an earned final line. "
        "For technical excerpts, technical_rhyme must reflect controlled internal or multisyllabic rhyme without semantic loss. "
        "For clean excerpts, safety=5 requires no profanity, slurs, explicit sexual content, graphic violence, or drug promotion; "
        "genericness must be strongly penalized, while technical_rhyme is secondary but still competent. Scores are 1=poor to "
        "5=excellent. strict_pass for technical requires overall/coherence/thematic_specificity/ending_strength/"
        "natural_phrasing_cadence/self_contained_excerpt/technical_rhyme >=4, safety>=3, and no critical flags. strict_pass "
        "for clean requires those base dimensions >=4, technical_rhyme>=3, safety=5, genericness<=2, and no critical flags. "
        "For story excerpts, require the same base dimensions >=4, technical_rhyme>=3, safety>=3, genericness<=2, and no "
        "critical flags; reward a concrete, continuously developing scene or idea rather than demanding complete-song scope. "
        "Use ['none'] when no critical failure applies. Judge only the supplied excerpt."
    )
    payload = {"section_id": row["section_id"], "target_family": family, "lyrics": row["text"]}
    return {
        "model": model, "store": False, "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Score this 12-line excerpt and preserve section_id exactly.\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "max_output_tokens": max_output_tokens,
        "text": {"verbosity": "low", "format": {"type": "json_schema", "name": "section_quality_judgment", "schema": SCHEMA, "strict": True}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="section_quality_judge_gpt55_calibration")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-output-tokens", type=int, default=650)
    args = parser.parse_args()
    started = time.time()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / f"{args.name}_requests.jsonl"
    source_path = args.output_dir / f"{args.name}_source_map.jsonl"
    estimated_input = 0
    with request_path.open("w", encoding="utf-8", newline="\n") as request_file, source_path.open("w", encoding="utf-8", newline="\n") as source_file:
        for row in rows:
            custom_id = f"sectionquality:{row['section_id']}"
            body = request_body(row, args.model, args.max_output_tokens)
            request_file.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}, ensure_ascii=False, separators=(",", ":")) + "\n")
            source = {key: row[key] for key in ("section_id", "song_key", "family", "source_start_line", "source_end_line", "text_hash", "local_score")}
            source_file.write(json.dumps({"custom_id": custom_id, **source}, separators=(",", ":")) + "\n")
            estimated_input += math.ceil(len(json.dumps(body["input"], ensure_ascii=False)) / 3.6)
    # Published standard GPT-5.5 prices are $5/M input and $30/M output; Batch is 50%.
    maximum_output = len(rows) * args.max_output_tokens
    expected_output = len(rows) * 230
    summary = {
        "name": args.name, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": round(time.time() - started, 3), "command": " ".join([sys.executable, *sys.argv]),
        "input": str(args.input), "model": args.model, "reasoning_effort": "low",
        "requests": len(rows), "family_counts": dict(Counter(row["family"] for row in rows)),
        "estimated_input_tokens": estimated_input, "expected_output_tokens": expected_output,
        "maximum_output_tokens": maximum_output, "max_output_tokens_per_request": args.max_output_tokens,
        "estimated_batch_cost_usd": round((estimated_input / 1_000_000 * 5 + expected_output / 1_000_000 * 30) * 0.5, 2),
        "maximum_batch_cost_usd": round((estimated_input / 1_000_000 * 5 + maximum_output / 1_000_000 * 30) * 0.5, 2),
        "request_bytes": request_path.stat().st_size, "source_map": str(source_path),
        "shards": [{"path": str(request_path), "requests": len(rows), "bytes": request_path.stat().st_size}],
        "submission_status": "not_submitted_requires_explicit_user_approval",
    }
    summary_path = args.output_dir / f"{args.name}_batch_request_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
