"""Build tiny-output OpenAI Batch requests for full-song corpus triage.

This is a budget-first pass over the materialized review pool. It asks for a
coarse keep/reject/uncertain decision and compact tags, not section extraction
or long critique.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/review_pool/rap_song_review_pool_top250k.parquet")
DEFAULT_OUTPUT_DIR = Path("data/openai_batch/song_triage_nano")
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_INPUT_PRICE_PER_M = 0.10
DEFAULT_OUTPUT_PRICE_PER_M = 0.625
DEFAULT_BUDGET_USD = 70.0


TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "reject", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_floor": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "string",
                "enum": [
                    "strong_candidate",
                    "technical_rhyme",
                    "clean_candidate",
                    "coherent_story",
                    "vivid_imagery",
                    "good_cadence",
                    "generic",
                    "too_messy",
                    "scrape_artifact",
                    "non_lyric",
                    "incomplete",
                    "repetitive",
                    "weak_craft",
                    "unsafe_clean_training",
                    "artist_or_metadata_leak",
                ],
            },
        },
        "reason": {"type": "string", "maxLength": 64},
    },
    "required": ["decision", "confidence", "quality_floor", "tags", "reason"],
}


TRIAGE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "song_key": {"type": "string"},
        **TRIAGE_SCHEMA["properties"],
    },
    "required": ["song_key", *TRIAGE_SCHEMA["required"]],
}


TRIAGE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reviews": {
            "type": "array",
            "items": TRIAGE_REVIEW_SCHEMA,
        }
    },
    "required": ["reviews"],
}


@dataclass(frozen=True)
class SongRow:
    rank: int
    song_key: str
    title: str
    artist: str
    year: Any
    views: Any
    lyrics: str


def approx_tokens(text: str) -> int:
    """Conservative enough for budgeting without requiring a tokenizer."""
    if not text:
        return 0
    return max(math.ceil(len(text) / 3.6), math.ceil(len(text.split()) * 1.35))


def compact_lyrics(lyrics: str, *, max_chars: int) -> tuple[str, bool]:
    lines = [line.strip() for line in str(lyrics or "").splitlines() if line.strip()]
    kept: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), len(kept) < len(lines)


def iter_song_rows(path: Path, *, offset: int, limit: int | None, columns: list[str]) -> Iterable[SongRow]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    selected = [column for column in columns if column in available]
    seen = 0
    emitted = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=selected):
        for row in batch.to_pylist():
            if seen < offset:
                seen += 1
                continue
            if limit is not None and emitted >= limit:
                return
            seen += 1
            emitted += 1
            song_key = str(row.get("song_key") or row.get("id") or "")
            yield SongRow(
                rank=int(row.get("manifest_rank") or row.get("rank") or seen),
                song_key=song_key,
                title=str(row.get("title") or ""),
                artist=str(row.get("artist") or row.get("artist_clean") or ""),
                year=row.get("year"),
                views=row.get("views"),
                lyrics=str(row.get("lyrics") or ""),
            )


def request_body(song: SongRow, *, model: str, max_chars: int, max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_body_many([song], model=model, max_chars=max_chars, max_output_tokens=max_output_tokens)


def request_body_many(
    songs: list[SongRow], *, model: str, max_chars: int, max_output_tokens: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not songs:
        raise ValueError("songs is required")
    song_payloads = []
    source_songs = []
    for song in songs:
        lyrics, truncated = compact_lyrics(song.lyrics, max_chars=max_chars)
        song_payloads.append(
            {
                "song_key": song.song_key,
                "rank": song.rank,
                "title": song.title,
                "artist": song.artist,
                "year": song.year,
                "views": song.views,
                "lyrics": lyrics,
                "truncated": truncated,
            }
        )
        source_songs.append(
            {
                "song_key": song.song_key,
                "rank": song.rank,
                "title": song.title,
                "artist": song.artist,
                "year": song.year,
                "views": song.views,
                "truncated": truncated,
                "input_chars": len(lyrics),
            }
        )
    user_payload = {"songs": song_payloads}
    schema = TRIAGE_SCHEMA if len(songs) == 1 else TRIAGE_BATCH_SCHEMA
    prompt = "Triage this rap song. Preserve song_key exactly." if len(songs) == 1 else "Triage each rap song. Preserve every song_key exactly."
    instructions = (
        "You are triaging rap lyrics for a training corpus. Decide whether this full song should move to stricter "
        "filtering. Keep songs with strong lyric craft, coherent verse writing, technical rhyme, vivid imagery, story, "
        "or clean-training potential. Reject obvious scrape junk, metadata/web artifacts, non-lyrics, very weak craft, "
        "generic filler, severe repetition, broken fragments, or songs unsuitable for clean training. Profanity, rap "
        "slang, dark imagery, and street vocabulary are not automatic rejection reasons. Return compact JSON only. "
        "Use at most 3 tags. Keep reason under 8 words."
    )
    body = {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": prompt + "\n" + json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "max_output_tokens": max_output_tokens,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "song_triage",
                "schema": schema,
                "strict": True,
            }
        },
    }
    custom_id = f"songtriage:{songs[0].song_key}" if len(songs) == 1 else f"songtriagepack:{songs[0].song_key}-{songs[-1].song_key}"
    source_row = {
        "custom_id": custom_id,
        "songs": source_songs,
    }
    if len(source_songs) == 1:
        source_row.update(source_songs[0])
    return body, source_row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return json.loads(content["text"])
    raise ValueError("Could not find JSON text in OpenAI response")


def iter_jsonl(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def cmd_build(args: argparse.Namespace) -> None:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / f"{args.name}_requests.jsonl"
    source_map_path = args.output_dir / f"{args.name}_source_map.jsonl"
    summary_path = args.output_dir / f"{args.name}_summary.json"
    command_path = args.output_dir / f"{args.name}_command.txt"

    total_input_tokens = 0
    requests = 0
    shard_index = 1
    shard_requests = 0
    shard_bytes = 0
    max_file_bytes = int(args.max_file_mb * 1024 * 1024)
    shards: list[dict[str, Any]] = []
    request_path = args.output_dir / f"{args.name}_shard_{shard_index:04d}.jsonl"
    source_rows: list[dict[str, Any]] = []
    columns = ["manifest_rank", "rank", "song_key", "id", "title", "artist", "artist_clean", "year", "views", "lyrics"]
    pending: list[SongRow] = []

    request_handle = request_path.open("w", encoding="utf-8")

    def rotate_shard_if_needed(encoded_bytes: int) -> None:
        nonlocal request_handle, request_path, shard_index, shard_requests, shard_bytes
        if shard_requests == 0:
            return
        if shard_requests < args.max_requests_per_file and shard_bytes + encoded_bytes <= max_file_bytes:
            return
        request_handle.close()
        shards.append({"path": str(request_path), "requests": shard_requests, "bytes": shard_bytes})
        shard_index += 1
        request_path = args.output_dir / f"{args.name}_shard_{shard_index:04d}.jsonl"
        request_handle = request_path.open("w", encoding="utf-8")
        shard_requests = 0
        shard_bytes = 0

    def flush_pending() -> None:
        nonlocal total_input_tokens, requests, pending, shard_requests, shard_bytes
        if not pending:
            return
        body, source_row = request_body_many(
            pending,
            model=args.model,
            max_chars=args.max_chars,
            max_output_tokens=args.max_output_tokens,
        )
        batch_row = {
            "custom_id": source_row["custom_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        encoded = json.dumps(batch_row, ensure_ascii=False, separators=(",", ":"))
        encoded_bytes = len(encoded.encode("utf-8")) + 1
        rotate_shard_if_needed(encoded_bytes)
        request_handle.write(encoded + "\n")
        total_input_tokens += approx_tokens(encoded)
        shard_requests += 1
        shard_bytes += encoded_bytes
        source_rows.append(source_row)
        requests += 1
        if args.progress_every and requests % args.progress_every == 0:
            print(json.dumps({"requests": requests}, ensure_ascii=False), flush=True)
        pending = []

    try:
        for song in iter_song_rows(args.input, offset=args.offset, limit=args.limit, columns=columns):
            body, source_row = request_body(
                song,
                model=args.model,
                max_chars=args.max_chars,
                max_output_tokens=args.max_output_tokens,
            )
            if args.songs_per_request <= 1:
                batch_row = {
                    "custom_id": source_row["custom_id"],
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                encoded = json.dumps(batch_row, ensure_ascii=False, separators=(",", ":"))
                encoded_bytes = len(encoded.encode("utf-8")) + 1
                rotate_shard_if_needed(encoded_bytes)
                request_handle.write(encoded + "\n")
                total_input_tokens += approx_tokens(encoded)
                shard_requests += 1
                shard_bytes += encoded_bytes
                source_rows.append(source_row)
                requests += 1
                if args.progress_every and requests % args.progress_every == 0:
                    print(json.dumps({"requests": requests}, ensure_ascii=False), flush=True)
            else:
                pending.append(song)
                if len(pending) >= args.songs_per_request:
                    flush_pending()
        flush_pending()
    finally:
        request_handle.close()
    if shard_requests > 0:
        shards.append({"path": str(request_path), "requests": shard_requests, "bytes": shard_bytes})

    write_jsonl(source_map_path, source_rows)
    request_bytes = sum(shard["bytes"] for shard in shards)
    estimated_output_tokens = requests * args.max_output_tokens
    estimated_input_cost = total_input_tokens / 1_000_000 * args.input_price_per_m
    estimated_output_cost = estimated_output_tokens / 1_000_000 * args.output_price_per_m
    estimated_total_cost = estimated_input_cost + estimated_output_cost
    if estimated_total_cost > args.budget_usd and not args.allow_over_budget:
        request_path.unlink(missing_ok=True)
        source_map_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Estimated ${estimated_total_cost:.2f} exceeds budget ${args.budget_usd:.2f}. "
            "Re-run with a lower limit/max-chars/max-output-tokens or --allow-over-budget."
        )

    summary = {
        "schema_version": 1,
        "name": args.name,
        "command": " ".join(args.raw_command),
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "wall_seconds": round(time.time() - started, 3),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "request_path": str(shards[0]["path"]) if shards else str(request_path),
        "source_map_path": str(source_map_path),
        "shards": shards,
        "endpoint": "/v1/responses",
        "model": args.model,
        "reasoning_effort": "low",
        "offset": args.offset,
        "limit": args.limit,
        "requests": requests,
        "songs_per_request": args.songs_per_request,
        "songs": len(source_rows) if args.songs_per_request <= 1 else sum(len(row["songs"]) for row in source_rows),
        "max_chars": args.max_chars,
        "max_output_tokens": args.max_output_tokens,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens_cap": estimated_output_tokens,
        "input_price_per_m": args.input_price_per_m,
        "output_price_per_m": args.output_price_per_m,
        "estimated_input_cost_usd": round(estimated_input_cost, 4),
        "estimated_output_cost_cap_usd": round(estimated_output_cost, 4),
        "estimated_total_cost_cap_usd": round(estimated_total_cost, 4),
        "budget_usd": args.budget_usd,
        "over_budget_allowed": args.allow_over_budget,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    command_path.write_text(summary["command"] + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_materialize(args: argparse.Namespace) -> None:
    started = time.time()
    source_by_custom_id = {}
    source_by_song_key = {}
    for row in iter_jsonl([args.source_map]):
        custom_id = str(row.get("custom_id") or "")
        if custom_id:
            source_by_custom_id[custom_id] = row
        songs = row.get("songs")
        if isinstance(songs, list):
            for song in songs:
                if isinstance(song, dict) and song.get("song_key"):
                    source_by_song_key[str(song["song_key"])] = song
    output_paths = sorted(args.batch_output_dir.glob("*_output.jsonl"))
    if not output_paths:
        raise SystemExit(f"No *_output.jsonl files found in {args.batch_output_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    decisions = {"keep": 0, "reject": 0, "uncertain": 0}
    response_statuses: dict[str, int] = {}
    incomplete_reasons: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    parsed_count = 0
    error_count = 0
    with args.output.open("w", encoding="utf-8") as output_handle, args.errors.open("w", encoding="utf-8") as error_handle:
        for row in iter_jsonl(output_paths):
            custom_id = str(row.get("custom_id") or "")
            response = row.get("response") if isinstance(row.get("response"), dict) else {}
            body = response.get("body") if isinstance(response.get("body"), dict) else {}
            status_code = int(response.get("status_code") or 0)
            status = str(body.get("status") or "missing_body")
            response_statuses[status] = response_statuses.get(status, 0) + 1
            incomplete = body.get("incomplete_details")
            if isinstance(incomplete, dict):
                reason = str(incomplete.get("reason") or "unknown")
                incomplete_reasons[reason] = incomplete_reasons.get(reason, 0) + 1
            body_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            usage["input_tokens"] += int(body_usage.get("input_tokens") or 0)
            usage["output_tokens"] += int(body_usage.get("output_tokens") or 0)
            usage["total_tokens"] += int(body_usage.get("total_tokens") or 0)
            output_details = body_usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                usage["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)
            if status_code != 200 or not body:
                error_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                error_count += 1
                continue
            try:
                parsed = parse_response_json(body)
                if isinstance(parsed.get("reviews"), list):
                    parsed_reviews = parsed["reviews"]
                else:
                    parsed_reviews = [parsed]
                for parsed_review in parsed_reviews:
                    decision = str(parsed_review.get("decision") or "")
                    if decision not in decisions:
                        raise ValueError(f"bad decision: {decision}")
            except Exception as exc:  # noqa: BLE001
                error_handle.write(
                    json.dumps(
                        {"custom_id": custom_id, "error": str(exc), "body": body},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                error_count += 1
                continue
            for parsed_review in parsed_reviews:
                song_key = str(parsed_review.get("song_key") or "")
                source = source_by_song_key.get(song_key) or source_by_custom_id.get(custom_id, {})
                out = {**source, **parsed_review, "custom_id": custom_id, "response_id": body.get("id")}
                output_handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                parsed_count += 1
                decisions[str(parsed_review["decision"])] += 1
                for tag in parsed_review.get("tags") or []:
                    tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1

    estimated_cost = (
        usage["input_tokens"] / 1_000_000 * args.input_price_per_m
        + usage["output_tokens"] / 1_000_000 * args.output_price_per_m
    )
    summary = {
        "schema_version": 1,
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "wall_seconds": round(time.time() - started, 3),
        "batch_output_dir": str(args.batch_output_dir),
        "source_map": str(args.source_map),
        "output": str(args.output),
        "errors": str(args.errors),
        "responses_found": parsed_count + error_count,
        "parsed": parsed_count,
        "errors_count": error_count,
        "response_statuses": dict(sorted(response_statuses.items())),
        "incomplete_reasons": dict(sorted(incomplete_reasons.items())),
        "decisions": decisions,
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "usage": usage,
        "input_price_per_m": args.input_price_per_m,
        "output_price_per_m": args.output_price_per_m,
        "estimated_cost_usd": round(estimated_cost, 4),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--name", default="song_triage_nano_calibration_5k")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--offset", type=int, default=0)
    build.add_argument("--limit", type=int, default=5000)
    build.add_argument("--max-chars", type=int, default=3600)
    build.add_argument("--max-output-tokens", type=int, default=120)
    build.add_argument("--songs-per-request", type=int, default=1)
    build.add_argument("--max-file-mb", type=float, default=180.0)
    build.add_argument("--max-requests-per-file", type=int, default=50000)
    build.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    build.add_argument("--input-price-per-m", type=float, default=DEFAULT_INPUT_PRICE_PER_M)
    build.add_argument("--output-price-per-m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M)
    build.add_argument("--allow-over-budget", action="store_true")
    build.add_argument("--progress-every", type=int, default=5000)
    build.set_defaults(func=cmd_build)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--batch-output-dir", type=Path, required=True)
    materialize.add_argument("--source-map", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--errors", type=Path, required=True)
    materialize.add_argument("--summary-output", type=Path, required=True)
    materialize.add_argument("--input-price-per-m", type=float, default=DEFAULT_INPUT_PRICE_PER_M)
    materialize.add_argument("--output-price-per-m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M)
    materialize.set_defaults(func=cmd_materialize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.raw_command = [sys.executable, *sys.argv]
    args.func(args)


if __name__ == "__main__":
    main()
