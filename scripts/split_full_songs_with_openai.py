"""Use OpenAI to split full raw rap songs into training-ready sections.

This is the expensive/high-leverage path: send whole song records from the raw
song parquet/CSV mirror and let OpenAI select usable verse/hook sections instead
of only judging locally split snippets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"

SONG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "songs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "song_key": {"type": "string"},
                    "keep_song": {"type": "boolean"},
                    "song_quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "song_notes": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "section_id": {"type": "string"},
                                "section_role": {"type": "string", "enum": ["verse", "hook", "bridge", "intro_outro", "drop"]},
                                "keep_for_sft": {"type": "boolean"},
                                "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "creativity_score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "control_score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "theme": {"type": "string"},
                                "line_count": {"type": "integer", "minimum": 0, "maximum": 80},
                                "clean_ending": {"type": "boolean"},
                                "lyric_only": {"type": "boolean"},
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
                                "start_line": {"type": "integer", "minimum": 1, "maximum": 1000},
                                "end_line": {"type": "integer", "minimum": 1, "maximum": 1000},
                                "notes": {"type": "string"},
                            },
                            "required": [
                                "section_id",
                                "section_role",
                                "keep_for_sft",
                                "quality_score",
                                "creativity_score",
                                "control_score",
                                "theme",
                                "line_count",
                                "clean_ending",
                                "lyric_only",
                                "failure_tags",
                                "start_line",
                                "end_line",
                                "notes",
                            ],
                        },
                    },
                },
                "required": ["song_key", "keep_song", "song_quality_score", "song_notes", "sections"],
            },
        }
    },
    "required": ["songs"],
}


def iter_parquet_rows(path: Path, *, batch_size: int, columns: list[str]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    selected = [column for column in columns if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield from batch.to_pylist()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)


def compact_song(row: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    lyrics = str(row.get("lyrics") or "")
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    numbered_lines: list[str] = []
    char_count = 0
    for index, line in enumerate(lines, start=1):
        numbered = f"{index}: {line}"
        if char_count + len(numbered) + 1 > max_chars:
            break
        numbered_lines.append(numbered)
        char_count += len(numbered) + 1
    return {
        "song_key": str(row.get("id") or hashlib.sha1(lyrics.encode("utf-8", errors="ignore")).hexdigest()[:16]),
        "title": str(row.get("title") or ""),
        "artist": str(row.get("artist") or row.get("artist_clean") or ""),
        "year": row.get("year"),
        "views": row.get("views"),
        "tag": row.get("tag"),
        "language": row.get("language") or row.get("language_ft") or row.get("language_cld3"),
        "numbered_lyrics": "\n".join(numbered_lines),
        "_source_lines": lines,
        "truncated": len(numbered_lines) < len(lines),
    }


def song_ok(row: dict[str, Any]) -> bool:
    lyrics = str(row.get("lyrics") or "")
    tag = str(row.get("tag") or "").lower()
    language = str(row.get("language") or row.get("language_ft") or row.get("language_cld3") or "").lower()
    if tag and tag != "rap":
        return False
    if language and language != "en":
        return False
    if len(words(lyrics)) < 80:
        return False
    return True


def load_done_song_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("song_key")
            if isinstance(key, str):
                done.add(key)
    return done


def load_failed_song_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    failed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys = row.get("song_keys")
            if isinstance(keys, list):
                failed.update(str(key) for key in keys if key)
            key = row.get("song_key")
            if isinstance(key, str):
                failed.add(key)
    return failed


def load_manifest_song_keys(path: Path, *, offset: int, limit: int | None) -> list[str]:
    keys: list[str] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if skipped < offset:
                skipped += 1
                continue
            row = json.loads(line)
            key = row.get("song_key")
            if isinstance(key, str) and key:
                keys.append(key)
            if limit is not None and len(keys) >= limit:
                break
    return keys


def request_payload(songs: list[dict[str, Any]], *, model: str, max_output_tokens: int) -> dict[str, Any]:
    instructions = (
        "You are preparing a rap lyric SFT corpus from full raw songs. "
        "For each song, inspect the full lyrics and extract only the best usable sections for training assistant outputs. "
        "Prefer complete verses and hooks that are lyric-only, vivid, coherent, line-broken, and have clean endings. "
        "Remove section headers, producer credits, artist/source names, bracket labels, annotations, stage directions, and non-lyric chatter. "
        "Do not reject solely for profanity, rap slang, dark imagery, street vocabulary, or the word nigga. "
        "Reject bad scrape junk, giant prose, dialogue/stage chatter, severe repetition, copied artist leakage, broken endings, or too-short fragments. "
        "Return only compact line ranges into the numbered lyrics. Do not copy lyric text into the JSON. "
        "Use start_line and end_line to identify the cleaned usable section, excluding section headers and producer credits. "
        "Keep at most 2 sections per song, prefer one complete verse and one hook when both are strong. "
        "If the song is too messy, keep zero sections."
    )
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "Split and label these full raw rap songs. Preserve song_key exactly.\n\n"
                + json.dumps(songs, ensure_ascii=False, indent=2),
            },
        ],
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "raw_song_section_split",
                "schema": SONG_SCHEMA,
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


def response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
    return ""


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
            time.sleep(min(120, 2**attempt + random.random()))
    raise RuntimeError(f"OpenAI API request failed after {retries + 1} attempts: {last_error}") from last_error


def text_from_range(source: dict[str, Any], start_line: int, end_line: int) -> str:
    lines = source.get("_source_lines") if isinstance(source.get("_source_lines"), list) else []
    if not lines:
        return ""
    start = max(1, int(start_line or 1))
    end = min(len(lines), int(end_line or start))
    if end < start:
        return ""
    selected = []
    for line in lines[start - 1 : end]:
        stripped = str(line).strip()
        if not stripped or (stripped.startswith("[") and stripped.endswith("]")):
            continue
        selected.append(stripped)
    return "\n".join(selected).strip()


def cmd_split(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is required.")
    if args.limit is not None and args.limit <= 0:
        args.limit = None

    columns = ["title", "tag", "artist", "artist_clean", "year", "views", "id", "language_cld3", "language_ft", "language", "lyrics"]
    done = load_done_song_keys(args.output_sections) if args.resume else set()
    failed = load_failed_song_keys(args.error_output) if args.resume and args.skip_failed else set()
    manifest_keys = load_manifest_song_keys(args.manifest, offset=args.manifest_offset, limit=args.limit) if args.manifest else None
    manifest_key_set = set(manifest_keys) if manifest_keys is not None else None
    args.output_sections.parent.mkdir(parents=True, exist_ok=True)
    args.output_songs.parent.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    offset_skipped = 0
    for row in iter_parquet_rows(args.input, batch_size=args.read_batch_size, columns=columns):
        scanned += 1
        if args.scan_limit is not None and scanned > args.scan_limit:
            break
        if not song_ok(row):
            continue
        eligible += 1
        lyrics = str(row.get("lyrics") or "")
        current_key = str(row.get("id") or hashlib.sha1(lyrics.encode("utf-8", errors="ignore")).hexdigest()[:16])
        if manifest_key_set is not None and current_key not in manifest_key_set:
            continue
        if manifest_key_set is None and eligible <= args.offset:
            offset_skipped += 1
            continue
        song = compact_song(row, max_chars=args.max_chars)
        if song["song_key"] in done:
            continue
        if song["song_key"] in failed:
            continue
        selected.append(song)
        if manifest_key_set is None and args.limit is not None and len(selected) >= args.limit:
            break
        if manifest_key_set is not None and len(selected) >= len(manifest_key_set - done - failed):
            break

    if args.dry_run:
        print(json.dumps({"scanned": scanned, "eligible": eligible, "offset_skipped": offset_skipped, "selected": len(selected), "first": selected[: args.batch_size]}, indent=2, ensure_ascii=False))
        return

    with args.output_sections.open("a", encoding="utf-8") as section_handle, args.output_songs.open("a", encoding="utf-8") as song_handle:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            started = time.perf_counter()
            response = post_with_retries(
                request_payload(batch, model=args.model, max_output_tokens=args.max_output_tokens),
                api_key=api_key or "",
                timeout=args.timeout,
                retries=args.retries,
            )
            elapsed = round(time.perf_counter() - started, 3)
            try:
                parsed = parse_response_json(response)
            except Exception as exc:  # noqa: BLE001
                args.error_output.parent.mkdir(parents=True, exist_ok=True)
                with args.error_output.open("a", encoding="utf-8") as error_handle:
                    error_handle.write(
                        json.dumps(
                            {
                                "batch": start // args.batch_size + 1,
                                "error": str(exc),
                                "song_keys": [song.get("song_key") for song in batch],
                                "response_id": response.get("id"),
                                "response_text": response_text(response)[:8000],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if args.continue_on_error:
                    print(json.dumps({"batch": start // args.batch_size + 1, "songs": len(batch), "sections": 0, "error": str(exc), "continued": True}, ensure_ascii=False))
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue
                raise
            songs = parsed.get("songs")
            if not isinstance(songs, list):
                raise RuntimeError("Structured response did not contain songs array")
            source_by_key = {song["song_key"]: song for song in batch}
            written_sections = 0
            for song_result in songs:
                song_key = str(song_result.get("song_key") or "")
                source = source_by_key.get(song_key, {})
                clean_source = {
                    key: value
                    for key, value in source.items()
                    if not key.startswith("_") and key != "numbered_lyrics"
                }
                song_row = {**clean_source, **song_result, "model": args.model, "elapsed_seconds": elapsed, "response_id": response.get("id"), "usage": response.get("usage")}
                song_handle.write(json.dumps(song_row, ensure_ascii=False) + "\n")
                for section_index, section in enumerate(song_result.get("sections") or []):
                    section_text = text_from_range(source, int(section.get("start_line") or 1), int(section.get("end_line") or 1))
                    if not section_text:
                        continue
                    record = {
                        **clean_source,
                        **section,
                        "text": section_text,
                        "song_key": song_key,
                        "section_index": section_index,
                        "record_id": f"song:{song_key}:openai_section:{section_index}:{hashlib.sha1(section_text.encode('utf-8', errors='ignore')).hexdigest()[:12]}",
                        "model": args.model,
                        "elapsed_seconds": elapsed,
                        "response_id": response.get("id"),
                    }
                    section_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written_sections += 1
            section_handle.flush()
            song_handle.flush()
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            print(json.dumps({"batch": start // args.batch_size + 1, "songs": len(batch), "sections": written_sections, "remaining": max(0, len(selected) - start - len(batch)), "elapsed_seconds": elapsed}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    split.add_argument("--manifest", type=Path, default=None, help="Ranked manifest JSONL with song_key values to select from the full corpus.")
    split.add_argument("--manifest-offset", type=int, default=0, help="Skip this many manifest rows before selecting keys.")
    split.add_argument("--output-sections", type=Path, default=Path("data/labels/raw_full_song_openai_sections.jsonl"))
    split.add_argument("--output-songs", type=Path, default=Path("data/labels/raw_full_song_openai_song_reviews.jsonl"))
    split.add_argument("--error-output", type=Path, default=Path("data/labels/raw_full_song_openai_errors.jsonl"))
    split.add_argument("--model", default=os.environ.get("OPENAI_FULL_SONG_MODEL", DEFAULT_MODEL))
    split.add_argument("--limit", type=int, default=5000)
    split.add_argument("--offset", type=int, default=0, help="Skip this many filter-eligible songs before selecting work; useful for independent shards.")
    split.add_argument("--scan-limit", type=int, default=None)
    split.add_argument("--read-batch-size", type=int, default=4096)
    split.add_argument("--batch-size", type=int, default=1)
    split.add_argument("--max-chars", type=int, default=8000)
    split.add_argument("--max-output-tokens", type=int, default=12000)
    split.add_argument("--timeout", type=int, default=240)
    split.add_argument("--retries", type=int, default=4)
    split.add_argument("--sleep-seconds", type=float, default=1.0)
    split.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    split.add_argument("--skip-failed", action=argparse.BooleanOptionalAction, default=True)
    split.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    split.add_argument("--dry-run", action="store_true")
    split.set_defaults(func=cmd_split)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
