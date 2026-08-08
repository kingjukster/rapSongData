"""Label pre-split rap sections with OpenAI and build broader SFT data.

This is meant to run earlier than the refined mixed-SFT playlist. It streams
the canonical processed parquet, labels section behavior, and then builds a
composition-controlled SFT dataset from the labels.
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
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"

TEXT_COLUMNS = [
    "target_completion",
    "clean_bar_text",
    "bar_text",
    "section_text",
    "clean_text",
    "cleaned_text",
    "lyrics",
    "lyric_text",
    "text",
    "content",
    "body",
]
ID_COLUMNS = ["section_id", "song_id", "track_id", "id", "source_id"]
TITLE_COLUMNS = ["title", "song_title", "track_title"]
ARTIST_COLUMNS = ["artist", "artist_name", "artist_clean"]
SECTION_COLUMNS = ["section_type", "section", "label", "predicted_section", "category"]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
STAGE_OR_SOURCE_LINE_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]|\([^)]+\)|(?:intro|outro|verse|hook|chorus|bridge|pre[- ]?chorus)\s*:)\s*$",
    re.IGNORECASE,
)
TRAILING_SOURCE_RE = re.compile(r"\s+[-–—]\s+[A-Z][A-Za-z0-9 .,'&-]{1,40}$")
INCOMPLETE_END_RE = re.compile(
    r"\b(?:and|but|or|so|because|cause|if|when|while|with|without|to|for|from|that|who|what|where|why|how|gon|gonna|wanna|tryna)$",
    re.IGNORECASE,
)
ARTIFACT_RE = re.compile(
    r"(?:download the full version|itunes|genius\\.com|lyrics taken from|you might also like|embed|"
    r"https?://|www\\.|Ã|Â|â€™|â€œ|â€|庭|旁|asarificing|fogggy)",
    re.IGNORECASE,
)
REPETITION_RE = re.compile(r"\\b(\\w{3,})\\b(?:\\W+\\1\\b){3,}", re.IGNORECASE)
ASCII_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)

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
                    "section_bucket": {
                        "type": "string",
                        "enum": [
                            "complete_verse",
                            "hook_or_chorus",
                            "bridge_or_prechorus",
                            "intro_outro",
                            "short_fragment",
                            "adlib_or_chant",
                            "prose_or_dialogue",
                            "artifact_or_corrupt",
                            "non_lyric_or_metadata",
                            "unsafe_off_prompt",
                            "other",
                        ],
                    },
                    "keep_for_sft": {"type": "boolean"},
                    "sft_role": {
                        "type": "string",
                        "enum": ["verse", "hook", "control", "fragment", "drop"],
                    },
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
                    "clean_ending": {"type": "boolean"},
                    "lyric_only": {"type": "boolean"},
                    "estimated_lines": {"type": "integer", "minimum": 0, "maximum": 256},
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
                                "unsafe_derailment",
                                "excessive_repetition",
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
                    "section_bucket",
                    "keep_for_sft",
                    "sft_role",
                    "quality_score",
                    "creativity_score",
                    "control_score",
                    "line_structure",
                    "completion_shape",
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
class SectionRecord:
    index: int
    record_id: str
    text: str
    title: str
    artist: str
    source_section: str
    local_shape: dict[str, Any]
    row: dict[str, Any]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def line_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_present(row: dict[str, Any], columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def stable_id(index: int, row: dict[str, Any], text: str) -> str:
    parts = []
    for column in ID_COLUMNS:
        value = row.get(column)
        if value is not None and str(value).strip():
            parts.append(f"{column}:{value}")
    if parts:
        return "|".join(parts) + f"|row:{index}"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"presplit_row:{index}:sha1:{digest}"


def local_shape(text: str) -> dict[str, Any]:
    lines = line_list(text)
    token_count = len(words(text))
    line_word_counts = [len(words(line)) for line in lines]
    flags: list[str] = []
    if ARTIFACT_RE.search(text):
        flags.append("artifact_or_encoding")
    if len(lines) <= 1 and token_count > 28:
        flags.append("paragraph")
    if token_count < 12 or len(lines) < 2:
        flags.append("too_short")
    if line_word_counts and max(line_word_counts) > 36:
        flags.append("long_line")
    if REPETITION_RE.search(text):
        flags.append("repetition")
    return {
        "line_count": len(lines),
        "word_count": token_count,
        "max_line_words": max(line_word_counts) if line_word_counts else 0,
        "avg_line_words": round(sum(line_word_counts) / len(line_word_counts), 2) if line_word_counts else 0,
        "flags": flags,
    }


def iter_parquet_rows(path: Path, *, batch_size: int, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = parquet.schema.names
    selected = [column for column in (columns or available) if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        for row in batch.to_pylist():
            yield row


def iter_grouped_sections(path: Path, *, batch_size: int, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_parquet_rows(path, batch_size=batch_size, columns=columns):
        section_id = str(row.get("section_id") or row.get("song_id") or f"row_{len(groups)}")
        groups[section_id].append(row)

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        try:
            return int(row.get("bar_index") or 0), str(row.get("bar_id") or "")
        except Exception:
            return 0, str(row.get("bar_id") or "")

    for section_id, rows in groups.items():
        rows = sorted(rows, key=sort_key)
        base = dict(rows[0])
        lines = []
        for row in rows:
            line = first_present(row, ["target_completion", "clean_bar_text", "bar_text"])
            if line:
                lines.append(line)
        base["section_id"] = section_id
        base["target_completion"] = "\n".join(lines)
        base["clean_bar_text"] = "\n".join(lines)
        base["bar_text"] = "\n".join(lines)
        base["bar_count"] = len(lines)
        return_rows = rows[:3]
        base["bar_id"] = str(base.get("bar_id") or "")
        base["source_bar_preview"] = " | ".join(str(item.get("bar_id") or "") for item in return_rows)
        yield base


def extract_section(index: int, row: dict[str, Any]) -> SectionRecord | None:
    text = first_present(row, TEXT_COLUMNS)
    if not text:
        # Some tables store lines as lists.
        for value in row.values():
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value[:5]):
                text = "\n".join(item for item in value if item.strip())
                break
    text = text.strip()
    if not text:
        return None
    title = first_present(row, TITLE_COLUMNS)
    artist = first_present(row, ARTIST_COLUMNS)
    source_section = first_present(row, SECTION_COLUMNS)
    return SectionRecord(
        index=index,
        record_id=stable_id(index, row, text),
        text=text,
        title=title,
        artist=artist,
        source_section=source_section,
        local_shape=local_shape(text),
        row=row,
    )


def load_reviewed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        record_id = payload.get("record_id")
        if isinstance(record_id, str):
            ids.add(record_id)
    return ids


def compact_payload(records: list[SectionRecord], *, max_chars: int) -> str:
    payload = []
    for record in records:
        payload.append(
            {
                "record_id": record.record_id,
                "title": record.title,
                "artist": record.artist,
                "source_section": record.source_section,
                "local_shape": record.local_shape,
                "text": record.text[:max_chars],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_request(records: list[SectionRecord], *, model: str, max_chars: int) -> dict[str, Any]:
    instructions = (
        "You are labeling pre-split rap lyric sections for dataset composition. "
        "Do not reject solely for profanity, rap slang, dark imagery, street vocabulary, or the word nigga. "
        "Keep good rap texture. Label what this section would teach: complete verse, hook, fragment, control-worthy lyric, "
        "or bad data. Drop only for artifacts, corrupt text, non-lyric metadata, giant prose/dialogue drift, severe repetition, "
        "unsafe derailment unrelated to lyric writing, or unusably short/broken content."
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
                    "Label these pre-split section records. Return one label for every record_id, preserving record_id exactly.\n\n"
                    f"{compact_payload(records, max_chars=max_chars)}"
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rap_presplit_section_labels",
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


def collect_candidates(args: argparse.Namespace) -> list[SectionRecord]:
    candidates: list[SectionRecord] = []
    rng = random.Random(args.seed)
    needed_columns = sorted(set(TEXT_COLUMNS + ID_COLUMNS + TITLE_COLUMNS + ARTIST_COLUMNS + SECTION_COLUMNS))
    row_iter = (
        iter_grouped_sections(args.input, batch_size=args.read_batch_size, columns=needed_columns)
        if args.group_by_section
        else iter_parquet_rows(args.input, batch_size=args.read_batch_size, columns=needed_columns)
    )
    for index, row in enumerate(row_iter):
        section = extract_section(index, row)
        if section is None:
            continue
        shape = section.local_shape
        if args.candidate_mode == "quality_shape":
            line_count = int(shape["line_count"])
            word_count = int(shape["word_count"])
            if not (4 <= line_count <= 32 and 30 <= word_count <= 320):
                continue
        elif args.candidate_mode == "complete_or_hook":
            line_count = int(shape["line_count"])
            word_count = int(shape["word_count"])
            if not ((8 <= line_count <= 24 and word_count >= 55) or (3 <= line_count <= 10 and "hook" in section.source_section.lower())):
                continue
        candidates.append(section)
        if args.scan_limit is not None and index + 1 >= args.scan_limit:
            break
    if args.shuffle:
        rng.shuffle(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    return candidates


def cmd_label(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    candidates = collect_candidates(args)
    reviewed = load_reviewed_ids(args.output_labels) if args.resume else set()
    todo = [item for item in candidates if item.record_id not in reviewed]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "selected": len(candidates),
                    "remaining": len(todo),
                    "candidate_mode": args.candidate_mode,
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
        raise SystemExit("OPENAI_API_KEY is required. Add it to .env or use --dry-run.")

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.output_labels.open("a", encoding="utf-8") as handle:
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
            labels = parsed.get("labels")
            if not isinstance(labels, list):
                raise RuntimeError("Structured response did not contain labels array")
            by_id = {item.record_id: item for item in batch}
            for label in labels:
                source = by_id.get(label.get("record_id"))
                if source is None:
                    continue
                output = {
                    **label,
                    "line_index": source.index,
                    "title": source.title,
                    "artist": source.artist,
                    "source_section": source.source_section,
                    "local_shape": source.local_shape,
                    "text": source.text,
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
                        "labeled": len(labels),
                        "remaining": max(0, len(todo) - batch_index - args.batch_size),
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                )
            )


def prompt_for_label(label: dict[str, Any]) -> str:
    role = label.get("sft_role")
    lines = int(label.get("estimated_lines") or (label.get("local_shape") or {}).get("line_count") or 0)
    if role == "hook":
        count = min(max(lines, 4), 8)
        return f"Write exactly {count} lines of a catchy rap hook. Return only lyrics, one line per bar."
    if role == "verse":
        count = min(max(lines, 8), 20)
        return f"Write exactly {count} lines of a rap verse. Return only lyrics, one bar per line."
    if role == "control":
        return "Write a controlled rap section with clean line breaks. Return only lyrics."
    return "Write a short rap lyric section. Return only lyrics."


def target_line_count_for_label(label: dict[str, Any]) -> int:
    role = label.get("sft_role")
    lines = int(label.get("estimated_lines") or (label.get("local_shape") or {}).get("line_count") or 0)
    if role == "hook":
        return min(max(lines, 4), 8)
    if role == "verse":
        return min(max(lines, 8), 20)
    if role == "control":
        return min(max(lines, 4), 12)
    return min(max(lines, 4), 8)


def title_artist_terms(item: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in ("title", "artist"):
        value = str(item.get(key) or "").strip()
        if not value:
            continue
        terms.add(value.lower())
        for token in re.findall(r"[A-Za-z0-9]{4,}", value):
            terms.add(token.lower())
    return terms


def looks_like_source_leak(line: str, item: dict[str, Any]) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if STAGE_OR_SOURCE_LINE_RE.match(stripped):
        return True
    if TRAILING_SOURCE_RE.search(stripped):
        return True
    lowered = stripped.lower()
    if any(marker in lowered for marker in ("lyrics by", "written by", "performed by", "produced by")):
        return True
    for term in title_artist_terms(item):
        if len(term) >= 4 and term in lowered:
            return True
    return False


def split_long_bar(line: str, *, max_words: int) -> list[str]:
    parts = [line.strip()]
    separators = [", ", "; ", " - ", " but ", " cause ", " because ", " while ", " when ", " and "]
    changed = True
    while changed:
        changed = False
        next_parts: list[str] = []
        for part in parts:
            if len(words(part)) <= max_words:
                next_parts.append(part)
                continue
            split_at = -1
            for sep in separators:
                matches = [match.start() + len(sep) for match in re.finditer(re.escape(sep), part, flags=re.IGNORECASE)]
                if not matches:
                    continue
                midpoint = len(part) // 2
                candidate = min(matches, key=lambda pos: abs(pos - midpoint))
                left_words = len(words(part[:candidate]))
                right_words = len(words(part[candidate:]))
                if left_words >= 4 and right_words >= 4:
                    split_at = candidate
                    break
            if split_at > 0:
                next_parts.extend([part[:split_at].strip(" ,;-"), part[split_at:].strip(" ,;-")])
                changed = True
            else:
                tokenized = part.split()
                next_parts.extend(
                    " ".join(tokenized[index : index + max_words]).strip()
                    for index in range(0, len(tokenized), max_words)
                )
                changed = True
        parts = [part for part in next_parts if part]
    return parts


def normalize_text_for_sft(
    text: str,
    *,
    ascii_punctuation: bool,
    item: dict[str, Any] | None = None,
    instruction_shape: bool = False,
    max_line_words: int = 30,
    drop_quoted_lines: bool = False,
    clean_trailing_fragment: bool = False,
) -> str:
    if ascii_punctuation:
        text = text.translate(ASCII_PUNCT_TRANSLATION)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if instruction_shape and item is not None:
        shaped: list[str] = []
        for line in lines:
            if drop_quoted_lines and any(mark in line for mark in ('"', "“", "”")):
                continue
            if looks_like_source_leak(line, item):
                continue
            shaped.extend(split_long_bar(line, max_words=max_line_words))
        lines = shaped
        target = target_line_count_for_label(item)
        if len(lines) > target:
            lines = lines[:target]
        if clean_trailing_fragment:
            while lines and INCOMPLETE_END_RE.search(lines[-1].rstrip(" .,!?'\"")):
                lines.pop()
    return "\n".join(lines).strip()


def has_bad_build_tag(item: dict[str, Any], disallowed_tags: set[str]) -> bool:
    tags = {str(tag) for tag in item.get("failure_tags", [])}
    return bool(tags & disallowed_tags)


def passes_strict_build_filter(item: dict[str, Any], args: argparse.Namespace) -> bool:
    if not item.get("keep_for_sft"):
        return False
    if item.get("sft_role") not in {"verse", "hook", "control", "fragment"}:
        return False
    if int(item.get("quality_score") or 0) < args.min_quality:
        return False
    if args.strict_complete and item.get("completion_shape") != "complete":
        return False
    if args.require_clean_ending and not item.get("clean_ending"):
        return False
    if args.require_lyric_only and not item.get("lyric_only"):
        return False
    if args.require_strong_structure and item.get("line_structure") != "strong":
        return False
    if has_bad_build_tag(item, set(args.disallow_failure_tag)):
        return False
    text = str(item.get("text") or "")
    shape = item.get("local_shape") if isinstance(item.get("local_shape"), dict) else {}
    if int(shape.get("max_line_words") or 0) > args.max_line_words:
        return False
    if args.drop_quoted_lines:
        quote_lines = sum(1 for line in text.splitlines() if '"' in line)
        if quote_lines > args.max_quoted_lines:
            return False
    return True


def cmd_build(args: argparse.Namespace) -> None:
    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [item for item in labels if passes_strict_build_filter(item, args)]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        buckets[str(item.get("sft_role"))].append(item)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    caps = {
        "verse": args.max_verse,
        "hook": args.max_hook,
        "control": args.max_control,
        "fragment": args.max_fragment,
    }
    for role, cap in caps.items():
        rows = buckets.get(role, [])
        rng.shuffle(rows)
        selected.extend(rows[:cap])
    rng.shuffle(selected)
    if args.max_records is not None:
        selected = selected[: args.max_records]

    records = []
    for item in selected:
        text = normalize_text_for_sft(
            str(item.get("text") or ""),
            ascii_punctuation=args.ascii_punctuation,
            item=item,
            instruction_shape=args.instruction_shape,
            max_line_words=args.output_max_line_words,
            drop_quoted_lines=args.output_drop_quoted_lines,
            clean_trailing_fragment=args.clean_trailing_fragment,
        )
        if not text:
            continue
        if args.require_shaped_min_lines and len(line_list(text)) < min(8, target_line_count_for_label(item)):
            continue
        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Write only original rap lyrics. Keep line breaks. Do not explain.",
                    },
                    {"role": "user", "content": prompt_for_label(item)},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {
                    "sft_source": "openai_labeled_presplit_section",
                    "record_id": item.get("record_id"),
                    "section_bucket": item.get("section_bucket"),
                    "sft_role": item.get("sft_role"),
                    "quality_score": item.get("quality_score"),
                    "creativity_score": item.get("creativity_score"),
                    "control_score": item.get("control_score"),
                    "source_section": item.get("source_section"),
                    "title": item.get("title"),
                    "artist": item.get("artist"),
                },
            }
        )

    val_count = min(args.validation_records, max(0, len(records) // 10))
    validation = records[:val_count]
    train = records[val_count:]
    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_validation.parent.mkdir(parents=True, exist_ok=True)
    with args.output_train.open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.output_validation.open("w", encoding="utf-8") as handle:
        for row in validation:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "labels": str(args.labels),
        "eligible": len(eligible),
        "selected": len(records),
        "train": len(train),
        "validation": len(validation),
        "selected_roles": dict(Counter((row["metadata"]["sft_role"] for row in records))),
        "selected_buckets": dict(Counter((row["metadata"]["section_bucket"] for row in records))),
        "build_shaping": {
            "instruction_shape": args.instruction_shape,
            "output_max_line_words": args.output_max_line_words,
            "output_drop_quoted_lines": args.output_drop_quoted_lines,
            "clean_trailing_fragment": args.clean_trailing_fragment,
            "require_shaped_min_lines": args.require_shaped_min_lines,
        },
        "outputs": {
            "train": str(args.output_train),
            "validation": str(args.output_validation),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser("label", help="Label pre-split parquet sections with OpenAI.")
    label.add_argument("--input", type=Path, default=Path("data/processed/rap_sections_labeled.parquet"))
    label.add_argument("--output-labels", type=Path, default=Path("data/labels/rap_presplit_sections_openai_labels.jsonl"))
    label.add_argument("--model", default=os.environ.get("OPENAI_REVIEW_MODEL", DEFAULT_MODEL))
    label.add_argument("--candidate-mode", choices=["all", "quality_shape", "complete_or_hook"], default="quality_shape")
    label.add_argument("--scan-limit", type=int, default=None)
    label.add_argument("--limit", type=int, default=2000)
    label.add_argument("--read-batch-size", type=int, default=512)
    label.add_argument("--group-by-section", action=argparse.BooleanOptionalAction, default=True)
    label.add_argument("--batch-size", type=int, default=6)
    label.add_argument("--max-chars", type=int, default=1800)
    label.add_argument("--seed", type=int, default=20260628)
    label.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    label.add_argument("--timeout", type=int, default=160)
    label.add_argument("--retries", type=int, default=4)
    label.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    label.add_argument("--dry-run", action="store_true")
    label.set_defaults(func=cmd_label)

    build = subparsers.add_parser("build", help="Build SFT JSONL from pre-split section labels.")
    build.add_argument("--labels", type=Path, default=Path("data/labels/rap_presplit_sections_openai_labels.jsonl"))
    build.add_argument("--output-train", type=Path, default=Path("data/sft/rap_presplit_openai_labeled_sft_train.jsonl"))
    build.add_argument("--output-validation", type=Path, default=Path("data/sft/rap_presplit_openai_labeled_sft_validation.jsonl"))
    build.add_argument("--summary-output", type=Path, default=Path("data/labels/rap_presplit_openai_labeled_sft_summary.json"))
    build.add_argument("--min-quality", type=int, default=3)
    build.add_argument("--strict-complete", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-clean-ending", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-lyric-only", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-strong-structure", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--max-line-words", type=int, default=36)
    build.add_argument("--ascii-punctuation", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--drop-quoted-lines", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--max-quoted-lines", type=int, default=1)
    build.add_argument(
        "--instruction-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rewrite selected assistant payloads into tighter instruction-shaped lyric completions.",
    )
    build.add_argument("--output-max-line-words", type=int, default=30)
    build.add_argument("--output-drop-quoted-lines", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--clean-trailing-fragment", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--require-shaped-min-lines", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument(
        "--disallow-failure-tag",
        action="append",
        default=[],
        help="Failure tag to reject. May be repeated.",
    )
    build.add_argument("--max-verse", type=int, default=5000)
    build.add_argument("--max-hook", type=int, default=2000)
    build.add_argument("--max-control", type=int, default=1500)
    build.add_argument("--max-fragment", type=int, default=750)
    build.add_argument("--max-records", type=int, default=None)
    build.add_argument("--validation-records", type=int, default=500)
    build.add_argument("--seed", type=int, default=20260628)
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
