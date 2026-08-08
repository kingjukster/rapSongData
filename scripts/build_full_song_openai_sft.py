"""Build SFT train/validation data from full-song OpenAI section labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BAD_FAILURE_TAGS = {
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
}

ADLIB_RE = re.compile(
    r"\b(?:yeah|uh|uhh|uh-huh|huh|ha|haha|ooh|oooh|oh|ayy|ay|whoa|woo|whoo)\b",
    re.IGNORECASE,
)
DIALOGUE_MARKER_RE = re.compile(
    r"\b(?:said|says|replied|asks|asked|told me|remarks|speaks|yelled|shouted)\b",
    re.IGNORECASE,
)
UNFINISHED_END_RE = re.compile(r"(?:\(|\[|,|:|;|-|\b(?:and|but|because|cause|when|if|the|a|an|I|you|he|she|we|they))\s*$", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)


def cleaned_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        lines.append(stripped)
    return lines


def stable_text_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def role_bounds(role: str, args: argparse.Namespace) -> tuple[int, int]:
    if role == "hook":
        return args.min_hook_lines, args.max_hook_lines
    if role == "bridge":
        return args.min_bridge_lines, args.max_bridge_lines
    return args.min_verse_lines, args.max_verse_lines


def reject_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    role = str(row.get("section_role") or "")
    text = str(row.get("text") or "").strip()
    lines = cleaned_lines(text)
    word_counts = [len(words(line)) for line in lines]
    min_lines, max_lines = role_bounds(role, args)
    failure_tags = set(str(tag) for tag in row.get("failure_tags") or [])

    if row.get("record_id") in {None, ""}:
        return "missing_record_id"
    if not row.get("keep_for_sft"):
        return "openai_not_keep"
    if role not in set(args.keep_role):
        return "role_not_kept"
    if int(row.get("quality_score") or 0) < args.min_quality:
        return "quality_below_min"
    if int(row.get("control_score") or 0) < args.min_control:
        return "control_below_min"
    if int(row.get("creativity_score") or 0) < args.min_creativity:
        return "creativity_below_min"
    if args.require_clean_ending and not row.get("clean_ending"):
        return "not_clean_ending"
    if args.require_lyric_only and not row.get("lyric_only"):
        return "not_lyric_only"
    if failure_tags & set(args.disallow_failure_tag):
        return "disallowed_failure_tag"
    if not lines:
        return "empty_text"
    if len(lines) < min_lines:
        return "too_few_lines"
    if len(lines) > max_lines:
        return "too_many_lines"
    if any(count > args.max_line_words for count in word_counts):
        return "line_too_long"
    if sum(word_counts) < args.min_total_words:
        return "too_few_words"
    if text.count('"') >= args.max_quote_chars:
        return "too_many_quotes"
    parenthetical_lines = sum(1 for line in lines if "(" in line or ")" in line)
    if parenthetical_lines > args.max_parenthetical_lines:
        return "too_many_parenthetical_lines"
    adlib_lines = sum(1 for line in lines if ADLIB_RE.search(line))
    if adlib_lines > args.max_adlib_lines:
        return "too_many_adlib_lines"
    dialogue_marker_lines = sum(1 for line in lines if DIALOGUE_MARKER_RE.search(line))
    if dialogue_marker_lines > args.max_dialogue_marker_lines:
        return "too_many_dialogue_marker_lines"
    if args.reject_unfinished_final_line and lines and UNFINISHED_END_RE.search(lines[-1]):
        return "unfinished_final_line"
    return None


def prompt_for(row: dict[str, Any], args: argparse.Namespace, *, variant: str = "exact") -> str:
    role = str(row.get("section_role") or "verse")
    theme = str(row.get("theme") or "").strip().rstrip(".")
    line_count = len(cleaned_lines(str(row.get("text") or "")))
    if role == "hook":
        count = min(max(line_count, args.min_hook_lines), args.max_hook_lines)
        topic = theme or "loyalty, pressure, and ambition"
        if variant == "natural_short":
            return (
                f"Write a catchy rap hook about {topic}. Keep it short and memorable. "
                "Return only the hook, 4 to 8 lines."
            )
        return (
            f"Write exactly {count} lines of a catchy rap hook about {topic}. "
            "Return only lyrics, one line per bar. No dialogue, no stage directions, no rambling outro."
        )
    if role == "bridge":
        count = min(max(line_count, args.min_bridge_lines), args.max_bridge_lines)
        topic = theme or "pressure before a breakthrough"
        return (
            f"Write exactly {count} lines of a rap bridge about {topic}. "
            "Return only lyrics, one line per bar. No dialogue, no stage directions, no rambling outro."
        )
    count = min(max(line_count, args.min_verse_lines), args.max_verse_lines)
    topic = theme or "ambition, pressure, and survival"
    return (
        f"Write exactly {count} lines of a rap verse about {topic}. "
        "Return only lyrics, one bar per line. No dialogue, no stage directions, no rambling outro."
    )


def build_record(row: dict[str, Any], args: argparse.Namespace, *, variant: str = "exact") -> dict[str, Any]:
    text = "\n".join(cleaned_lines(str(row.get("text") or "")))
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write only original rap lyrics. Keep line breaks. Do not explain. "
                    "Avoid dialogue, stage directions, excessive ad-libs, and unfinished endings."
                ),
            },
            {"role": "user", "content": prompt_for(row, args, variant=variant)},
            {"role": "assistant", "content": text},
        ],
        "metadata": {
            "sft_source": "raw_full_song_openai_selected_section",
            "prompt_variant": variant,
            "record_id": row.get("record_id"),
            "song_key": row.get("song_key"),
            "title": row.get("title"),
            "artist": row.get("artist"),
            "year": row.get("year"),
            "section_role": row.get("section_role"),
            "line_count": len(cleaned_lines(text)),
            "quality_score": row.get("quality_score"),
            "control_score": row.get("control_score"),
            "creativity_score": row.get("creativity_score"),
            "theme": row.get("theme"),
        },
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def cmd_build(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.labels)
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()

    for row in rows:
        record_id = str(row.get("record_id") or "")
        text_key = stable_text_key(str(row.get("text") or ""))
        reason = None
        if record_id in seen_ids:
            reason = "duplicate_record_id"
        elif text_key in seen_text:
            reason = "duplicate_text"
        else:
            reason = reject_reason(row, args)

        if reason is None:
            accepted.append(row)
            seen_ids.add(record_id)
            seen_text.add(text_key)
        else:
            reject_counts[reason] += 1
            rejected.append({**row, "reject_reason": reason})

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_role[str(row.get("section_role") or "other")].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    caps = {"verse": args.max_verse, "hook": args.max_hook, "bridge": args.max_bridge}
    for role, cap in caps.items():
        bucket = list(by_role.get(role, []))
        rng.shuffle(bucket)
        selected.extend(bucket[:cap])
    rng.shuffle(selected)
    if args.max_records is not None:
        selected = selected[: args.max_records]

    records = []
    for row in selected:
        records.append(build_record(row, args))
        if args.add_hook_short_prompt_variant and str(row.get("section_role") or "") == "hook":
            records.append(build_record(row, args, variant="natural_short"))
    validation_count = min(args.validation_records, max(0, len(records) // 10))
    validation = records[:validation_count]
    train = records[validation_count:]

    write_jsonl(args.output_train, train)
    write_jsonl(args.output_validation, validation)
    write_jsonl(args.output_clean_labels, selected)
    write_jsonl(args.output_rejected, rejected)

    summary = {
        "labels": str(args.labels),
        "outputs": {
            "train": str(args.output_train),
            "validation": str(args.output_validation),
            "clean_labels": str(args.output_clean_labels),
            "rejected": str(args.output_rejected),
            "summary": str(args.summary_output),
        },
        "counts": {
            "input_rows": len(rows),
            "accepted_after_gates": len(accepted),
            "selected_records": len(records),
            "train": len(train),
            "validation": len(validation),
            "rejected": len(rejected),
            "unique_record_ids_seen": len(seen_ids),
            "unique_texts_seen": len(seen_text),
        },
        "roles": {
            "input": dict(Counter(str(row.get("section_role") or "other") for row in rows)),
            "accepted": dict(Counter(str(row.get("section_role") or "other") for row in accepted)),
            "selected": dict(Counter(row["metadata"]["section_role"] for row in records)),
        },
        "prompt_variants": dict(Counter(row["metadata"].get("prompt_variant", "exact") for row in records)),
        "scores": {
            "quality": dict(Counter(str(row.get("quality_score")) for row in accepted)),
            "control": dict(Counter(str(row.get("control_score")) for row in accepted)),
            "creativity": dict(Counter(str(row.get("creativity_score")) for row in accepted)),
        },
        "reject_reasons": dict(reject_counts),
        "criteria": jsonable_args(args),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/raw_full_song_openai_sections_2k.jsonl"))
    parser.add_argument("--output-train", type=Path, default=Path("data/sft/rap_full_song_openai_sft_2k_train.jsonl"))
    parser.add_argument("--output-validation", type=Path, default=Path("data/sft/rap_full_song_openai_sft_2k_validation.jsonl"))
    parser.add_argument("--output-clean-labels", type=Path, default=Path("data/labels/raw_full_song_openai_sections_2k_clean.jsonl"))
    parser.add_argument("--output-rejected", type=Path, default=Path("data/labels/raw_full_song_openai_sections_2k_rejected.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/labels/rap_full_song_openai_sft_2k_summary.json"))
    parser.add_argument("--keep-role", action="append", default=["verse", "hook", "bridge"])
    parser.add_argument("--min-quality", type=int, default=4)
    parser.add_argument("--min-control", type=int, default=4)
    parser.add_argument("--min-creativity", type=int, default=3)
    parser.add_argument("--require-clean-ending", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-lyric-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disallow-failure-tag", action="append", default=sorted(BAD_FAILURE_TAGS))
    parser.add_argument("--min-verse-lines", type=int, default=8)
    parser.add_argument("--max-verse-lines", type=int, default=28)
    parser.add_argument("--min-hook-lines", type=int, default=3)
    parser.add_argument("--max-hook-lines", type=int, default=12)
    parser.add_argument("--min-bridge-lines", type=int, default=4)
    parser.add_argument("--max-bridge-lines", type=int, default=16)
    parser.add_argument("--max-line-words", type=int, default=30)
    parser.add_argument("--min-total-words", type=int, default=20)
    parser.add_argument("--max-quote-chars", type=int, default=10)
    parser.add_argument("--max-parenthetical-lines", type=int, default=2)
    parser.add_argument("--max-adlib-lines", type=int, default=2)
    parser.add_argument("--max-dialogue-marker-lines", type=int, default=0)
    parser.add_argument("--reject-unfinished-final-line", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-hook-short-prompt-variant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-verse", type=int, default=3000)
    parser.add_argument("--max-hook", type=int, default=1200)
    parser.add_argument("--max-bridge", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--validation-records", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cmd_build(args)


if __name__ == "__main__":
    main()
