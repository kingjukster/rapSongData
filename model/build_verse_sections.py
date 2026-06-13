"""Extract verse-only sections from raw rap lyrics with bracket headers."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_SOURCE = Path("data/rap_english_clean_categorized_with_families.parquet")
DEFAULT_OUTPUT = Path("model/data/full_verse_sections.parquet")
HEADER_RE = re.compile(r"^\s*\[(?P<label>[^\]]{1,140})\]\s*$")
BARE_HEADER_RE = re.compile(r"^\s*(?P<label>(?:verse|chorus|hook|intro|outro|bridge|pre[- ]?chorus)[^:\n]{0,120})\s*:\s*$", re.I)
VERSE_RE = re.compile(r"\bverse\b", re.I)
NON_VERSE_RE = re.compile(
    r"\b(?:chorus|hook|intro|outro|bridge|pre[- ]?chorus|post[- ]?chorus|refrain|interlude|skit|sample|spoken|break)\b",
    re.I,
)
ARTIFACT_RE = re.compile(
    r"\b(?:lyrics taken from|lyrics from|genius\.com|you might also like|embed|contributors?)\b|https?://",
    re.I,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
HTML_BREAK_RE = re.compile(r"<\s*br\s*/?\s*>", re.I)
HTML_TAG_RE = re.compile(r"</?[^>\n]{1,80}>")
STAGE_DIRECTION_RE = re.compile(r"\*[^*\n]{1,80}\*")
QUOTE_DIALOGUE_RE = re.compile(r'"[^"\n]{1,160}"')
SECTION_LABEL_RE = re.compile(r"^\s*(?:\[[^\]]{1,140}\]|\{[^}]{1,140}\})\s*$")

try:
    from build_training_data import GENERAL_THEME, infer_theme_info, infer_theme_keywords
except ModuleNotFoundError:
    from model.build_training_data import GENERAL_THEME, infer_theme_info, infer_theme_keywords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-bars", type=int, default=4)
    parser.add_argument("--target-min-bars", type=int, default=10)
    parser.add_argument("--target-max-bars", type=int, default=25)
    parser.add_argument("--max-avg-words-per-bar", type=float, default=18.0)
    parser.add_argument("--hard-max-words-per-bar", type=int, default=40)
    parser.add_argument("--min-words", type=int, default=60)
    parser.add_argument("--max-repeated-line-ratio", type=float, default=0.25)
    parser.add_argument("--max-parenthetical-line-ratio", type=float, default=0.35)
    parser.add_argument("--theme-keyword-count", type=int, default=8)
    return parser.parse_args()


def clean_control_value(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def normalize_line(line: str) -> str:
    line = HTML_BREAK_RE.sub("\n", line)
    line = HTML_TAG_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def is_verse_label(label: str) -> bool:
    return bool(VERSE_RE.search(label)) and not NON_VERSE_RE.search(label)


def should_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("["):
        return True
    if SECTION_LABEL_RE.match(stripped):
        return True
    if ARTIFACT_RE.search(stripped):
        return True
    return False


def repeated_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines if line.strip()]
    if not normalized:
        return 0.0
    repeated = sum(1 for line in normalized if normalized.count(line) > 1)
    return repeated / len(normalized)


def header_label(raw_line: str) -> str | None:
    bracketed = HEADER_RE.match(raw_line)
    if bracketed:
        return clean_control_value(bracketed.group("label"))
    bare = BARE_HEADER_RE.match(raw_line)
    if bare:
        return clean_control_value(bare.group("label"))
    return None


def structural_noise_metrics(lines: list[str]) -> dict[str, Any]:
    parenthetical_lines = 0
    stage_direction_lines = 0
    quoted_dialogue_lines = 0
    html_lines = 0
    section_label_lines = 0
    malformed_control_lines = 0
    for line in lines:
        stripped = line.strip()
        if "(" in stripped and ")" in stripped:
            parenthetical_lines += 1
        if STAGE_DIRECTION_RE.search(stripped):
            stage_direction_lines += 1
        if QUOTE_DIALOGUE_RE.search(stripped):
            quoted_dialogue_lines += 1
        if HTML_TAG_RE.search(stripped):
            html_lines += 1
        if SECTION_LABEL_RE.match(stripped):
            section_label_lines += 1
        if "<|" in stripped or "|>" in stripped:
            malformed_control_lines += 1
    bar_count = len(lines) or 1
    return {
        "parenthetical_line_ratio": round(parenthetical_lines / bar_count, 4),
        "stage_direction_lines": stage_direction_lines,
        "quoted_dialogue_lines": quoted_dialogue_lines,
        "html_lines": html_lines,
        "section_label_lines": section_label_lines,
        "malformed_control_lines": malformed_control_lines,
    }


def extract_verse_sections(lyrics: Any) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_label = ""
    current_is_verse = False
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if not current_is_verse or not current_lines:
            current_lines = []
            return
        lines = [line for line in current_lines if line.strip()]
        if lines:
            sections.append({"section_label": current_label, "lines": lines})
        current_lines = []

    normalized_text = HTML_BREAK_RE.sub("\n", str(lyrics or "")).replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized_text.split("\n"):
        label = header_label(raw_line)
        if label is not None:
            flush()
            current_label = label
            current_is_verse = is_verse_label(current_label)
            continue
        if not current_is_verse:
            continue
        line = normalize_line(raw_line)
        if not line or should_skip_line(line):
            continue
        if HEADER_RE.match(line):
            continue
        current_lines.append(line)

    flush()
    return sections


def quality_flags(
    lines: list[str],
    *,
    min_bars: int,
    target_min_bars: int,
    target_max_bars: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    min_words: int,
    max_repeated_line_ratio: float,
    max_parenthetical_line_ratio: float,
) -> dict[str, Any]:
    counts = [line_word_count(line) for line in lines]
    bar_count = len(lines)
    word_count = sum(counts)
    avg_words = word_count / bar_count if bar_count else 0.0
    max_words = max(counts, default=0)
    repeat_ratio = repeated_line_ratio(lines)
    noise = structural_noise_metrics(lines)
    is_training_length = target_min_bars <= bar_count <= target_max_bars
    passes_quality_filter = (
        is_training_length
        and word_count >= min_words
        and avg_words <= max_avg_words_per_bar
        and max_words <= hard_max_words_per_bar
        and repeat_ratio <= max_repeated_line_ratio
        and noise["parenthetical_line_ratio"] <= max_parenthetical_line_ratio
        and noise["stage_direction_lines"] == 0
        and noise["quoted_dialogue_lines"] == 0
        and noise["html_lines"] == 0
        and noise["section_label_lines"] == 0
        and noise["malformed_control_lines"] == 0
    )
    return {
        "bar_count": bar_count,
        "word_count": word_count,
        "avg_words_per_bar": round(avg_words, 4),
        "max_words_per_bar": max_words,
        "repeated_line_ratio": round(repeat_ratio, 4),
        "is_training_length": is_training_length,
        "is_full_verse": True,
        "passes_quality_filter": passes_quality_filter,
        **noise,
    }


def build_records(df: pl.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        sections = extract_verse_sections(row.get("lyrics"))
        for section_index, section in enumerate(sections, start=1):
            lines = section["lines"]
            flags = quality_flags(
                lines,
                min_bars=args.min_bars,
                target_min_bars=args.target_min_bars,
                target_max_bars=args.target_max_bars,
                max_avg_words_per_bar=args.max_avg_words_per_bar,
                hard_max_words_per_bar=args.hard_max_words_per_bar,
                min_words=args.min_words,
                max_repeated_line_ratio=args.max_repeated_line_ratio,
                max_parenthetical_line_ratio=args.max_parenthetical_line_ratio,
            )
            if not flags["passes_quality_filter"]:
                continue
            views = int(row.get("views") or 0)
            verse_text = "\n".join(lines)
            theme_info = infer_theme_info(verse_text)
            theme = theme_info["theme"]
            theme_keywords = infer_theme_keywords(verse_text, theme, max_keywords=args.theme_keyword_count)
            records.append(
                {
                    "title": clean_control_value(row.get("title")),
                    "artist": clean_control_value(row.get("artist")),
                    "artist_clean": clean_control_value(row.get("artist_clean") or row.get("artist")),
                    "year": int(row.get("year") or 0),
                    "views": views,
                    "log_views": round(math.log1p(max(0, views)), 4),
                    "rap_category": clean_control_value(row.get("rap_category")),
                    "rap_family": clean_control_value(row.get("rap_family")),
                    "section_label": clean_control_value(section["section_label"]),
                    "section_index": section_index,
                    "verse_text": verse_text,
                    "theme": theme,
                    "theme_confidence": theme_info["theme_confidence"],
                    "theme_score": theme_info["theme_score"],
                    "theme_margin": theme_info["theme_margin"],
                    "theme_keywords": theme_keywords,
                    "theme_is_general": theme == GENERAL_THEME,
                    **flags,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Verse section source not found: {args.source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")

    columns = [
        "title",
        "artist",
        "artist_clean",
        "year",
        "views",
        "lyrics",
        "rap_category",
        "rap_family",
    ]
    df = pl.read_parquet(args.source, columns=columns)
    records = build_records(df, args)
    out_df = pl.DataFrame(records)
    out_df.write_parquet(args.output)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.source),
        "output": str(args.output),
        "total_verse_sections": len(records),
        "training_length_sections": int(out_df.get_column("is_training_length").sum()) if len(records) else 0,
        "target_min_bars": args.target_min_bars,
        "target_max_bars": args.target_max_bars,
        "min_words": args.min_words,
        "max_repeated_line_ratio": args.max_repeated_line_ratio,
        "max_parenthetical_line_ratio": args.max_parenthetical_line_ratio,
        "theme_counts": dict(out_df.get_column("theme").value_counts().rows()) if len(records) else {},
        "theme_confidence_counts": dict(out_df.get_column("theme_confidence").value_counts().rows()) if len(records) else {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote verse sections: {args.output}")


if __name__ == "__main__":
    main()
