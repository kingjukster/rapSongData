"""Clean, audit, label, and report the categorized rap lyric corpus.

The module keeps raw inputs untouched. It writes cleaned records, review queues,
quarantine/drop queues, train text, and reports that make corpus quality visible
before the next local QLoRA run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("data/rap_english_clean_categorized_with_families.parquet")
DEFAULT_OUTPUT_DIR = Path("data/cleaned")
DEFAULT_REPORT_DIR = Path("reports")
SUMMARY_FILENAME = "corpus_cleaning_summary.json"
REPORT_FILENAME = "corpus_cleaning_report.md"
TRAIN_TXT_FILENAME = "categorized_rap_corpus_train.txt"
VALIDATION_TXT_FILENAME = "categorized_rap_corpus_validation.txt"

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[a-z][^>]{0,200}>", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
PROMPT_LEFTOVER_RE = re.compile(r"^\s*(prompt|output|assistant|user|system)\s*:", re.IGNORECASE | re.MULTILINE)
JSON_CODE_RE = re.compile(r"(```|^\s*[{}]\s*$|^\s*\"[A-Za-z0-9_ -]{1,60}\"\s*:)", re.MULTILINE)
MOJIBAKE_RE = re.compile(r"(?:\u00c3[\u0080-\u00bf]|\u00c2[\u0080-\u00bf]|\u00e2[\u0080-\u00bf]{1,2}|\\u00[0-9a-fA-F]{2}|\\x[0-9a-fA-F]{2}|\ufffd)")
LYRIC_LINE_KEY_RE = re.compile(r"[^a-z0-9']+")

SECTION_TAG_RE = re.compile(
    r"^\s*[\[\(\{]?\s*"
    r"(intro|verse|chorus|hook|bridge|outro)"
    r"(?:\s+(?:\d+|[ivx]+))?"
    r"(?:\s*[:\-].*)?"
    r"\s*[\]\)\}]?\s*$",
    re.IGNORECASE,
)
SECTION_CANONICAL = {
    "intro": "[INTRO]",
    "verse": "[VERSE]",
    "chorus": "[HOOK]",
    "hook": "[HOOK]",
    "bridge": "[BRIDGE]",
    "outro": "[OUTRO]",
}

JUNK_LINE_RE = re.compile(
    r"^\s*(?:"
    r"embed|you might also like|genius annotation|lyrics|contributors?|"
    r"translations?|translated by|read more|see .* live|get tickets|"
    r"advertisement|ad choices|cookie policy|privacy policy|terms of use|"
    r"share|tweet|facebook|instagram|tiktok|youtube|soundcloud|spotify|"
    r"home|songs|albums|artists|charts|news|videos|page \d+|"
    r"all rights reserved|copyright|contact us|about genius|"
    r"produced by|written by|release date"
    r")\b",
    re.IGNORECASE,
)
PAGE_TITLE_RE = re.compile(r"\blyrics\s*(?:\||-|by)\s*(?:genius|azlyrics|metrolyrics)\b", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\s*$")
TRAILING_EMBED_RE = re.compile(r"\s*\d*\s*embed\s*$", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
PROSE_SENTENCE_RE = re.compile(r"[.!?][\"')\]]?(?:\s+|$)")

MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
    "\u00c2\u00a0": " ",
    "\u00c2": "",
    "\ufffd": "",
}

ENGLISH_STOPWORDS = {
    "the",
    "and",
    "you",
    "that",
    "with",
    "for",
    "this",
    "not",
    "but",
    "all",
    "like",
    "just",
    "when",
    "they",
    "have",
    "from",
    "your",
    "what",
    "out",
    "get",
    "got",
    "know",
    "ain",
    "dont",
    "cant",
    "im",
    "ive",
    "we",
    "my",
    "me",
    "in",
    "on",
    "to",
    "of",
}

SERIOUS_FLAGS = {
    "empty_lyrics",
    "exact_duplicate",
    "near_duplicate",
    "repeated_line_spam",
    "mostly_symbol_or_number_text",
    "non_english_language",
    "html_json_markdown_or_code_remnant",
    "prompt_output_chat_leftover",
}


@dataclass
class CorpusCleaningConfig:
    source_path: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    audit_only: bool = False
    write_report: bool = True
    include_bronze: bool = False
    min_quality: float = 0.70
    dedupe: str = "both"
    near_duplicate_threshold: float = 0.90
    validation_ratio: float = 0.01
    sample_limit: int = 25
    progress_every: int = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--clean", action="store_true", help="Compatibility flag; cleaning is the default action.")
    parser.add_argument("--audit-only", action="store_true", help="Write audit artifacts but skip train text output.")
    parser.add_argument("--write-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-bronze", action="store_true")
    parser.add_argument("--min-quality", type=float, default=0.70)
    parser.add_argument("--dedupe", choices=["exact", "near", "both"], default="both")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CorpusCleaningConfig:
    return CorpusCleaningConfig(
        source_path=args.source,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        audit_only=args.audit_only,
        write_report=args.write_report,
        include_bronze=args.include_bronze,
        min_quality=args.min_quality,
        dedupe=args.dedupe,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_source_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus source not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if stripped:
                    row = json.loads(stripped)
                    row.setdefault("_source_line", line_number)
                    records.append(row)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"JSON corpus source must be a list of objects: {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    if suffix == ".parquet":
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - exercised in local envs with polars.
            raise SystemExit("Reading Parquet requires polars. Use .venv or install requirements-local-cuda.txt.") from exc
        return pl.read_parquet(path).to_dicts()
    raise ValueError(f"Unsupported corpus source format: {path.suffix}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def bad_sequence_score(text: str) -> int:
    return len(MOJIBAKE_RE.findall(text)) + text.count("\ufffd")


def repair_mojibake(text: str) -> tuple[str, bool]:
    repaired = text
    changed = False
    try:
        candidate = text.encode("latin1").decode("utf-8")
        if bad_sequence_score(candidate) < bad_sequence_score(text):
            repaired = candidate
            changed = True
    except UnicodeError:
        pass
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        if bad in repaired:
            repaired = repaired.replace(bad, good)
            changed = True
    return repaired, changed


def normalize_unicode_text(value: Any) -> tuple[str, list[str]]:
    flags: list[str] = []
    if value is None:
        return "", ["empty_source_text"]
    text = str(value)
    unescaped = html.unescape(text)
    if unescaped != text:
        flags.append("html_entity_decoded")
        text = unescaped
    if bad_sequence_score(text):
        flags.append("mojibake_or_replacement_char")
    text, repaired = repair_mojibake(text)
    if repaired:
        flags.append("mojibake_repaired")
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        flags.append("unicode_nfkc_normalized")
    text = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if ZERO_WIDTH_RE.search(text):
        flags.append("zero_width_or_bom_removed")
        text = ZERO_WIDTH_RE.sub("", text)
    if "\ufffd" in text:
        flags.append("replacement_char_removed")
        text = text.replace("\ufffd", "")
    if CONTROL_RE.search(text):
        flags.append("control_char_removed")
        text = CONTROL_RE.sub("", text)
    return text, flags


def canonical_section_tag(line: str) -> str | None:
    match = SECTION_TAG_RE.match(line)
    if not match:
        return None
    return SECTION_CANONICAL[match.group(1).lower()]


def clean_lyrics_text(value: Any) -> tuple[str, list[str]]:
    text, flags = normalize_unicode_text(value)
    kept: list[str] = []
    last_tag = ""
    blank_seen = False
    removed_junk = 0
    removed_urls = 0

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            if kept and not blank_seen:
                kept.append("")
            blank_seen = True
            continue
        blank_seen = False

        line = TRAILING_EMBED_RE.sub("", line).strip()
        if not line:
            removed_junk += 1
            continue
        if URL_RE.fullmatch(line):
            removed_urls += 1
            continue
        if TIMESTAMP_RE.fullmatch(line) or JUNK_LINE_RE.match(line) or PAGE_TITLE_RE.search(line):
            removed_junk += 1
            continue
        if HTML_TAG_RE.search(line):
            flags.append("html_tag_removed")
            line = HTML_TAG_RE.sub(" ", line)
        if URL_RE.search(line):
            removed_urls += len(URL_RE.findall(line))
            line = URL_RE.sub("", line).strip()
        if not line:
            continue

        tag = canonical_section_tag(line)
        if tag:
            if tag == last_tag:
                flags.append("duplicate_adjacent_section_tag_removed")
                continue
            kept.append(tag)
            last_tag = tag
            continue

        last_tag = ""
        if MARKDOWN_HEADING_RE.match(line):
            removed_junk += 1
            continue
        line = re.sub(r" {2,}", " ", line).strip()
        if line:
            kept.append(line)

    while kept and kept[-1] == "":
        kept.pop()
    while kept and kept[0] == "":
        kept.pop(0)

    if removed_junk:
        flags.append("scrape_junk_removed")
    if removed_urls:
        flags.append("url_removed")
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, sorted(set(flags))


def lyric_line_key(line: str) -> str:
    lowered = unicodedata.normalize("NFKC", line).lower()
    lowered = re.sub(r"^\[[a-z]+\]$", "", lowered)
    return LYRIC_LINE_KEY_RE.sub(" ", lowered).strip()


def normalized_text_key(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).lower()
    lowered = re.sub(r"\[[a-z]+\]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9'\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def stable_hash(value: str, digest_size: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=digest_size).hexdigest()


def stable_hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=8).digest(), "big")


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or unicodedata.category(char) == "So" and codepoint > 0x7F
    )


def classify_language(text: str, row: dict[str, Any] | None = None) -> str:
    row = row or {}
    labels = [
        str(row.get(column) or "").strip().lower()
        for column in ("language", "language_cld3", "language_ft", "language_label")
    ]
    labels = [label for label in labels if label and label not in {"none", "null", "nan"}]
    english_labels = {"en", "eng", "english", "en-us", "en-gb"}
    has_english = any(label in english_labels or label.startswith("en-") for label in labels)
    has_other = any(label not in english_labels and not label.startswith("en-") for label in labels)
    if has_english and not has_other:
        return "en"
    if has_english and has_other:
        return "mixed"
    if has_other:
        return "non_en"

    tokens = [token.lower().replace("'", "") for token in word_tokens(text)]
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unknown"
    ascii_letters = sum(1 for char in letters if char.isascii())
    ascii_ratio = safe_ratio(ascii_letters, len(letters))
    if not tokens:
        return "non_en" if ascii_ratio < 0.50 else "unknown"
    stopword_hits = sum(1 for token in tokens if token in ENGLISH_STOPWORDS)
    stopword_ratio = safe_ratio(stopword_hits, len(tokens))
    if ascii_ratio >= 0.90 and (stopword_hits >= 2 or stopword_ratio >= 0.04):
        return "en"
    if ascii_ratio >= 0.55:
        return "mixed"
    return "non_en"


def calculate_metrics(text: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    visible_chars = [char for char in text if not char.isspace()]
    lines = [line for line in text.splitlines() if line.strip()]
    line_keys = [lyric_line_key(line) for line in lines if lyric_line_key(line)]
    line_counter = Counter(line_keys)
    repeated_lines = sum(count for count in line_counter.values() if count > 1)
    words = word_tokens(text)
    alpha = sum(1 for char in visible_chars if char.isalpha())
    punctuation = sum(1 for char in visible_chars if unicodedata.category(char).startswith("P") or char in string.punctuation)
    digits = sum(1 for char in visible_chars if char.isdigit())
    non_ascii = sum(1 for char in visible_chars if ord(char) > 127)
    line_lengths = [len(line) for line in lines]
    return {
        "line_count": len(lines),
        "char_count": len(text),
        "estimated_token_count": int(math.ceil(max(len(words) * 1.3, len(text) / 4.0))),
        "alpha_ratio": round(safe_ratio(alpha, len(visible_chars)), 4),
        "punctuation_ratio": round(safe_ratio(punctuation, len(visible_chars)), 4),
        "digit_ratio": round(safe_ratio(digits, len(visible_chars)), 4),
        "non_ascii_ratio": round(safe_ratio(non_ascii, len(visible_chars)), 4),
        "repeated_line_ratio": round(safe_ratio(repeated_lines, len(line_keys)), 4),
        "unique_line_ratio": round(safe_ratio(len(set(line_keys)), len(line_keys)), 4),
        "avg_line_length": round(safe_ratio(sum(line_lengths), len(line_lengths)), 2),
        "max_line_length": max(line_lengths, default=0),
        "word_count": len(words),
        "emoji_count": sum(1 for char in text if is_emoji(char)),
        "language_label": classify_language(text, row),
    }


def artifact_flags(text: str, metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not text.strip():
        return ["empty_lyrics"]
    if MOJIBAKE_RE.search(text):
        flags.append("weird_unicode_or_mojibake")
    if URL_RE.search(text):
        flags.append("url_remnant")
    if HTML_TAG_RE.search(text) or JSON_CODE_RE.search(text):
        flags.append("html_json_markdown_or_code_remnant")
    if PROMPT_LEFTOVER_RE.search(text):
        flags.append("prompt_output_chat_leftover")
    if metrics["emoji_count"] >= 8 or safe_ratio(metrics["emoji_count"], max(metrics["char_count"], 1)) > 0.02:
        flags.append("excessive_emoji")
    if metrics["line_count"] <= 3 and metrics["char_count"] > 350:
        flags.append("non_lyric_metadata_or_prose")
    if metrics["avg_line_length"] > 150 or metrics["max_line_length"] > 420:
        flags.append("non_lyric_metadata_or_prose")
    sentence_count = len(PROSE_SENTENCE_RE.findall(text))
    if metrics["line_count"] and sentence_count / max(metrics["line_count"], 1) > 1.5 and metrics["avg_line_length"] > 90:
        flags.append("non_lyric_metadata_or_prose")
    if metrics["alpha_ratio"] < 0.45 and (metrics["digit_ratio"] + metrics["punctuation_ratio"]) > 0.45:
        flags.append("mostly_symbol_or_number_text")
    if metrics["language_label"] == "non_en":
        flags.append("non_english_language")
    elif metrics["language_label"] == "mixed":
        flags.append("mixed_language")
    return sorted(set(flags))


def source_text_for_row(row: dict[str, Any]) -> str:
    for column in ("lyrics", "lyrics_raw", "text", "lyrics_clean", "lyrics_model_text"):
        value = row.get(column)
        if value:
            return str(value)
    return ""


def clean_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    raw_text = source_text_for_row(row)
    cleaned_text, cleaning_flags = clean_lyrics_text(raw_text)
    metrics = calculate_metrics(cleaned_text, row)
    flags = sorted(set(cleaning_flags + artifact_flags(cleaned_text, metrics)))
    title = str(row.get("title") or "").strip()
    artist = str(row.get("artist_clean") or row.get("artist") or "").strip()
    record_id = str(row.get("id") or stable_hash(f"{artist}|{title}|{index}|{raw_text}", digest_size=10))
    normalized_key = normalized_text_key(cleaned_text)
    record = {
        "record_id": record_id,
        "source_index": index,
        "title": title,
        "artist": str(row.get("artist") or "").strip(),
        "artist_clean": artist,
        "year": row.get("year"),
        "views": row.get("views"),
        "rap_category": str(row.get("rap_category") or "").strip(),
        "rap_family": str(row.get("rap_family") or "").strip(),
        "lyrics_cleaned": cleaned_text,
        "normalized_text_hash": stable_hash(normalized_key) if normalized_key else "",
        "duplicate_status": "unique",
        "duplicate_of": None,
        "duplicate_type": None,
        "duplicate_score": 0.0,
        "flags": flags,
        "quality_score": 0.0,
        "quality_tier": "drop",
        "decision": "drop",
        "sample_weight": 0.0,
        "included_in_train": False,
        "metrics": metrics,
        "raw_char_count": len(raw_text),
        "before_excerpt": raw_text[:1200],
        "after_excerpt": cleaned_text[:1200],
    }
    return record


def simhash(tokens: list[str]) -> int:
    vector = [0] * 64
    if not tokens:
        return 0
    token_step = max(1, len(tokens) // 384)
    units = tokens[::token_step]
    if len(tokens) >= 2:
        bigram_step = max(1, len(tokens) // 192)
        units = units + [f"{tokens[index]} {tokens[index + 1]}" for index in range(0, len(tokens) - 1, bigram_step)]
    for token in units:
        hashed = stable_hash64(token)
        for bit in range(64):
            vector[bit] += 1 if hashed & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mark_duplicates(records: list[dict[str, Any]], config: CorpusCleaningConfig) -> None:
    first_by_hash: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        if config.progress_every and index % config.progress_every == 0:
            print(f"[cleaning] exact duplicate scan: {index}/{len(records)}", flush=True)
        text_hash = record["normalized_text_hash"]
        if not text_hash:
            continue
        original = first_by_hash.get(text_hash)
        if original is None:
            first_by_hash[text_hash] = record
            continue
        record["duplicate_status"] = "duplicate"
        record["duplicate_of"] = original["record_id"]
        record["duplicate_type"] = "exact"
        record["duplicate_score"] = 1.0
        record["flags"] = sorted(set(record["flags"] + ["exact_duplicate"]))

    if config.dedupe not in {"near", "both"}:
        return

    hashes: list[int] = []
    token_sets: list[set[str]] = []
    for index, record in enumerate(records, start=1):
        if config.progress_every and index % config.progress_every == 0:
            print(f"[cleaning] near duplicate hash prep: {index}/{len(records)}", flush=True)
        key = normalized_text_key(record["lyrics_cleaned"])
        tokens = word_tokens(key)
        hashes.append(simhash(tokens))
        token_sets.append(set(tokens))

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    max_distance = max(2, int(round((1.0 - config.near_duplicate_threshold) * 64)))
    for index, record in enumerate(records):
        if config.progress_every and index and index % config.progress_every == 0:
            print(f"[cleaning] near duplicate scan: {index}/{len(records)}", flush=True)
        if record["duplicate_type"] == "exact" or record["metrics"]["char_count"] < 300:
            continue
        candidates: set[int] = set()
        current_hash = hashes[index]
        for band in range(4):
            band_key = (band, (current_hash >> (band * 16)) & 0xFFFF)
            candidates.update(buckets[band_key][-80:])
        best: tuple[float, int] | None = None
        for candidate_index in candidates:
            candidate = records[candidate_index]
            if candidate["duplicate_type"] == "exact":
                continue
            distance = hamming_distance(current_hash, hashes[candidate_index])
            simhash_score = 1.0 - distance / 64.0
            token_score = jaccard(token_sets[index], token_sets[candidate_index])
            score = max(simhash_score, token_score)
            if distance <= max_distance or token_score >= 0.82:
                if best is None or score > best[0]:
                    best = (score, candidate_index)
        if best:
            score, candidate_index = best
            record["duplicate_status"] = "duplicate"
            record["duplicate_of"] = records[candidate_index]["record_id"]
            record["duplicate_type"] = "near"
            record["duplicate_score"] = round(score, 4)
            record["flags"] = sorted(set(record["flags"] + ["near_duplicate"]))
        for band in range(4):
            band_key = (band, (current_hash >> (band * 16)) & 0xFFFF)
            buckets[band_key].append(index)


def assign_quality(record: dict[str, Any], config: CorpusCleaningConfig) -> None:
    metrics = record["metrics"]
    flags = set(record["flags"])
    score = 1.0

    if record["duplicate_type"] in {"exact", "near"}:
        score = 0.0
    if not record["lyrics_cleaned"].strip():
        flags.add("empty_lyrics")
        score = 0.0
    if metrics["char_count"] < 180 or metrics["line_count"] < 4:
        flags.add("too_short_fragment")
        score -= 0.55
    elif metrics["char_count"] < 500 or metrics["line_count"] < 8:
        flags.add("short_fragment")
        score -= 0.25
    if metrics["char_count"] > 12000 or metrics["line_count"] > 220:
        flags.add("length_outlier")
        score -= 0.20
    if metrics["repeated_line_ratio"] > 0.55:
        flags.add("repeated_line_spam")
        score = min(score, 0.20)
    elif metrics["repeated_line_ratio"] > 0.30:
        flags.add("high_repeated_line_ratio")
        score -= 0.25
    if metrics["unique_line_ratio"] and metrics["unique_line_ratio"] < 0.45:
        flags.add("low_unique_line_ratio")
        score -= 0.20
    if "mixed_language" in flags:
        score -= 0.25
    if "non_english_language" in flags:
        score = min(score, 0.25)
    if "mostly_symbol_or_number_text" in flags:
        score = min(score, 0.25)
    if "non_lyric_metadata_or_prose" in flags:
        score -= 0.25
    if "html_json_markdown_or_code_remnant" in flags or "prompt_output_chat_leftover" in flags:
        score -= 0.30
    if "weird_unicode_or_mojibake" in flags:
        score -= 0.20
    if "excessive_emoji" in flags:
        score -= 0.25

    score = max(0.0, min(1.0, score))
    if score >= 0.90 and not (flags & SERIOUS_FLAGS):
        tier = "gold"
    elif score >= config.min_quality and not (flags & {"non_english_language", "repeated_line_spam"}):
        tier = "silver"
    elif score >= 0.50:
        tier = "bronze"
    elif score >= 0.30:
        tier = "review"
    elif score >= 0.10:
        tier = "quarantine"
    else:
        tier = "drop"

    if record["duplicate_type"] in {"exact", "near"}:
        tier = "drop"

    if tier in {"gold", "silver", "bronze"}:
        decision = "keep"
    elif tier == "review":
        decision = "review"
    elif tier == "quarantine":
        decision = "quarantine"
    else:
        decision = "drop"

    record["flags"] = sorted(flags)
    record["quality_score"] = round(score, 4)
    record["quality_tier"] = tier
    record["decision"] = decision
    record["sample_weight"] = {"gold": 1.0, "silver": 0.75, "bronze": 0.35}.get(tier, 0.0)
    record["included_in_train"] = decision == "keep" and (
        tier in {"gold", "silver"} or (tier == "bronze" and config.include_bronze)
    )


def audit_records(source_rows: list[dict[str, Any]], config: CorpusCleaningConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(source_rows)
    for index, row in enumerate(source_rows, start=1):
        records.append(clean_record(row, index))
        if config.progress_every and index % config.progress_every == 0:
            print(f"[cleaning] cleaned records: {index}/{total}", flush=True)
    print(f"[cleaning] cleaned records: {len(records)}/{total}", flush=True)
    print("[cleaning] duplicate audit started", flush=True)
    mark_duplicates(records, config)
    print("[cleaning] quality scoring started", flush=True)
    for record in records:
        assign_quality(record, config)
    print("[cleaning] quality scoring complete", flush=True)
    return records


def compact_record(record: dict[str, Any], include_excerpts: bool = False) -> dict[str, Any]:
    keys = [
        "record_id",
        "source_index",
        "title",
        "artist",
        "artist_clean",
        "year",
        "views",
        "rap_category",
        "rap_family",
        "lyrics_cleaned",
        "normalized_text_hash",
        "duplicate_status",
        "duplicate_of",
        "duplicate_type",
        "duplicate_score",
        "flags",
        "quality_score",
        "quality_tier",
        "decision",
        "sample_weight",
        "included_in_train",
        "metrics",
        "raw_char_count",
    ]
    payload = {key: record.get(key) for key in keys}
    if include_excerpts:
        payload["before_excerpt"] = record.get("before_excerpt", "")
        payload["after_excerpt"] = record.get("after_excerpt", "")
    return payload


def training_text(record: dict[str, Any]) -> str:
    metadata_lines = [
        "<|task|>generate_song",
        f"<|title|>{record.get('title') or ''}",
        f"<|artist|>{record.get('artist_clean') or record.get('artist') or ''}",
        f"<|rap_family|>{record.get('rap_family') or ''}",
        f"<|rap_category|>{record.get('rap_category') or ''}",
        f"<|quality_tier|>{record.get('quality_tier') or ''}",
        f"<|quality_score|>{record.get('quality_score') or 0}",
        "<|lyrics|>",
        record["lyrics_cleaned"],
        "<|end|>",
    ]
    return "\n".join(metadata_lines).strip()


def split_train_validation(records: list[dict[str, Any]], validation_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included = [record for record in records if record["included_in_train"]]
    if not included:
        return [], []
    ordered = sorted(included, key=lambda record: stable_hash(str(record["record_id"])))
    if len(ordered) == 1:
        return ordered, ordered
    validation_count = max(1, int(round(len(ordered) * validation_ratio)))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = ordered[:validation_count]
    train = ordered[validation_count:]
    return train, validation


def write_text_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(training_text(record))
            file.write("\n\n")


def aggregate_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "unknown") for record in records))


def summarize_by(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(group_key) or "unknown")].append(record)
    rows: list[dict[str, Any]] = []
    for group, group_records in sorted(groups.items()):
        tier_counts = Counter(record["quality_tier"] for record in group_records)
        decision_counts = Counter(record["decision"] for record in group_records)
        rows.append(
            {
                group_key: group,
                "record_count": len(group_records),
                "train_records": sum(1 for record in group_records if record["included_in_train"]),
                "avg_quality_score": round(sum(record["quality_score"] for record in group_records) / len(group_records), 4),
                "gold": tier_counts.get("gold", 0),
                "silver": tier_counts.get("silver", 0),
                "bronze": tier_counts.get("bronze", 0),
                "review": decision_counts.get("review", 0),
                "quarantine": decision_counts.get("quarantine", 0),
                "drop": decision_counts.get("drop", 0),
            }
        )
    rows.sort(key=lambda row: row["record_count"], reverse=True)
    return rows


def duplicate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record["duplicate_type"]:
            rows.append(
                {
                    "record_id": record["record_id"],
                    "duplicate_of": record["duplicate_of"],
                    "duplicate_type": record["duplicate_type"],
                    "duplicate_score": record["duplicate_score"],
                    "title": record["title"],
                    "artist_clean": record["artist_clean"],
                    "quality_tier": record["quality_tier"],
                    "decision": record["decision"],
                }
            )
    return rows


def report_samples(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    interesting = [
        record
        for record in records
        if record["flags"] or record.get("before_excerpt", "") != record.get("after_excerpt", "")
    ]
    interesting.sort(key=lambda record: (record["decision"] != "keep", -len(record["flags"]), record["source_index"]), reverse=True)
    samples = []
    for record in interesting[:limit]:
        samples.append(
            {
                "record_id": record["record_id"],
                "title": record["title"],
                "artist_clean": record["artist_clean"],
                "quality_tier": record["quality_tier"],
                "decision": record["decision"],
                "flags": record["flags"],
                "before_excerpt": record.get("before_excerpt", ""),
                "after_excerpt": record.get("after_excerpt", ""),
            }
        )
    return samples


def build_summary(records: list[dict[str, Any]], config: CorpusCleaningConfig, paths: dict[str, str | None]) -> dict[str, Any]:
    train_records, validation_records = split_train_validation(records, config.validation_ratio)
    flag_counts = Counter(flag for record in records for flag in record["flags"])
    duplicate_counts = Counter(record["duplicate_type"] or "unique" for record in records)
    quality_scores = [record["quality_score"] for record in records]
    train_token_estimate = sum(record["metrics"]["estimated_token_count"] for record in train_records)
    return {
        "generated_at": utc_now(),
        "source_path": str(config.source_path),
        "output_dir": str(config.output_dir),
        "report_dir": str(config.report_dir),
        "audit_only": config.audit_only,
        "config": {
            "include_bronze": config.include_bronze,
            "min_quality": config.min_quality,
            "dedupe": config.dedupe,
            "near_duplicate_threshold": config.near_duplicate_threshold,
            "validation_ratio": config.validation_ratio,
        },
        "counts": {
            "input_records": len(records),
            "cleaned_records": len(records),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "estimated_train_tokens": train_token_estimate,
            "tiers": aggregate_counts(records, "quality_tier"),
            "decisions": aggregate_counts(records, "decision"),
            "duplicates": dict(duplicate_counts),
            "top_flags": dict(flag_counts.most_common(30)),
        },
        "quality": {
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
            "min_quality_score": min(quality_scores) if quality_scores else 0.0,
            "max_quality_score": max(quality_scores) if quality_scores else 0.0,
            "included_quality_tiers": ["gold", "silver"] + (["bronze"] if config.include_bronze else []),
        },
        "paths": paths,
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    lines = [
        "# Corpus Cleaning Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Source: `{summary['source_path']}`",
        f"Output: `{summary['output_dir']}`",
        "",
        "## Counts",
        "",
        f"- Input records: {counts['input_records']}",
        f"- Train records: {counts['train_records']}",
        f"- Validation records: {counts['validation_records']}",
        f"- Estimated train tokens: {counts['estimated_train_tokens']}",
        "",
        "## Quality Tiers",
        "",
    ]
    for tier, count in counts["tiers"].items():
        lines.append(f"- {tier}: {count}")
    lines.extend(["", "## Decisions", ""])
    for decision, count in counts["decisions"].items():
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## Duplicate Detection", ""])
    for duplicate_type, count in counts["duplicates"].items():
        lines.append(f"- {duplicate_type}: {count}")
    lines.extend(["", "## Top Flags", ""])
    for flag, count in counts["top_flags"].items():
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Training Policy", ""])
    lines.append("- Default training includes gold and silver records only.")
    lines.append("- Bronze records are included only when `--include-bronze` is used.")
    lines.append("- Review, quarantine, and drop records are excluded from training text.")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    for key, value in summary["paths"].items():
        if value:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(records: list[dict[str, Any]], config: CorpusCleaningConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    cleaned_jsonl = config.output_dir / "categorized_rap_corpus_cleaned.jsonl"
    train_jsonl = config.output_dir / "categorized_rap_corpus_train.jsonl"
    review_jsonl = config.output_dir / "categorized_rap_corpus_review.jsonl"
    quarantine_jsonl = config.output_dir / "categorized_rap_corpus_quarantine.jsonl"
    dropped_jsonl = config.output_dir / "categorized_rap_corpus_dropped.jsonl"
    train_txt = config.output_dir / TRAIN_TXT_FILENAME
    validation_txt = config.output_dir / VALIDATION_TXT_FILENAME
    summary_json = config.output_dir / SUMMARY_FILENAME
    report_summary_json = config.report_dir / SUMMARY_FILENAME
    markdown_report = config.report_dir / REPORT_FILENAME
    family_csv = config.report_dir / "corpus_quality_by_rap_family.csv"
    category_csv = config.report_dir / "corpus_quality_by_rap_category.csv"
    duplicate_csv = config.report_dir / "corpus_duplicates.csv"
    samples_jsonl = config.report_dir / "corpus_before_after_samples.jsonl"

    train_records, validation_records = split_train_validation(records, config.validation_ratio)
    included_records = [record for record in records if record["included_in_train"]]

    write_jsonl(cleaned_jsonl, (compact_record(record) for record in records))
    write_jsonl(train_jsonl, (compact_record(record) for record in included_records))
    write_jsonl(review_jsonl, (compact_record(record, include_excerpts=True) for record in records if record["decision"] == "review"))
    write_jsonl(
        quarantine_jsonl,
        (compact_record(record, include_excerpts=True) for record in records if record["decision"] == "quarantine"),
    )
    write_jsonl(dropped_jsonl, (compact_record(record, include_excerpts=True) for record in records if record["decision"] == "drop"))

    if not config.audit_only:
        write_text_dataset(train_txt, train_records)
        write_text_dataset(validation_txt, validation_records)

    paths = {
        "cleaned_jsonl": str(cleaned_jsonl),
        "train_jsonl": str(train_jsonl),
        "train_txt": None if config.audit_only else str(train_txt),
        "validation_txt": None if config.audit_only else str(validation_txt),
        "review_jsonl": str(review_jsonl),
        "quarantine_jsonl": str(quarantine_jsonl),
        "dropped_jsonl": str(dropped_jsonl),
        "summary_json": str(summary_json),
        "report_summary_json": str(report_summary_json),
        "markdown_report": str(markdown_report) if config.write_report else None,
        "family_csv": str(family_csv) if config.write_report else None,
        "category_csv": str(category_csv) if config.write_report else None,
        "duplicate_csv": str(duplicate_csv) if config.write_report else None,
        "before_after_samples": str(samples_jsonl) if config.write_report else None,
    }
    summary = build_summary(records, config, paths)

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if config.write_report:
        write_markdown_report(markdown_report, summary)
        category_fields = ["rap_category", "record_count", "train_records", "avg_quality_score", "gold", "silver", "bronze", "review", "quarantine", "drop"]
        family_fields = ["rap_family", "record_count", "train_records", "avg_quality_score", "gold", "silver", "bronze", "review", "quarantine", "drop"]
        write_csv(category_csv, summarize_by(records, "rap_category"), category_fields)
        write_csv(family_csv, summarize_by(records, "rap_family"), family_fields)
        write_csv(
            duplicate_csv,
            duplicate_rows(records),
            ["record_id", "duplicate_of", "duplicate_type", "duplicate_score", "title", "artist_clean", "quality_tier", "decision"],
        )
        write_jsonl(samples_jsonl, report_samples(records, config.sample_limit))

    return summary


def clean_corpus(config: CorpusCleaningConfig) -> dict[str, Any]:
    print(f"[cleaning] reading source: {config.source_path}", flush=True)
    source_rows = read_source_records(config.source_path)
    print(f"[cleaning] source records: {len(source_rows)}", flush=True)
    records = audit_records(source_rows, config)
    print("[cleaning] writing outputs", flush=True)
    return write_outputs(records, config)


def main() -> None:
    args = parse_args()
    summary = clean_corpus(config_from_args(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
