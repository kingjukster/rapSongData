"""Clean categorized English rap lyrics into model-ready training corpora.

The pipeline is intentionally explicit: every dropped row gets one or more
rejection reasons, and every derived artifact is written to disk for audit.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl


REQUIRED_COLUMNS = [
    "title",
    "artist",
    "year",
    "views",
    "lyrics",
    "artist_clean",
    "rap_category",
    "rap_family",
    "lyrics_clean",
    "lyrics_chars",
    "line_count",
]

IMPORTANT_COLUMNS = [
    "title",
    "artist",
    "year",
    "views",
    "lyrics",
    "artist_clean",
    "rap_category",
    "rap_family",
    "lyrics_clean",
    "lyrics_chars",
    "line_count",
    "language",
    "language_cld3",
    "language_ft",
]

NUMERIC_COLUMNS = {
    "year": pl.Int32,
    "views": pl.Int64,
    "lyrics_chars": pl.Int64,
    "line_count": pl.Int64,
}

OUTPUT_FULL_COLUMNS = [
    "title",
    "artist",
    "year",
    "views",
    "artist_clean",
    "rap_category",
    "rap_family",
    "lyrics_model_text",
    "lyrics_model_chars",
    "lyrics_chars",
    "line_count",
    "word_count",
    "unique_word_count",
    "line_count_recomputed",
    "avg_words_per_line",
    "avg_chars_per_line",
    "repeated_line_ratio",
    "chorus_repetition_score",
    "profanity_count",
    "profanity_ratio",
    "explicit_token_count",
    "log_views",
    "near_duplicate_group",
    "near_duplicate_score",
    "quality_flags",
    "id",
    "tag",
    "features",
    "language",
    "language_cld3",
    "language_ft",
]

MODEL_COLUMNS = [
    "lyrics_model_text",
    "title",
    "artist",
    "artist_clean",
    "year",
    "views",
    "log_views",
    "rap_category",
    "rap_family",
    "word_count",
    "line_count_recomputed",
    "repeated_line_ratio",
    "chorus_repetition_score",
    "profanity_ratio",
]

SECTION_LABEL_RE = re.compile(
    r"^\s*\[?\s*(intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus|"
    r"post[- ]?chorus|refrain|interlude|skit|sample|part|break|beat switch|"
    r"instrumental|spoken|producer tag)(?:\s*[:\-\d].*)?\]?\s*$",
    re.IGNORECASE,
)
ARTIFACT_LINE_RE = re.compile(
    r"^\s*(you might also like|embed|read more|contributors?|translations?|"
    r"romanizations?|genius annotation|see .* live|get tickets|"
    r"how to format lyrics|lyrics powered by|more on genius)\b",
    re.IGNORECASE,
)
TRAILING_EMBED_RE = re.compile(r"\s*\d*\s*embed\s*$", re.IGNORECASE)
BAD_TITLE_RE = re.compile(
    r"(translation|romanization|tracklist|album art|liner notes|credits|"
    r"discography|annotated|meaning|interview|bio|setlist|concert review|"
    r"album review|release date|cover art)",
    re.IGNORECASE,
)
BAD_ARTIST_RE = re.compile(
    r"^(unknown|various artists|genius|rap genius|spotify|soundcloud|"
    r"youtube|tiktok|apple music|album art|translations?|romanizations?|"
    r"community contributors?|users?)$",
    re.IGNORECASE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
PROFANITY_RE = re.compile(
    r"\b("
    r"fuck(?:ing|ed|er|ers)?|shit(?:ty)?|bitch(?:es)?|nigga(?:s)?|nigger(?:s)?|"
    r"ass(?:hole)?|damn|piss|dick|pussy|cunt|motherfucker(?:s)?|"
    r"hoe(?:s)?|whore(?:s)?|slut(?:s)?"
    r")\b",
    re.IGNORECASE,
)

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2026": "...",
    }
)


def normalize_text(value: object, preserve_lines: bool = False) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub("", text)
    if not preserve_lines:
        return re.sub(r"\s+", " ", text).strip()

    lines = []
    blank_seen = False
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            if not blank_seen and lines:
                lines.append("")
            blank_seen = True
            continue
        blank_seen = False
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def clean_lyrics(value: object) -> str:
    text = normalize_text(value, preserve_lines=True)
    kept_lines: list[str] = []
    for line in text.split("\n"):
        stripped = TRAILING_EMBED_RE.sub("", line).strip()
        if not stripped:
            if kept_lines and kept_lines[-1]:
                kept_lines.append("")
            continue
        if SECTION_LABEL_RE.match(stripped):
            continue
        if ARTIFACT_LINE_RE.match(stripped):
            continue
        if stripped in {"[?]", "?", "..."}:
            continue
        kept_lines.append(stripped)

    while kept_lines and not kept_lines[-1]:
        kept_lines.pop()
    return "\n".join(kept_lines).strip()


def lyrics_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_artist_key(title: str, artist: str) -> str:
    return f"{simple_key(artist)}::{simple_key(title)}"


def simple_key(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def compute_text_features(text: str) -> dict[str, float | int]:
    lines = [line for line in text.split("\n") if line.strip()]
    words = word_tokens(text)
    line_word_counts = [len(word_tokens(line)) for line in lines]
    line_char_counts = [len(line) for line in lines]
    line_counter = Counter(simple_key(line) for line in lines if simple_key(line))
    repeated_lines = sum(count for count in line_counter.values() if count > 1)
    max_repeated = max(line_counter.values(), default=0)
    profanity_count = len(PROFANITY_RE.findall(text))
    word_count = len(words)
    return {
        "word_count": word_count,
        "unique_word_count": len(set(words)),
        "line_count_recomputed": len(lines),
        "avg_words_per_line": safe_div(sum(line_word_counts), len(lines)),
        "avg_chars_per_line": safe_div(sum(line_char_counts), len(lines)),
        "repeated_line_ratio": safe_div(repeated_lines, len(lines)),
        "chorus_repetition_score": safe_div(max_repeated, len(lines)),
        "profanity_count": profanity_count,
        "profanity_ratio": safe_div(profanity_count, word_count),
        "explicit_token_count": profanity_count,
    }


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def suspicious_text(text: str) -> list[str]:
    flags: list[str] = []
    if "\ufffd" in text:
        flags.append("replacement_character")
    if CONTROL_CHAR_RE.search(text):
        flags.append("control_character")
    chars = [char for char in text if not char.isspace()]
    if chars:
        ascii_letters = sum(char.isascii() and char.isalpha() for char in chars)
        letters = sum(char.isalpha() for char in chars)
        if letters and ascii_letters / letters < 0.75:
            flags.append("low_ascii_letter_ratio")
    if len(set(text.lower())) < 12 and len(text) > 300:
        flags.append("low_character_diversity")
    return flags


def stable_hash64(value: str) -> int:
    data = value.encode("utf-8", errors="ignore")
    high = zlib.crc32(data)
    low = zlib.crc32(data, 0xA5A5A5A5)
    return ((high << 32) | low) & 0xFFFFFFFFFFFFFFFF


def simhash(tokens: list[str]) -> int:
    vector = [0] * 64
    if len(tokens) >= 5:
        shingle_count = len(tokens) - 4
        step = max(1, shingle_count // 500)
        units = (" ".join(tokens[i : i + 5]) for i in range(0, shingle_count, step))
    else:
        units = iter(tokens)
    for token in units:
        h = stable_hash64(token)
        for bit in range(64):
            vector[bit] += 1 if h & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_duplicate_groups(keys: list[str], max_distance: int = 4) -> tuple[list[int], list[float]]:
    hashes = [simhash(word_tokens(key)) if key else 0 for key in keys]
    buckets: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    parent = list(range(len(keys)))
    best_score = [0.0] * len(keys)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, score: float) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root
        best_score[left] = max(best_score[left], score)
        best_score[right] = max(best_score[right], score)

    for index, key in enumerate(keys):
        if len(key) < 500:
            continue
        h = hashes[index]
        candidates: set[int] = set()
        for band in range(4):
            band_key = (band, (h >> (band * 16)) & 0xFFFF)
            candidates.update(buckets[band_key][-50:])
        for candidate in candidates:
            distance = hamming_distance(h, hashes[candidate])
            if distance <= max_distance:
                score = 1.0 - (distance / 64.0)
                union(index, candidate, score)
        for band in range(4):
            band_key = (band, (h >> (band * 16)) & 0xFFFF)
            buckets[band_key].append(index)

    roots = [find(index) for index in range(len(keys))]
    counts = Counter(roots)
    root_to_group: dict[int, int] = {}
    next_group = 1
    groups = [0] * len(keys)
    for index, root in enumerate(roots):
        if counts[root] <= 1:
            continue
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        groups[index] = root_to_group[root]
    return groups, best_score


def build_rejection_reasons(row: dict, args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    title = row["title_norm"]
    artist = row["artist_norm"]
    artist_clean = row["artist_clean_norm"]
    lyrics = row["lyrics_model_text"]
    language = str(row.get("language") or "").lower().strip()
    language_cld3 = str(row.get("language_cld3") or "").lower().strip()
    language_ft = str(row.get("language_ft") or "").lower().strip()
    year = row["year"]
    line_count = row["line_count_recomputed"]
    char_count = len(lyrics)
    word_count = row["word_count"]

    if not title:
        reasons.append("missing_title")
    if not artist or not artist_clean:
        reasons.append("missing_artist")
    if BAD_TITLE_RE.search(title):
        reasons.append("metadata_or_translation_title")
    if BAD_ARTIST_RE.search(artist_clean) or BAD_ARTIST_RE.search(artist):
        reasons.append("non_artist_or_unknown_artist")
    if not (args.min_year <= year <= args.max_year):
        reasons.append("suspicious_year")
    labels = [label for label in [language, language_cld3, language_ft] if label and label != "null"]
    invalid_labels = [label for label in labels if not (label == "en" or label.startswith("en-"))]
    if invalid_labels:
        reasons.append("non_english_language_label")
    if not lyrics:
        reasons.append("empty_lyrics")
    if char_count < args.min_chars or word_count < args.min_words or line_count < args.min_lines:
        reasons.append("too_short")
    if char_count > args.max_chars or line_count > args.max_lines:
        reasons.append("too_long")
    reasons.extend(suspicious_text(lyrics))
    if row["rap_category_norm"] in {"", "unknown", "other", "none", "nan"}:
        reasons.append("invalid_rap_category")
    if row["rap_family_norm"] in {"", "unknown", "none", "nan"}:
        reasons.append("invalid_rap_family")
    return reasons


def read_source(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(
            path,
            infer_schema_length=10_000,
            schema_overrides=NUMERIC_COLUMNS,
            null_values=["", "NA", "N/A", "null", "None"],
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column, dtype in NUMERIC_COLUMNS.items():
        df = df.with_columns(pl.col(column).cast(dtype, strict=False))
    return df


def missing_report(df: pl.DataFrame) -> dict[str, int]:
    columns = [column for column in IMPORTANT_COLUMNS if column in df.columns]
    result = df.select([pl.col(column).null_count().alias(column) for column in columns]).to_dicts()[0]
    for column in columns:
        result[f"{column}_blank"] = int(
            df.select(
                pl.col(column)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .eq("")
                .sum()
                .alias(column)
            ).item()
        )
    return {key: int(value or 0) for key, value in result.items()}


def quantile_summary(df: pl.DataFrame, column: str) -> dict[str, float]:
    if df.is_empty() or column not in df.columns:
        return {}
    exprs = [
        pl.col(column).min().alias("min"),
        pl.col(column).quantile(0.01).alias("p01"),
        pl.col(column).quantile(0.05).alias("p05"),
        pl.col(column).median().alias("median"),
        pl.col(column).quantile(0.95).alias("p95"),
        pl.col(column).quantile(0.99).alias("p99"),
        pl.col(column).max().alias("max"),
    ]
    values = df.select(exprs).to_dicts()[0]
    return {key: float(value) if value is not None else math.nan for key, value in values.items()}


def dataframe_head_records(df: pl.DataFrame, columns: list[str], n: int = 25) -> list[dict]:
    available = [column for column in columns if column in df.columns]
    return df.select(available).head(n).to_dicts()


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# Rap Lyrics Cleaning Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Source: `{report['source_path']}`",
        "",
        "## Row Counts",
        "",
        f"- Input rows: {report['row_counts']['input']}",
        f"- Rejected rows: {report['row_counts']['rejected']}",
        f"- Retained rows: {report['row_counts']['retained']}",
        "",
        "## Removed Rows by Reason",
        "",
    ]
    for reason, count in report["removed_rows_by_reason"].items():
        lines.append(f"- `{reason}`: {count}")

    lines.extend(["", "## Missing Values", ""])
    for column, count in report["missing_values_before"].items():
        if count:
            lines.append(f"- `{column}`: {count}")
    if lines[-1] == "":
        lines.append("- No missing or blank values found in important columns.")

    lines.extend(["", "## Category Summary", ""])
    for row in report["category_summary"][:30]:
        lines.append(f"- {row['rap_category']}: {row['songs']} songs, {row['artists']} artists")

    lines.extend(["", "## Family Summary", ""])
    for row in report["family_summary"]:
        lines.append(f"- {row['rap_family']}: {row['songs']} songs, {row['artists']} artists")

    lines.extend(["", "## Sparse Categories", ""])
    sparse = report["sparse_categories"]
    if sparse:
        for row in sparse:
            lines.append(f"- {row['rap_category']}: {row['songs']} songs, {row['artists']} artists")
    else:
        lines.append("- None under configured thresholds.")

    lines.extend(["", "## Output Files", ""])
    for label, output_path in report["outputs"].items():
        lines.append(f"- {label}: `{output_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rap_english_clean_categorized_with_families.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_ready"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--min-year", type=int, default=1970)
    parser.add_argument("--max-year", type=int, default=datetime.now().year + 1)
    parser.add_argument("--min-chars", type=int, default=450)
    parser.add_argument("--min-words", type=int, default=90)
    parser.add_argument("--min-lines", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=25_000)
    parser.add_argument("--max-lines", type=int, default=260)
    parser.add_argument("--min-category-songs", type=int, default=250)
    parser.add_argument("--min-category-artists", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = read_source(args.input)
    input_rows = df.height
    schema = {column: str(dtype) for column, dtype in df.schema.items()}
    missing_before = missing_report(df)

    records = df.to_dicts()
    processed: list[dict] = []
    rejected: list[dict] = []

    for original_index, row in enumerate(records):
        normalized = dict(row)
        normalized["source_row_index"] = original_index
        normalized["title_norm"] = normalize_text(row.get("title"))
        normalized["artist_norm"] = normalize_text(row.get("artist"))
        normalized["artist_clean_norm"] = normalize_text(row.get("artist_clean"))
        normalized["rap_category_norm"] = normalize_text(row.get("rap_category"))
        normalized["rap_family_norm"] = normalize_text(row.get("rap_family"))
        normalized["lyrics_model_text"] = clean_lyrics(row.get("lyrics_clean") or row.get("lyrics"))
        normalized["lyrics_normalized_key"] = lyrics_key(normalized["lyrics_model_text"])
        normalized["artist_title_key"] = title_artist_key(normalized["title_norm"], normalized["artist_clean_norm"] or normalized["artist_norm"])
        normalized["quality_flags"] = ",".join(suspicious_text(normalized["lyrics_model_text"]))
        features = compute_text_features(normalized["lyrics_model_text"])
        normalized.update(features)
        normalized["log_views"] = math.log1p(int(normalized.get("views") or 0))
        normalized["metadata_completeness_score"] = sum(
            1
            for column in ["title", "artist", "artist_clean", "year", "views", "rap_category", "rap_family", "language"]
            if normalized.get(column) not in [None, ""]
        )
        normalized["year"] = int(normalized["year"]) if normalized.get("year") is not None else -1
        normalized["views"] = int(normalized["views"]) if normalized.get("views") is not None else 0
        reasons = build_rejection_reasons(normalized, args)
        if reasons:
            normalized["rejection_reason"] = ";".join(sorted(set(reasons)))
            rejected.append(normalized)
        else:
            processed.append(normalized)

    candidate_df = pl.DataFrame(processed, infer_schema_length=10_000) if processed else pl.DataFrame()
    rejected_rows = rejected

    if not candidate_df.is_empty():
        candidate_df = candidate_df.sort(
            ["metadata_completeness_score", "views", "word_count"],
            descending=[True, True, True],
        )

        duplicate_lyrics = candidate_df.filter(pl.col("lyrics_normalized_key").is_duplicated())
        duplicate_artist_titles = candidate_df.filter(pl.col("artist_title_key").is_duplicated())

        keep_df = candidate_df.unique(subset=["lyrics_normalized_key"], keep="first", maintain_order=True)
        lyric_removed_keys = set(candidate_df["source_row_index"]) - set(keep_df["source_row_index"])
        for row in candidate_df.filter(pl.col("source_row_index").is_in(list(lyric_removed_keys))).to_dicts():
            row["rejection_reason"] = "duplicate_lyrics_exact"
            rejected_rows.append(row)

        before_artist_title = keep_df
        keep_df = keep_df.unique(subset=["artist_title_key"], keep="first", maintain_order=True)
        artist_title_removed = set(before_artist_title["source_row_index"]) - set(keep_df["source_row_index"])
        for row in before_artist_title.filter(pl.col("source_row_index").is_in(list(artist_title_removed))).to_dicts():
            row["rejection_reason"] = "duplicate_artist_title"
            rejected_rows.append(row)

        near_groups, near_scores = near_duplicate_groups(keep_df["lyrics_normalized_key"].to_list())
        keep_df = keep_df.with_columns(
            pl.Series("near_duplicate_group", near_groups, dtype=pl.Int64),
            pl.Series("near_duplicate_score", near_scores, dtype=pl.Float64),
        )
    else:
        duplicate_lyrics = pl.DataFrame()
        duplicate_artist_titles = pl.DataFrame()
        keep_df = candidate_df

    rejected_df = pl.DataFrame(rejected_rows, infer_schema_length=10_000) if rejected_rows else pl.DataFrame()
    reason_counts = Counter()
    for row in rejected_rows:
        for reason in str(row.get("rejection_reason", "")).split(";"):
            if reason:
                reason_counts[reason] += 1

    keep_df = keep_df.with_columns(
        pl.col("title_norm").alias("title"),
        pl.col("artist_norm").alias("artist"),
        pl.col("artist_clean_norm").alias("artist_clean"),
        pl.col("rap_category_norm").alias("rap_category"),
        pl.col("rap_family_norm").alias("rap_family"),
        pl.col("lyrics_model_text").str.len_chars().alias("lyrics_model_chars"),
    )

    full_columns = [column for column in OUTPUT_FULL_COLUMNS if column in keep_df.columns]
    model_columns = [column for column in MODEL_COLUMNS if column in keep_df.columns]
    full_df = keep_df.select(full_columns)
    model_df = keep_df.select(model_columns)

    full_csv = args.output_dir / "rap_lyrics_model_ready_full.csv"
    full_parquet = args.output_dir / "rap_lyrics_model_ready_full.parquet"
    model_csv = args.output_dir / "rap_lyrics_training_dataset.csv"
    model_parquet = args.output_dir / "rap_lyrics_training_dataset.parquet"
    rejected_csv = args.output_dir / "rap_lyrics_rejected_rows.csv"
    near_dupes_csv = args.output_dir / "rap_lyrics_near_duplicate_flags.csv"
    report_json = args.report_dir / "rap_lyrics_cleaning_report.json"
    report_md = args.report_dir / "rap_lyrics_cleaning_report.md"

    full_df.write_csv(full_csv)
    full_df.write_parquet(full_parquet, compression="zstd")
    model_df.write_csv(model_csv)
    model_df.write_parquet(model_parquet, compression="zstd")
    if rejected_df.is_empty():
        rejected_df = pl.DataFrame({"rejection_reason": []})
    rejected_df.write_csv(rejected_csv)
    full_df.filter(pl.col("near_duplicate_group") > 0).write_csv(near_dupes_csv)

    category_summary = (
        full_df.group_by("rap_category")
        .agg(pl.len().alias("songs"), pl.col("artist_clean").n_unique().alias("artists"))
        .sort("songs", descending=True)
    )
    family_summary = (
        full_df.group_by("rap_family")
        .agg(pl.len().alias("songs"), pl.col("artist_clean").n_unique().alias("artists"))
        .sort("songs", descending=True)
    )
    sparse_categories = category_summary.filter(
        (pl.col("songs") < args.min_category_songs) | (pl.col("artists") < args.min_category_artists)
    )

    suspicious_review = full_df.sort(
        ["near_duplicate_group", "repeated_line_ratio", "lyrics_model_chars"],
        descending=[True, True, True],
    ).head(50)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_path": str(args.input),
        "schema": schema,
        "required_columns": REQUIRED_COLUMNS,
        "missing_required_columns": [column for column in REQUIRED_COLUMNS if column not in df.columns],
        "numeric_casts": {column: str(dtype) for column, dtype in NUMERIC_COLUMNS.items()},
        "row_counts": {
            "input": input_rows,
            "candidate_after_rule_rejections": len(processed),
            "rejected": len(rejected_rows),
            "retained": full_df.height,
            "exact_duplicate_lyrics_seen": duplicate_lyrics.height,
            "duplicate_artist_title_seen": duplicate_artist_titles.height,
            "near_duplicate_flagged": full_df.filter(pl.col("near_duplicate_group") > 0).height,
        },
        "removed_rows_by_reason": dict(sorted(reason_counts.items())),
        "missing_values_before": missing_before,
        "missing_values_after": missing_report(full_df),
        "year_distribution": {
            "summary": quantile_summary(full_df, "year"),
            "counts": full_df.group_by("year").agg(pl.len().alias("songs")).sort("year").to_dicts(),
        },
        "artist_distribution_top": (
            full_df.group_by("artist_clean")
            .agg(pl.len().alias("songs"), pl.col("views").sum().alias("total_views"))
            .sort("songs", descending=True)
            .head(50)
            .to_dicts()
        ),
        "category_summary": category_summary.to_dicts(),
        "family_summary": family_summary.to_dicts(),
        "sparse_categories": sparse_categories.to_dicts(),
        "lyric_length_distribution": {
            "chars": quantile_summary(full_df, "lyrics_model_chars"),
            "words": quantile_summary(full_df, "word_count"),
            "lines": quantile_summary(full_df, "line_count_recomputed"),
        },
        "top_suspicious_rows": dataframe_head_records(
            suspicious_review,
            [
                "title",
                "artist",
                "year",
                "views",
                "rap_category",
                "rap_family",
                "word_count",
                "line_count_recomputed",
                "repeated_line_ratio",
                "near_duplicate_group",
                "near_duplicate_score",
                "quality_flags",
            ],
            n=50,
        ),
        "outputs": {
            "full_csv": str(full_csv),
            "full_parquet": str(full_parquet),
            "training_csv": str(model_csv),
            "training_parquet": str(model_parquet),
            "rejected_rows_csv": str(rejected_csv),
            "near_duplicate_flags_csv": str(near_dupes_csv),
            "report_json": str(report_json),
            "report_markdown": str(report_md),
        },
        "thresholds": {
            "min_year": args.min_year,
            "max_year": args.max_year,
            "min_chars": args.min_chars,
            "min_words": args.min_words,
            "min_lines": args.min_lines,
            "max_chars": args.max_chars,
            "max_lines": args.max_lines,
            "min_category_songs": args.min_category_songs,
            "min_category_artists": args.min_category_artists,
        },
    }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, report_md)
    print(json.dumps(report["row_counts"], indent=2))
    print(f"Wrote full dataset: {full_parquet}")
    print(f"Wrote training dataset: {model_parquet}")
    print(f"Wrote report: {report_json}")


if __name__ == "__main__":
    main()
