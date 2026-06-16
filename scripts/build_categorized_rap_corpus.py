"""Build the categorized English rap corpus from the raw song lyrics CSV.

The original artist category map lives in ``rap_songs_filter.ipynb``. This
script reuses that map and makes the notebook workflow repeatable from the
command line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus_cleaning import CorpusCleaningConfig, clean_corpus


DEFAULT_INPUT = Path("data/song_lyrics.csv")
DEFAULT_NOTEBOOK = Path("rap_songs_filter.ipynb")
DEFAULT_OUTPUT_DIR = Path("data")

SCHEMA_OVERRIDES = {
    "title": pl.Utf8,
    "tag": pl.Utf8,
    "artist": pl.Utf8,
    "year": pl.Int32,
    "views": pl.Int64,
    "features": pl.Utf8,
    "lyrics": pl.Utf8,
    "id": pl.Int64,
    "language_cld3": pl.Utf8,
    "language_ft": pl.Utf8,
    "language": pl.Utf8,
}

KEEP_COLUMNS = [
    "title",
    "tag",
    "artist",
    "year",
    "views",
    "features",
    "id",
    "language_cld3",
    "language_ft",
    "language",
    "lyrics",
]

BAD_CATEGORIES = {
    "Other / Non-rap / Metadata",
    "Other / Non-rap / Unknown",
    "Other / Non-rap / Compilation",
    "Other / Non-rap / Uncategorized",
    "Uncategorized",
}

METADATA_ARTIST_RE = (
    r"(?i)"
    r"genius|translations|traducciones|tradues|traductions|"
    r"romanizations|spotify|unknown artist|various artists"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rap-cache", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    parser.add_argument("--force-rap-cache", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Run corpus_cleaning.py after categorized outputs are built.")
    parser.add_argument("--audit-only", action="store_true", help="Run cleaning audit but skip cleaned train text output.")
    parser.add_argument("--write-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-bronze", action="store_true", help="Include bronze records in cleaned train text.")
    parser.add_argument("--min-quality", type=float, default=0.70)
    parser.add_argument("--dedupe", choices=["exact", "near", "both"], default="both")
    return parser.parse_args()


def normalize_artist_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("&nbsp;", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text or None


def clean_artist_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("&nbsp;", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("'").strip()
    return text or None


def clean_lyrics_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"\[(intro|verse|chorus|hook|bridge|outro|pre-chorus|refrain|skit|interlude).*?\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\d+ Contributors?.*?Lyrics", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"You might also like", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    text = "\n".join(lines).strip()
    return text or None


def load_notebook_categories(path: Path) -> tuple[dict[str, str], Any]:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if (
            "artist_category_map =" in source
            or "artist_category_map.update" in source
            or "def broad_rap_category" in source
        ):
            exec(source, namespace)
    artist_category_map = namespace.get("artist_category_map")
    broad_rap_category = namespace.get("broad_rap_category")
    if not artist_category_map or broad_rap_category is None:
        raise ValueError("Could not extract artist_category_map and broad_rap_category from notebook.")
    return artist_category_map, broad_rap_category


def build_rap_cache(input_path: Path, cache_path: Path, force: bool) -> None:
    if cache_path.exists() and not force:
        print(f"Using existing rap cache: {cache_path}")
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    songs_lf = pl.scan_csv(
        input_path,
        schema_overrides=SCHEMA_OVERRIDES,
        infer_schema_length=1_000,
        null_values=["", "NA", "N/A", "null", "None"],
    )
    available_columns = songs_lf.collect_schema().names()
    selected_columns = [column for column in KEEP_COLUMNS if column in available_columns]
    rap_lf = (
        songs_lf
        .filter(pl.col("tag").str.to_lowercase() == "rap")
        .select(selected_columns)
    )
    print(f"Writing rap cache: {cache_path}")
    rap_lf.sink_parquet(cache_path, compression="zstd")


def build_categorized_outputs(args: argparse.Namespace) -> dict[str, Any]:
    artist_category_map, broad_rap_category = load_notebook_categories(args.notebook)
    category_df = pl.DataFrame(
        {
            "artist_norm": [normalize_artist_name(artist) for artist in artist_category_map],
            "rap_category": list(artist_category_map.values()),
        }
    ).unique("artist_norm")

    rap_df = pl.scan_parquet(args.rap_cache)
    categorized = (
        rap_df
        .with_columns(
            pl.col("artist")
            .map_elements(normalize_artist_name, return_dtype=pl.String)
            .alias("artist_norm"),
            pl.col("artist")
            .map_elements(clean_artist_name, return_dtype=pl.String)
            .alias("artist_clean"),
        )
        .filter(~pl.col("artist_clean").str.contains(METADATA_ARTIST_RE))
        .join(category_df.lazy(), on="artist_norm", how="left")
        .filter(~pl.col("rap_category").is_in(BAD_CATEGORIES))
        .filter(pl.col("rap_category").is_not_null())
        .with_columns(
            pl.col("lyrics")
            .map_elements(clean_lyrics_text, return_dtype=pl.String)
            .alias("lyrics_clean")
        )
        .filter(pl.col("lyrics_clean").is_not_null())
        .with_columns(
            pl.col("lyrics_clean").str.len_chars().alias("lyrics_chars"),
            pl.col("lyrics_clean").str.split("\n").list.len().alias("line_count"),
        )
        .filter(pl.col("lyrics_chars") >= 500)
        .filter(pl.col("lyrics_chars") <= 12000)
        .filter(pl.col("line_count") >= 8)
        .unique(subset=["artist_clean", "title"], keep="first")
        .unique(subset=["lyrics_clean"], keep="first")
        .filter(
            pl.col("language").cast(pl.String).str.to_lowercase().is_in(["en", "eng", "english"])
            | pl.col("language_cld3").cast(pl.String).str.to_lowercase().is_in(["en", "eng", "english"])
            | pl.col("language_ft").cast(pl.String).str.to_lowercase().is_in(["en", "eng", "english"])
        )
        .with_columns(
            pl.col("rap_category")
            .map_elements(broad_rap_category, return_dtype=pl.String)
            .alias("rap_family")
        )
        .drop("artist_norm")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    categorized_parquet = args.output_dir / "rap_english_clean_categorized_with_families.parquet"
    categorized_csv = args.output_dir / "rap_english_clean_categorized_with_families.csv"
    intermediate_parquet = args.output_dir / "rap_english_clean_categorized.parquet"
    intermediate_csv = args.output_dir / "rap_english_clean_categorized.csv"

    print(f"Collecting categorized corpus and writing: {categorized_parquet}")
    df = categorized.collect()
    df.write_parquet(categorized_parquet, compression="zstd")
    df.write_csv(categorized_csv)
    df.drop("rap_family").write_parquet(intermediate_parquet, compression="zstd")
    df.drop("rap_family").write_csv(intermediate_csv)

    family_summary = (
        df.group_by("rap_family")
        .agg(
            pl.len().alias("song_count"),
            pl.col("artist_clean").n_unique().alias("artist_count"),
            pl.col("rap_category").n_unique().alias("source_categories"),
            pl.col("lyrics_chars").mean().alias("avg_chars"),
            pl.col("line_count").mean().alias("avg_lines"),
        )
        .sort("song_count", descending=True)
    )
    return {
        "categorized_parquet": str(categorized_parquet),
        "categorized_csv": str(categorized_csv),
        "rows": df.height,
        "artists": df["artist_clean"].n_unique(),
        "family_summary": family_summary.to_dicts(),
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Raw lyrics CSV not found: {args.input}")
    build_rap_cache(args.input, args.rap_cache, args.force_rap_cache)
    result = build_categorized_outputs(args)
    if args.clean or args.audit_only:
        cleaning_summary = clean_corpus(
            CorpusCleaningConfig(
                source_path=Path(result["categorized_parquet"]),
                output_dir=Path("data/cleaned"),
                report_dir=Path("reports"),
                audit_only=args.audit_only,
                write_report=args.write_report,
                include_bronze=args.include_bronze,
                min_quality=args.min_quality,
                dedupe=args.dedupe,
            )
        )
        result["cleaning"] = cleaning_summary
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
