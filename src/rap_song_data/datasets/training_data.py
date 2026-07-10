"""Build model-training JSONL files from the cleaned rap lyrics corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_CONFIG_PATH = Path("configs/datasets/dataset_config.json")
DEFAULT_ARCHITECTURE_SEEDS_PATH = Path("configs/datasets/lyrical_architecture_seeds.json")
REQUIRED_COLUMNS = [
    "lyrics_model_text",
    "title",
    "artist_clean",
    "year",
    "views",
    "log_views",
    "rap_category",
    "rap_family",
]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
BRACKET_LABEL_RE = re.compile(r"^\s*\[[^\]]{1,80}\]\s*$")
ADLIB_RE = re.compile(r"\([^)]{1,40}\)")
STAGE_DIRECTION_RE = re.compile(r"\*[^*\n]{1,80}\*")
QUOTED_DIALOGUE_RE = re.compile(r'"[^"\n]{1,160}"')
HARD_BAD_CONTENT_RE = re.compile(
    r"\b(?:dick|suck|fuck(?:in|ing)?|bitch(?:es)?|hoe(?:s)?|shoot|kill|die|fight|murder|gun|strap|knife|club|shots?|skin|panties|wet|drugs?|molly|weed|coke)\b",
    re.IGNORECASE,
)
SOFT_DRIFT_RE = re.compile(
    r"\b(?:baby|girl|girls|love|date|romance|drink|party|poppin|flex|diamond|chain|million|famous|hatin|hater|jewelry|foreign|skrilla|pirate|tailgate|kiss|heart|lullaby|nightmare|fairytale)\b",
    re.IGNORECASE,
)
SCENE_KEYWORD_RE = re.compile(
    r"\b(?:night|shift|late|clock|overtime|work|working|job|paycheck|payroll|warehouse|streetlights?|tired|eyes|city|lights|midnight|morning|floor|bus|train|uniform|breakroom|coffee|ambition|pressure|grind|rent|dreams?|future)\b",
    re.IGNORECASE,
)
WORK_ANCHOR_RE = re.compile(
    r"\b(?:shift|clock|overtime|work|working|job|paycheck|payroll|warehouse|uniform|breakroom|coffee|rent|boss|hours?|morning|midnight)\b",
    re.IGNORECASE,
)
GENERAL_THEME = "general_vibe"
THEME_FILLER_WORDS = {
    "baby",
    "check",
    "feel",
    "feelin",
    "feeling",
    "girl",
    "girls",
    "life",
    "love",
    "man",
    "pain",
    "shit",
    "thing",
    "things",
    "time",
    "wait",
    "yeah",
}
STOPWORDS = {
    "about",
    "again",
    "ain't",
    "also",
    "because",
    "been",
    "being",
    "could",
    "down",
    "from",
    "have",
    "into",
    "just",
    "like",
    "make",
    "more",
    "some",
    "that",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "through",
    "want",
    "were",
    "what",
    "when",
    "where",
    "with",
    "would",
    "yeah",
    "your",
    *THEME_FILLER_WORDS,
}
SKIP_ARTIFACT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\[?\s*lyrics\s+(?:taken\s+from|from)\b.*\]?$|"
    r"important\s*:\s*rap\s+genius\b.*|"
    r"url\s*:\s*https?://.*|"
    r"contributors?\b.*|"
    r"embed\s*$"
    r")",
    re.IGNORECASE,
)
BREAK_ARTIFACT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"lyrics\s+(?:taken\s+from|from)\b|"
    r"continue\s+into\s+https?://|"
    r"source\s*:|"
    r"https?://|"
    r".*\bgenius\.com\b|"
    r"you\s+might\s+also\s+like\b"
    r")",
    re.IGNORECASE,
)
MIN_SANITIZED_LINES = 8
MIN_SANITIZED_WORDS = 90
DEFAULT_VERSE_RULES = (
    "Write only original lyrics. Keep line breaks. Do not explain. "
    "Avoid bracket labels, dialogue, URLs, source notes, and copied song text."
)
DEFAULT_CLEAN_SCENE_RULES = (
    "Write only clean original lyrics. Keep line breaks. Stay focused on the scene, "
    "keywords, fatigue, pressure, city lights, and ambition. Avoid romance, clubs, "
    "violence, explicit content, dialogue, ad-libs, URLs, source notes, and copied song text."
)
VERSE_START_TOKEN = "<|verse_start|>"
VERSE_END_TOKEN = "<|verse_end|>"
BAR_START_TOKEN = "<|bar_start|>"
STRUCTURAL_SPECIAL_TOKENS = [VERSE_START_TOKEN, VERSE_END_TOKEN, BAR_START_TOKEN]
ASCII_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)
THEME_BUCKETS = {
    "late nights, pressure, ambition, city lights": {
        "night",
        "late",
        "shift",
        "job",
        "paid",
        "hour",
        "hours",
        "clock",
        "overtime",
        "morning",
        "midnight",
        "city",
        "light",
        "lights",
        "work",
        "working",
        "grind",
        "pressure",
        "dream",
        "ambition",
        "sleep",
        "awake",
        "tired",
        "lonely",
        "alone",
    },
    "heartbreak, loneliness, memory, emotional conflict": {
        "heart",
        "alone",
        "lonely",
        "miss",
        "cry",
        "tears",
        "memory",
        "memories",
        "broken",
    },
    "street survival, danger, loyalty, paranoia": {
        "street",
        "block",
        "opp",
        "ops",
        "gun",
        "strap",
        "shoot",
        "survive",
        "loyal",
        "brother",
        "hood",
        "trap",
    },
    "money, flexing, status, success": {
        "money",
        "cash",
        "rich",
        "ice",
        "chain",
        "wrist",
        "car",
        "foreign",
        "designer",
        "boss",
        "success",
    },
    "party, drugs, energy, nightlife": {
        "club",
        "party",
        "bottle",
        "drink",
        "weed",
        "smoke",
        "high",
        "molly",
        "lit",
        "dance",
    },
    "reflection, growth, struggle, self-doubt": {
        "think",
        "mind",
        "soul",
        "grow",
        "growth",
        "change",
        "struggle",
        "doubt",
        "faith",
        "pray",
        "truth",
    },
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.source is not None:
        config["source"] = str(args.source)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        config["random_seed"] = args.seed
    return config


def validate_config(config: dict[str, Any]) -> None:
    split = config.get("split", {})
    required_split_keys = {"train", "validation", "test"}
    missing_split_keys = sorted(required_split_keys - set(split))
    if missing_split_keys:
        raise ValueError(f"Missing split keys in config: {missing_split_keys}")

    split_total = sum(float(split[key]) for key in required_split_keys)
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {split_total}")

    tasks = set(config.get("tasks", []))
    supported_tasks = {
        "generate_song",
        "continue_verse",
        "generate_verse",
        "generate_clean_scene_verse",
        "generate_architectural_verse",
    }
    unsupported_tasks = sorted(tasks - supported_tasks)
    if unsupported_tasks:
        raise ValueError(f"Unsupported task(s): {unsupported_tasks}. Supported: {sorted(supported_tasks)}")


def read_source(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned training dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path, infer_schema_length=10_000)
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}. Use .parquet or .csv")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Source dataset is missing required column(s): {missing}")
    return df


def normalize_match_text(value: Any) -> str:
    text = clean_control_value(value).lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_architecture_seeds(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        seeds = json.load(file)
    seed_index: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in seeds:
        artist = normalize_match_text(seed.get("artist"))
        title = normalize_match_text(seed.get("title"))
        if artist and title:
            seed_index[(artist, title)] = {
                "architecture": clean_control_value(seed.get("architecture")),
                "features": [clean_control_value(feature) for feature in seed.get("features", []) if clean_control_value(feature)],
            }
    return seed_index


def architecture_for_row(row: dict[str, Any], seed_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    if not seed_index:
        return {}
    artist = normalize_match_text(row.get("artist_clean"))
    title = normalize_match_text(row.get("title"))
    return seed_index.get((artist, title), {})


def clean_control_value(value: Any) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value).translate(ASCII_REPLACEMENTS))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.replace("\r\n", "\n").replace("\r", "\n").split())


def clean_lyric_line(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).translate(ASCII_REPLACEMENTS))
    return normalized.encode("ascii", "ignore").decode("ascii")


def stable_id(*parts: Any) -> str:
    payload = "\u241f".join(str(part) for part in parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.0000"


def metadata_from_row(row: dict[str, Any], seed_index: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    metadata = {
        "title": clean_control_value(row.get("title")),
        "artist_clean": clean_control_value(row.get("artist_clean")),
        "rap_family": clean_control_value(row.get("rap_family")),
        "rap_category": clean_control_value(row.get("rap_category")),
        "year": int(row.get("year") or 0),
        "views": int(row.get("views") or 0),
        "log_views": round(float(row.get("log_views") or 0.0), 4),
    }
    architecture = architecture_for_row(row, seed_index or {})
    if architecture:
        metadata.update(architecture)
    return metadata


def lyric_lines(text: Any) -> list[str]:
    if text is None:
        return []
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return [clean_lyric_line(line).rstrip() for line in normalized.split("\n")]


def sanitize_lyrics_for_training(text: Any) -> str:
    kept: list[str] = []
    for raw_line in lyric_lines(text):
        line = raw_line.strip()
        if not line:
            if kept and kept[-1]:
                kept.append("")
            continue
        if SKIP_ARTIFACT_LINE_RE.match(line):
            continue
        if BREAK_ARTIFACT_LINE_RE.match(line):
            break
        kept.append(line)

    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept).strip()


def has_enough_lyrics(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)
    return len(lines) >= MIN_SANITIZED_LINES and len(words) >= MIN_SANITIZED_WORDS


def format_annotated_lyrics(text: str) -> str:
    """Wrap the verse and prefix each non-empty lyric line with a trainable special token."""
    bars = [line.strip() for line in lyric_lines(text) if line.strip()]
    annotated_bars = [f"{BAR_START_TOKEN}{bar}" for bar in bars]
    return "\n".join([VERSE_START_TOKEN, *annotated_bars, VERSE_END_TOKEN])


def infer_theme_info(text: str) -> dict[str, Any]:
    tokens = [
        token
        for token in WORD_RE.findall(text.lower())
        if len(token) > 3 and token not in STOPWORDS and token not in THEME_FILLER_WORDS
    ]
    token_set = set(tokens)
    if not token_set:
        return {
            "theme": GENERAL_THEME,
            "theme_confidence": "low",
            "theme_score": 0,
            "theme_margin": 0,
            "theme_matches": [],
        }

    scored: list[tuple[str, int, list[str]]] = []
    for theme, keywords in THEME_BUCKETS.items():
        matches = sorted(token_set & keywords)
        scored.append((theme, len(matches), matches))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_theme, best_score, best_matches = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0
    margin = best_score - second_score

    if best_score >= 3 and margin >= 2:
        confidence = "high"
    elif best_score >= 2 and margin >= 1:
        confidence = "medium"
    else:
        return {
            "theme": GENERAL_THEME,
            "theme_confidence": "low",
            "theme_score": best_score,
            "theme_margin": margin,
            "theme_matches": best_matches,
        }

    return {
        "theme": best_theme,
        "theme_confidence": confidence,
        "theme_score": best_score,
        "theme_margin": margin,
        "theme_matches": best_matches,
    }


def infer_theme(text: str, fallback_family: str = "") -> str:
    return infer_theme_info(text)["theme"]


def infer_theme_keywords(text: str, theme: str, max_keywords: int = 6) -> list[str]:
    """Return concrete content words from the verse that tie it to its inferred theme."""
    token_counts = Counter(
        token
        for token in WORD_RE.findall(text.lower())
        if len(token) > 3 and token not in STOPWORDS
    )
    theme_keywords = THEME_BUCKETS.get(theme, set())
    selected: list[str] = []

    for token, _count in token_counts.most_common():
        if token in theme_keywords and token not in selected:
            selected.append(token)
        if len(selected) >= max_keywords:
            return selected

    return selected


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def chunk_has_short_bars(
    chunk: list[str],
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
) -> bool:
    counts = [line_word_count(line) for line in chunk if line.strip()]
    if len(counts) != len(chunk):
        return False
    if max(counts, default=999) > hard_max_words_per_bar:
        return False
    avg_words = sum(counts) / len(counts) if counts else 999.0
    long_ratio = sum(1 for count in counts if count > max_words_per_bar) / len(counts) if counts else 1.0
    return avg_words <= max_avg_words_per_bar and long_ratio <= max_long_bar_ratio


def has_structural_training_noise(lines: list[str], max_parenthetical_lines: int = 2) -> bool:
    parenthetical_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            return True
        if "<|" in stripped or "|>" in stripped:
            return True
        if re.search(r"[^\x00-\x7F]{2,}", stripped):
            return True
        if BRACKET_LABEL_RE.match(stripped) or re.match(r"^\s*\{[^}]{1,80}\}\s*$", stripped):
            return True
        if STAGE_DIRECTION_RE.search(stripped):
            return True
        if QUOTED_DIALOGUE_RE.search(stripped):
            return True
        if re.fullmatch(r"\([^)]{1,80}\)", stripped):
            return True
        if ADLIB_RE.search(stripped):
            parenthetical_lines += 1
    return parenthetical_lines > max_parenthetical_lines


def verse_chunks(
    lines: list[str],
    verse_line_count: int,
    max_examples: int,
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    if len(lines) < verse_line_count:
        return chunks
    stride = verse_line_count
    for start in range(0, len(lines) - verse_line_count + 1, stride):
        chunk = lines[start:start + verse_line_count]
        joined = "\n".join(chunk)
        if has_enough_lyrics(joined) and chunk_has_short_bars(
            chunk,
            max_words_per_bar=max_words_per_bar,
            max_avg_words_per_bar=max_avg_words_per_bar,
            hard_max_words_per_bar=hard_max_words_per_bar,
            max_long_bar_ratio=max_long_bar_ratio,
        ):
            chunks.append(chunk)
        if len(chunks) >= max_examples:
            break
    return chunks


def clean_scene_chunk_score(chunk: list[str]) -> dict[str, Any]:
    text = "\n".join(chunk)
    tokens = WORD_RE.findall(text.lower())
    scene_hits = SCENE_KEYWORD_RE.findall(text)
    work_anchor_hits = WORK_ANCHOR_RE.findall(text)
    hard_bad = HARD_BAD_CONTENT_RE.findall(text)
    soft_drift = SOFT_DRIFT_RE.findall(text)
    bracket_labels = sum(1 for line in chunk if BRACKET_LABEL_RE.match(line.strip()))
    adlibs = ADLIB_RE.findall(text)
    repeated = len(chunk) - len({re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in chunk})
    return {
        "word_count": len(tokens),
        "scene_hit_count": len(scene_hits),
        "scene_keywords": sorted({hit.lower() for hit in scene_hits})[:10],
        "unique_scene_keyword_count": len({hit.lower() for hit in scene_hits}),
        "work_anchor_hit_count": len(work_anchor_hits),
        "unique_work_anchor_count": len({hit.lower() for hit in work_anchor_hits}),
        "hard_bad_count": len(hard_bad),
        "soft_drift_count": len(soft_drift),
        "bracket_label_count": bracket_labels,
        "adlib_count": len(adlibs),
        "repeated_line_count": repeated,
    }


def is_clean_scene_chunk(
    chunk: list[str],
    min_scene_keyword_hits: int,
    min_unique_scene_keywords: int,
    min_work_anchor_hits: int,
    min_unique_work_anchors: int,
    max_soft_drift_count: int,
    max_adlib_count: int,
) -> bool:
    score = clean_scene_chunk_score(chunk)
    if score["scene_hit_count"] < min_scene_keyword_hits:
        return False
    if score["unique_scene_keyword_count"] < min_unique_scene_keywords:
        return False
    if score["work_anchor_hit_count"] < min_work_anchor_hits:
        return False
    if score["unique_work_anchor_count"] < min_unique_work_anchors:
        return False
    if score["hard_bad_count"] > 0:
        return False
    if score["soft_drift_count"] > max_soft_drift_count:
        return False
    if score["bracket_label_count"] > 0:
        return False
    if score["adlib_count"] > max_adlib_count:
        return False
    if re.search(r"\[[^\]]{1,80}\]", "\n".join(chunk)):
        return False
    if score["repeated_line_count"] > 1:
        return False
    return True


def clean_scene_chunks(
    lines: list[str],
    verse_line_count: int,
    max_examples: int,
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
    min_scene_keyword_hits: int,
    min_unique_scene_keywords: int,
    min_work_anchor_hits: int,
    min_unique_work_anchors: int,
    max_soft_drift_count: int,
    max_adlib_count: int,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    candidate_sizes = sorted({8, 10, verse_line_count})
    for line_count in candidate_sizes:
        if len(lines) < line_count:
            continue
        stride = max(4, line_count // 2)
        for start in range(0, len(lines) - line_count + 1, stride):
            chunk = lines[start:start + line_count]
            total_words = sum(line_word_count(line) for line in chunk)
            if total_words < 45:
                continue
            if chunk_has_short_bars(
                chunk,
                max_words_per_bar=max_words_per_bar,
                max_avg_words_per_bar=max_avg_words_per_bar,
                hard_max_words_per_bar=hard_max_words_per_bar,
                max_long_bar_ratio=max_long_bar_ratio,
            ):
                chunks.append(chunk)
    clean_chunks = [
        chunk
        for chunk in chunks
        if is_clean_scene_chunk(
            chunk,
            min_scene_keyword_hits=min_scene_keyword_hits,
            min_unique_scene_keywords=min_unique_scene_keywords,
            min_work_anchor_hits=min_work_anchor_hits,
            min_unique_work_anchors=min_unique_work_anchors,
            max_soft_drift_count=max_soft_drift_count,
            max_adlib_count=max_adlib_count,
        )
    ]
    clean_chunks.sort(key=lambda chunk: clean_scene_chunk_score(chunk)["scene_hit_count"], reverse=True)
    return clean_chunks[:max_examples]


def architecture_control_text(metadata: dict[str, Any]) -> str:
    if not metadata.get("architecture"):
        return ""
    features = ", ".join(metadata.get("features", []))
    return (
        f"<|architecture|>{metadata['architecture']}\n"
        f"<|features|>{features}\n"
    )


def make_generate_song_example(row: dict[str, Any], seed_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    metadata = metadata_from_row(row, seed_index)
    lyrics = sanitize_lyrics_for_training(row.get("lyrics_model_text"))
    if not has_enough_lyrics(lyrics):
        return None
    training_text = (
        "<|task|>generate_song\n"
        f"<|title|>{metadata['title']}\n"
        f"<|artist|>{metadata['artist_clean']}\n"
        f"<|rap_family|>{metadata['rap_family']}\n"
        f"<|rap_category|>{metadata['rap_category']}\n"
        f"<|year|>{metadata['year']}\n"
        f"<|views_log|>{format_float(metadata['log_views'])}\n"
        f"{architecture_control_text(metadata)}"
        "<|lyrics|>\n"
        f"{format_annotated_lyrics(lyrics)}\n"
        "<|end|>"
    )
    return {
        "id": stable_id("generate_song", metadata["artist_clean"], metadata["title"], metadata["year"], lyrics),
        "task": "generate_song",
        "training_text": training_text,
        "metadata": metadata,
    }


def make_continue_verse_example(
    row: dict[str, Any],
    minimum_source_lines: int,
    split_line: int,
    seed_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    lyrics = sanitize_lyrics_for_training(row.get("lyrics_model_text"))
    if not has_enough_lyrics(lyrics):
        return None
    lines = [line for line in lyric_lines(lyrics) if line.strip()]
    if len(lines) < minimum_source_lines or len(lines) <= split_line:
        return None

    metadata = metadata_from_row(row, seed_index)
    prompt = "\n".join(lines[:split_line])
    target = "\n".join(lines[split_line:])
    if not prompt or not target:
        return None

    training_text = (
        "<|task|>continue_verse\n"
        f"<|artist|>{metadata['artist_clean']}\n"
        f"<|rap_family|>{metadata['rap_family']}\n"
        f"<|rap_category|>{metadata['rap_category']}\n"
        f"<|year|>{metadata['year']}\n"
        f"{architecture_control_text(metadata)}"
        "<|input_lyrics|>\n"
        f"{format_annotated_lyrics(prompt)}\n"
        "<|output_lyrics|>\n"
        f"{format_annotated_lyrics(target)}\n"
        "<|end|>"
    )
    return {
        "id": stable_id("continue_verse", metadata["artist_clean"], metadata["title"], metadata["year"], prompt, target),
        "task": "continue_verse",
        "training_text": training_text,
        "metadata": metadata,
    }


def make_generate_verse_examples(
    row: dict[str, Any],
    verse_line_count: int,
    max_examples: int,
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
    theme_keyword_count: int,
    min_theme_keyword_matches: int,
    strict_artifact_filter: bool,
    seed_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    lyrics = sanitize_lyrics_for_training(row.get("lyrics_model_text"))
    if not has_enough_lyrics(lyrics):
        return []
    lines = [line for line in lyric_lines(lyrics) if line.strip()]
    metadata = metadata_from_row(row, seed_index)
    examples: list[dict[str, Any]] = []
    chunks = verse_chunks(
        lines,
        verse_line_count=verse_line_count,
        max_examples=max_examples,
        max_words_per_bar=max_words_per_bar,
        max_avg_words_per_bar=max_avg_words_per_bar,
        hard_max_words_per_bar=hard_max_words_per_bar,
        max_long_bar_ratio=max_long_bar_ratio,
    )
    for index, chunk in enumerate(chunks, start=1):
        if strict_artifact_filter and has_structural_training_noise(chunk):
            continue
        verse = "\n".join(chunk)
        theme_info = infer_theme_info(verse)
        theme = theme_info["theme"]
        keywords = infer_theme_keywords(verse, theme, max_keywords=theme_keyword_count)
        if theme != GENERAL_THEME and len(keywords) < min_theme_keyword_matches:
            continue
        keyword_text = ", ".join(keywords)
        structure = f"{verse_line_count}-line verse, no chorus, no bracket labels"
        training_text = (
            "<|task|>generate_verse\n"
            f"<|title|>{metadata['title']}\n"
            f"<|artist|>{metadata['artist_clean']}\n"
            f"<|rap_family|>{metadata['rap_family']}\n"
            f"<|rap_category|>{metadata['rap_category']}\n"
            f"<|year|>{metadata['year']}\n"
            f"<|views_log|>{format_float(metadata['log_views'])}\n"
            f"{architecture_control_text(metadata)}"
            f"<|structure|>{structure}\n"
            f"<|max_words_per_bar|>{max_words_per_bar}\n"
            f"<|theme|>{theme}\n"
            f"<|theme_confidence|>{theme_info['theme_confidence']}\n"
            f"<|keywords|>{keyword_text}\n"
            f"<|rules|>{DEFAULT_VERSE_RULES}\n"
            "<|lyrics|>\n"
            f"{format_annotated_lyrics(verse)}\n"
            "<|end|>"
        )
        example_metadata = dict(metadata)
        example_metadata.update({
            "structure": structure,
            "max_words_per_bar": max_words_per_bar,
            "theme": theme,
            "theme_confidence": theme_info["theme_confidence"],
            "theme_score": theme_info["theme_score"],
            "theme_margin": theme_info["theme_margin"],
            "keywords": keywords,
            "verse_index": index,
        })
        examples.append(
            {
                "id": stable_id("generate_verse", metadata["artist_clean"], metadata["title"], metadata["year"], index, verse),
                "task": "generate_verse",
                "training_text": training_text,
                "metadata": example_metadata,
            }
        )
    return examples


def make_generate_clean_scene_verse_examples(
    row: dict[str, Any],
    verse_line_count: int,
    max_examples: int,
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
    min_scene_keyword_hits: int,
    min_unique_scene_keywords: int,
    min_work_anchor_hits: int,
    min_unique_work_anchors: int,
    max_soft_drift_count: int,
    max_adlib_count: int,
    seed_index: dict[tuple[str, str], dict[str, Any]],
    strict_artifact_filter: bool = False,
) -> list[dict[str, Any]]:
    lyrics = sanitize_lyrics_for_training(row.get("lyrics_model_text"))
    if not has_enough_lyrics(lyrics):
        return []
    lines = [line for line in lyric_lines(lyrics) if line.strip()]
    metadata = metadata_from_row(row, seed_index)
    examples: list[dict[str, Any]] = []
    chunks = clean_scene_chunks(
        lines,
        verse_line_count=verse_line_count,
        max_examples=max_examples,
        max_words_per_bar=max_words_per_bar,
        max_avg_words_per_bar=max_avg_words_per_bar,
        hard_max_words_per_bar=hard_max_words_per_bar,
        max_long_bar_ratio=max_long_bar_ratio,
        min_scene_keyword_hits=min_scene_keyword_hits,
        min_unique_scene_keywords=min_unique_scene_keywords,
        min_work_anchor_hits=min_work_anchor_hits,
        min_unique_work_anchors=min_unique_work_anchors,
        max_soft_drift_count=max_soft_drift_count,
        max_adlib_count=max_adlib_count,
    )
    for index, chunk in enumerate(chunks, start=1):
        if strict_artifact_filter and has_structural_training_noise(chunk):
            continue
        verse = "\n".join(chunk)
        score = clean_scene_chunk_score(chunk)
        keywords = score["scene_keywords"][:6]
        keyword_text = ", ".join(keywords)
        structure = f"{verse_line_count}-line clean verse, complete phrases, no chorus"
        scene = "working late, fatigue, city lights, money pressure, ambition"
        style = "grounded, clean, cinematic, first-person"
        avoid = "romance, clubs, violence, explicit content, ad-libs, dialogue, source notes"
        training_text = (
            "<|task|>generate_clean_scene_verse\n"
            f"<|title|>{metadata['title']}\n"
            f"<|artist|>{metadata['artist_clean']}\n"
            f"<|rap_family|>{metadata['rap_family']}\n"
            f"<|rap_category|>{metadata['rap_category']}\n"
            f"<|year|>{metadata['year']}\n"
            f"<|views_log|>{format_float(metadata['log_views'])}\n"
            f"{architecture_control_text(metadata)}"
            f"<|scene|>{scene}\n"
            f"<|keywords|>{keyword_text}\n"
            f"<|style|>{style}\n"
            f"<|structure|>{structure}\n"
            f"<|max_words_per_bar|>{max_words_per_bar}\n"
            f"<|avoid|>{avoid}\n"
            f"<|rules|>{DEFAULT_CLEAN_SCENE_RULES}\n"
            "<|lyrics|>\n"
            f"{format_annotated_lyrics(verse)}\n"
            "<|end|>"
        )
        example_metadata = dict(metadata)
        example_metadata.update({
            "scene": scene,
            "keywords": keywords,
            "style": style,
            "structure": structure,
            "max_words_per_bar": max_words_per_bar,
            "avoid": avoid,
            "scene_hit_count": score["scene_hit_count"],
            "verse_index": index,
        })
        examples.append(
            {
                "id": stable_id("generate_clean_scene_verse", metadata["artist_clean"], metadata["title"], metadata["year"], index, verse),
                "task": "generate_clean_scene_verse",
                "training_text": training_text,
                "metadata": example_metadata,
            }
        )
    return examples


def make_generate_architectural_verse_examples(
    row: dict[str, Any],
    verse_line_count: int,
    max_examples: int,
    max_words_per_bar: int,
    max_avg_words_per_bar: float,
    hard_max_words_per_bar: int,
    max_long_bar_ratio: float,
    seed_index: dict[tuple[str, str], dict[str, Any]],
    strict_artifact_filter: bool = False,
) -> list[dict[str, Any]]:
    metadata = metadata_from_row(row, seed_index)
    if not metadata.get("architecture"):
        return []
    lyrics = sanitize_lyrics_for_training(row.get("lyrics_model_text"))
    if not has_enough_lyrics(lyrics):
        return []
    lines = [line for line in lyric_lines(lyrics) if line.strip()]
    chunks: list[list[str]] = []
    for line_count in sorted({8, 10, verse_line_count}):
        if len(lines) < line_count:
            continue
        stride = max(4, line_count // 2)
        for start in range(0, len(lines) - line_count + 1, stride):
            chunk = lines[start:start + line_count]
            if sum(line_word_count(line) for line in chunk) < 55:
                continue
            if chunk_has_short_bars(
                chunk,
                max_words_per_bar=max_words_per_bar + 3,
                max_avg_words_per_bar=max_avg_words_per_bar + 3,
                hard_max_words_per_bar=hard_max_words_per_bar + 8,
                max_long_bar_ratio=max(max_long_bar_ratio, 0.5),
            ):
                chunks.append(chunk)
            if len(chunks) >= max_examples:
                break
        if len(chunks) >= max_examples:
            break
    examples: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if strict_artifact_filter and has_structural_training_noise(chunk):
            continue
        verse = "\n".join(chunk)
        features = ", ".join(metadata.get("features", []))
        structure = f"{verse_line_count}-line verse, architecture-guided, no chorus"
        training_text = (
            "<|task|>generate_architectural_verse\n"
            f"<|title|>{metadata['title']}\n"
            f"<|artist|>{metadata['artist_clean']}\n"
            f"<|rap_family|>{metadata['rap_family']}\n"
            f"<|rap_category|>{metadata['rap_category']}\n"
            f"<|year|>{metadata['year']}\n"
            f"<|views_log|>{format_float(metadata['log_views'])}\n"
            f"{architecture_control_text(metadata)}"
            f"<|structure|>{structure}\n"
            f"<|max_words_per_bar|>{max_words_per_bar}\n"
            "<|rules|>Write only original lyrics. Preserve coherent structure, dense imagery, and complete line breaks. Avoid filler bars and source notes.\n"
            "<|lyrics|>\n"
            f"{format_annotated_lyrics(verse)}\n"
            "<|end|>"
        )
        example_metadata = dict(metadata)
        example_metadata.update({
            "structure": structure,
            "max_words_per_bar": max_words_per_bar,
            "features_text": features,
            "verse_index": index,
        })
        examples.append(
            {
                "id": stable_id("generate_architectural_verse", metadata["artist_clean"], metadata["title"], metadata["year"], index, verse),
                "task": "generate_architectural_verse",
                "training_text": training_text,
                "metadata": example_metadata,
            }
        )
    return examples


def read_verse_sections(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Verse section dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        return pl.read_ndjson(path)
    raise ValueError(f"Unsupported verse section format: {path.suffix}. Use .parquet or .jsonl")


def make_section_verse_example(
    row: dict[str, Any],
    seed_index: dict[tuple[str, str], dict[str, Any]],
    bar_count_range: str,
    strict_artifact_filter: bool = False,
) -> dict[str, Any] | None:
    verse = sanitize_lyrics_for_training(row.get("verse_text"))
    lines = [line for line in lyric_lines(verse) if line.strip()]
    if not lines:
        return None
    if strict_artifact_filter and has_structural_training_noise(lines):
        return None
    bar_count = int(row.get("bar_count") or len(lines))
    if bar_count != len(lines):
        bar_count = len(lines)
    if not bool(row.get("is_training_length", True)):
        return None

    metadata = metadata_from_row(row, seed_index)
    inferred_theme_info = infer_theme_info(verse)
    row_theme = clean_control_value(row.get("theme"))
    row_confidence = clean_control_value(row.get("theme_confidence"))
    theme_info = {
        "theme": row_theme or inferred_theme_info["theme"],
        "theme_confidence": row_confidence or inferred_theme_info["theme_confidence"],
        "theme_score": int(row.get("theme_score") or inferred_theme_info["theme_score"]),
        "theme_margin": int(row.get("theme_margin") or inferred_theme_info["theme_margin"]),
        "theme_matches": inferred_theme_info.get("theme_matches", []),
    }
    theme = theme_info["theme"]
    row_keywords = row.get("theme_keywords")
    if isinstance(row_keywords, list):
        keywords = [clean_control_value(keyword) for keyword in row_keywords if clean_control_value(keyword)]
    elif row_keywords:
        keywords = [clean_control_value(keyword) for keyword in str(row_keywords).split(",") if clean_control_value(keyword)]
    else:
        keywords = infer_theme_keywords(verse, theme, max_keywords=8)
    keyword_text = ", ".join(keywords)
    structure = f"{bar_count}-bar full verse, no chorus, no hook"
    rules = (
        "Write only original verse lyrics. Keep line breaks. Do not include chorus, hook, intro, outro, "
        "section labels, source notes, or explanations."
    )
    training_text = (
        "<|task|>generate_verse\n"
        f"<|title|>{metadata['title']}\n"
        f"<|artist|>{metadata['artist_clean']}\n"
        f"<|rap_family|>{metadata['rap_family']}\n"
        f"<|rap_category|>{metadata['rap_category']}\n"
        f"<|year|>{metadata['year']}\n"
        f"<|views_log|>{format_float(metadata['log_views'])}\n"
        f"{architecture_control_text(metadata)}"
        f"<|structure|>{structure}\n"
        f"<|bar_count|>{bar_count}\n"
        f"<|target_bars|>{bar_count}\n"
        f"<|bar_count_range|>{bar_count_range}\n"
        f"<|theme|>{theme}\n"
        f"<|theme_confidence|>{theme_info['theme_confidence']}\n"
        f"<|keywords|>{keyword_text}\n"
        f"<|section_source|>{clean_control_value(row.get('section_label'))}\n"
        f"<|rules|>{rules}\n"
        "<|lyrics|>\n"
        f"{format_annotated_lyrics(verse)}\n"
        "<|end|>"
    )
    example_metadata = dict(metadata)
    example_metadata.update(
        {
            "structure": structure,
            "bar_count": bar_count,
            "target_bars": bar_count,
            "bar_count_range": bar_count_range,
            "theme": theme,
            "theme_confidence": theme_info["theme_confidence"],
            "theme_score": theme_info["theme_score"],
            "theme_margin": theme_info["theme_margin"],
            "keywords": keywords,
            "section_label": clean_control_value(row.get("section_label")),
            "section_index": int(row.get("section_index") or 0),
            "section_source": "bracket_verse_header",
            "is_full_verse": bool(row.get("is_full_verse", True)),
            "avg_words_per_bar": round(float(row.get("avg_words_per_bar") or 0.0), 4),
            "max_words_per_bar": int(row.get("max_words_per_bar") or 0),
            "repeated_line_ratio": round(float(row.get("repeated_line_ratio") or 0.0), 4),
        }
    )
    return {
        "id": stable_id(
            "generate_section_verse",
            metadata["artist_clean"],
            metadata["title"],
            metadata["year"],
            example_metadata["section_index"],
            verse,
        ),
        "task": "generate_verse",
        "training_text": training_text,
        "metadata": example_metadata,
    }


def build_section_verse_examples(config: dict[str, Any], seed_index: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    section_path_value = config.get("verse_section_source")
    if not section_path_value:
        return []
    section_path = Path(section_path_value)
    min_bars = int(config.get("section_verse_min_bars", 10))
    max_bars = int(config.get("section_verse_max_bars", 25))
    max_examples = int(config.get("max_section_verse_examples", 0))
    strict_artifact_filter = bool(config.get("strict_verse_artifact_filter", False))
    bar_count_range = f"{min_bars}-{max_bars}"
    df = read_verse_sections(section_path)
    required = {"verse_text", "title", "artist_clean", "year", "views", "log_views", "rap_category", "rap_family", "bar_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Verse section dataset is missing required column(s): {missing}")
    df = df.filter(
        (pl.col("bar_count") >= min_bars)
        & (pl.col("bar_count") <= max_bars)
        & pl.col("passes_quality_filter")
        & pl.col("is_training_length")
    )
    if max_examples > 0:
        df = df.head(max_examples)
    examples: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        example = make_section_verse_example(row, seed_index, bar_count_range, strict_artifact_filter)
        if example is not None:
            examples.append(example)
    return balance_section_theme_examples(examples, config)


def balance_section_theme_examples(examples: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cap broad full-verse buckets and repeat theme-confident examples deterministically."""
    if not examples:
        return examples

    rng = random.Random(int(config.get("random_seed", 42)))
    max_general = int(config.get("max_general_vibe_section_examples", 0))
    min_theme_confidence = clean_control_value(config.get("section_theme_min_confidence"))
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = confidence_rank.get(min_theme_confidence, 0) if min_theme_confidence else 0

    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        metadata = example.get("metadata", {})
        theme = metadata.get("theme") or GENERAL_THEME
        confidence = metadata.get("theme_confidence") or "low"
        if theme != GENERAL_THEME and confidence_rank.get(confidence, 0) < min_rank:
            continue
        grouped.setdefault(theme, []).append(example)

    selected: list[dict[str, Any]] = []
    for theme, items in grouped.items():
        if theme == GENERAL_THEME and max_general > 0 and len(items) > max_general:
            items = list(items)
            rng.shuffle(items)
            items = items[:max_general]
        selected.extend(items)

    repeat_counts = config.get("section_theme_repeat_counts", {})
    if not repeat_counts:
        return selected

    expanded: list[dict[str, Any]] = []
    for example in selected:
        expanded.append(example)
        theme = example.get("metadata", {}).get("theme") or GENERAL_THEME
        repeat_count = int(repeat_counts.get(theme, 1))
        for repeat_index in range(2, repeat_count + 1):
            repeated = {
                "id": stable_id(example["id"], "theme_repeat", repeat_index),
                "task": example["task"],
                "training_text": example["training_text"],
                "metadata": dict(example["metadata"]),
            }
            repeated["metadata"]["theme_repeat_index"] = repeat_index
            expanded.append(repeated)
    return expanded


def build_examples(df: pl.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = set(config.get("tasks", []))
    architecture_seed_path = Path(config.get("architecture_seed_path", DEFAULT_ARCHITECTURE_SEEDS_PATH))
    seed_index = load_architecture_seeds(architecture_seed_path)
    minimum_source_lines = int(config.get("minimum_continuation_source_lines", 16))
    split_line = int(config.get("continuation_split_line", 8))
    verse_line_count = int(config.get("verse_line_count", 12))
    max_verse_examples = int(config.get("max_verse_examples_per_song", 2))
    max_words_per_bar = int(config.get("max_words_per_bar", 9))
    max_avg_words_per_bar = float(config.get("max_avg_words_per_bar", 11))
    hard_max_words_per_bar = int(config.get("hard_max_words_per_bar", 16))
    max_long_bar_ratio = float(config.get("max_long_bar_ratio", 0.25))
    theme_keyword_count = int(config.get("theme_keyword_count", 6))
    min_theme_keyword_matches = int(config.get("min_theme_keyword_matches", 2))
    clean_scene_line_count = int(config.get("clean_scene_line_count", verse_line_count))
    max_clean_scene_examples = int(config.get("max_clean_scene_examples_per_song", 1))
    architecture_verse_line_count = int(config.get("architecture_verse_line_count", verse_line_count))
    max_architecture_verse_examples = int(config.get("max_architecture_verse_examples_per_song", 2))
    min_clean_scene_keyword_hits = int(config.get("min_clean_scene_keyword_hits", 4))
    min_unique_scene_keywords = int(config.get("min_unique_clean_scene_keywords", 2))
    min_work_anchor_hits = int(config.get("min_clean_scene_work_anchor_hits", 2))
    min_unique_work_anchors = int(config.get("min_unique_clean_scene_work_anchors", 2))
    max_clean_scene_soft_drift_count = int(config.get("max_clean_scene_soft_drift_count", 0))
    max_clean_scene_adlib_count = int(config.get("max_clean_scene_adlib_count", 0))
    strict_artifact_filter = bool(config.get("strict_verse_artifact_filter", False))
    include_corpus_verse_chunks = bool(config.get("include_corpus_verse_chunks", True))
    include_corpus_song_examples = bool(config.get("include_corpus_song_examples", True))
    include_corpus_continuation_examples = bool(config.get("include_corpus_continuation_examples", True))

    examples: list[dict[str, Any]] = []
    needs_corpus_loop = (
        ("generate_song" in tasks and include_corpus_song_examples)
        or ("continue_verse" in tasks and include_corpus_continuation_examples)
        or ("generate_verse" in tasks and include_corpus_verse_chunks)
        or ("generate_clean_scene_verse" in tasks)
        or ("generate_architectural_verse" in tasks)
    )
    if needs_corpus_loop:
        for row in df.iter_rows(named=True):
            if "generate_song" in tasks and include_corpus_song_examples:
                example = make_generate_song_example(row, seed_index)
                if example is not None:
                    examples.append(example)
            if "continue_verse" in tasks and include_corpus_continuation_examples:
                example = make_continue_verse_example(row, minimum_source_lines, split_line, seed_index)
                if example is not None:
                    examples.append(example)
            if "generate_verse" in tasks and include_corpus_verse_chunks:
                examples.extend(
                    make_generate_verse_examples(
                        row,
                        verse_line_count,
                        max_verse_examples,
                        max_words_per_bar,
                        max_avg_words_per_bar,
                        hard_max_words_per_bar,
                        max_long_bar_ratio,
                        theme_keyword_count,
                        min_theme_keyword_matches,
                        strict_artifact_filter,
                        seed_index,
                    )
                )
            if "generate_clean_scene_verse" in tasks:
                family = clean_control_value(row.get("rap_family")).lower()
                if "trap" in family or "street" in family or "drill" in family or "mainstream" in family:
                    continue
                examples.extend(
                    make_generate_clean_scene_verse_examples(
                        row,
                        clean_scene_line_count,
                        max_clean_scene_examples,
                        max_words_per_bar,
                        max_avg_words_per_bar,
                        hard_max_words_per_bar,
                        max_long_bar_ratio,
                        min_clean_scene_keyword_hits,
                        min_unique_scene_keywords,
                        min_work_anchor_hits,
                        min_unique_work_anchors,
                        max_clean_scene_soft_drift_count,
                        max_clean_scene_adlib_count,
                        seed_index,
                        strict_artifact_filter,
                    )
                )
            if "generate_architectural_verse" in tasks:
                examples.extend(
                    make_generate_architectural_verse_examples(
                        row,
                        architecture_verse_line_count,
                        max_architecture_verse_examples,
                        max_words_per_bar,
                        max_avg_words_per_bar,
                        hard_max_words_per_bar,
                        max_long_bar_ratio,
                        seed_index,
                        strict_artifact_filter,
                    )
                )
    if config.get("verse_section_source"):
        examples.extend(build_section_verse_examples(config, seed_index))
    return apply_task_repeats(examples, config.get("task_repeat_counts", {}))


def apply_task_repeats(examples: list[dict[str, Any]], repeat_counts: dict[str, Any]) -> list[dict[str, Any]]:
    if not repeat_counts:
        return examples
    expanded: list[dict[str, Any]] = []
    for example in examples:
        expanded.append(example)
        repeat_count = int(repeat_counts.get(example["task"], 1))
        for repeat_index in range(2, repeat_count + 1):
            repeated = {
                "id": stable_id(example["id"], "repeat", repeat_index),
                "task": example["task"],
                "training_text": example["training_text"],
                "metadata": dict(example["metadata"]),
            }
            repeated["metadata"]["repeat_index"] = repeat_index
            expanded.append(repeated)
    return expanded


def assign_split(index: int, total: int, split: dict[str, float]) -> str:
    train_end = int(total * float(split["train"]))
    validation_end = train_end + int(total * float(split["validation"]))
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def build_manifest(
    config: dict[str, Any],
    source_path: Path,
    output_dir: Path,
    examples: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    task_counts = Counter(example["task"] for example in examples)
    theme_counts = Counter(
        example.get("metadata", {}).get("theme")
        for example in examples
        if example.get("task") == "generate_verse" and example.get("metadata", {}).get("theme")
    )
    theme_confidence_counts = Counter(
        example.get("metadata", {}).get("theme_confidence")
        for example in examples
        if example.get("task") == "generate_verse" and example.get("metadata", {}).get("theme_confidence")
    )
    file_counts = {
        split_name: count_jsonl(output_dir / f"{split_name}.jsonl")
        for split_name in ["train", "validation", "test"]
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "random_seed": int(config.get("random_seed", 42)),
        "split_ratios": config["split"],
        "tasks": config.get("tasks", []),
        "required_columns": REQUIRED_COLUMNS,
        "lyric_annotation": {
            "enabled": True,
            "mode": "tokenizer_special_tokens",
            "verse_start": VERSE_START_TOKEN,
            "verse_end": VERSE_END_TOKEN,
            "bar_start": BAR_START_TOKEN,
            "special_tokens": STRUCTURAL_SPECIAL_TOKENS,
        },
        "total_examples": len(examples),
        "task_counts": dict(sorted(task_counts.items())),
        "generate_verse_theme_counts": dict(theme_counts.most_common()),
        "generate_verse_theme_confidence_counts": dict(theme_confidence_counts.most_common()),
        "generate_verse_bar_count_counts": dict(
            Counter(
                example.get("metadata", {}).get("bar_count")
                for example in examples
                if example.get("task") == "generate_verse" and example.get("metadata", {}).get("bar_count")
            ).most_common()
        ),
        "split_counts": {name: len(records) for name, records in split_records.items()},
        "file_counts": file_counts,
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "validation": str(output_dir / "validation.jsonl"),
            "test": str(output_dir / "test.jsonl"),
        },
    }


def main() -> None:
    args = parse_args()
    config = resolve_settings(args)
    validate_config(config)

    source_path = Path(config["source"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_source(source_path)
    examples = build_examples(df, config)
    random.Random(int(config.get("random_seed", 42))).shuffle(examples)

    split_records = {"train": [], "validation": [], "test": []}
    for index, example in enumerate(examples):
        split_name = assign_split(index, len(examples), config["split"])
        split_records[split_name].append(example)

    for split_name, records in split_records.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", records)

    manifest = build_manifest(config, source_path, output_dir, examples, split_records)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({key: manifest[key] for key in ["total_examples", "task_counts", "split_counts"]}, indent=2))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
