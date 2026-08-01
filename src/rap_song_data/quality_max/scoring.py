"""Transparent, local pre-ranking for lyric candidates.

The score is deliberately heuristic. It narrows a multi-candidate pool for human or
model review; it is not treated as a ground-truth quality judgment.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
LABEL_RE = re.compile(r"^\s*(?:\[.*?]|(?:verse|hook|chorus|bridge|intro|outro)\s*:?\s*)$", re.I)
ARTIFACT_RE = re.compile(r"(?:https?://|genius\.com|lyrics taken from|you might also like|embed\s*$)", re.I)
SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>|</?think>", re.I | re.S)
MOJIBAKE_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "â€”": "—",
    "â€“": "–",
    "â€¦": "…",
    "Â": "",
}


def lyric_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def clean_lyrics(text: str) -> str:
    """Remove model wrappers without enforcing or truncating a bar count."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = SPECIAL_TOKEN_RE.sub("", text)
    for broken, repaired in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(broken, repaired)
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`")
        if not line or LABEL_RE.fullmatch(line):
            continue
        if ARTIFACT_RE.search(line):
            continue
        line = re.sub(r"^\s*(?:verse|hook|chorus|bridge|intro|outro)\s*:\s*", "", line, flags=re.I)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + size]) for index in range(max(0, len(tokens) - size + 1))]


def _soft_range_score(value: int, minimum: int, maximum: int) -> float:
    if minimum <= value <= maximum:
        return 1.0
    distance = minimum - value if value < minimum else value - maximum
    return math.exp(-distance / max(2.0, (maximum - minimum + 1) / 3.0))


def _keyword_score(tokens: list[str], keywords: str) -> float:
    requested = {token for token in _tokens(keywords) if len(token) > 2}
    if not requested:
        return 1.0
    present = set(tokens)
    return len(requested & present) / len(requested)


def _rhyme_cohesion(lines: list[str]) -> float:
    endings = []
    for line in lines:
        words = _tokens(line)
        if words:
            endings.append(words[-1][-3:])
    if len(endings) < 2:
        return 0.0
    counts = Counter(endings)
    matched = sum(count for count in counts.values() if count > 1)
    density = matched / len(endings)
    return min(1.0, density / 0.65)


def score_candidate(
    text: str,
    *,
    min_bars: int,
    max_bars: int,
    keywords: str = "",
    hit_token_cap: bool = False,
) -> dict[str, Any]:
    """Return component scores and a weighted local pre-rank score."""
    lines = lyric_lines(text)
    tokens = _tokens(text)
    normalized_lines = [" ".join(_tokens(line)) for line in lines]
    unique_line_ratio = len(set(normalized_lines)) / max(1, len(normalized_lines))
    lexical_diversity = len(set(tokens)) / max(1, len(tokens))
    trigrams = _ngrams(tokens, 3)
    repeated_trigram_ratio = (
        sum(count - 1 for count in Counter(trigrams).values() if count > 1) / max(1, len(trigrams))
    )
    artifact_free = 0.0 if ARTIFACT_RE.search(text) or "<|" in text else 1.0
    complete_ending = 1.0
    if lines:
        last_words = _tokens(lines[-1])
        if len(last_words) < 4 or lines[-1].endswith((",", ":", ";", "-")):
            complete_ending = 0.35
    else:
        complete_ending = 0.0
    if hit_token_cap:
        complete_ending = 0.0

    components = {
        "bar_range": _soft_range_score(len(lines), min_bars, max_bars),
        "unique_lines": unique_line_ratio,
        "lexical_diversity": min(1.0, lexical_diversity / 0.62),
        "low_ngram_repetition": max(0.0, 1.0 - repeated_trigram_ratio * 4.0),
        "keyword_adherence": _keyword_score(tokens, keywords),
        "rhyme_cohesion": _rhyme_cohesion(lines),
        "artifact_free": artifact_free,
        "complete_ending": complete_ending,
    }
    weights = {
        "bar_range": 0.12,
        "unique_lines": 0.16,
        "lexical_diversity": 0.16,
        "low_ngram_repetition": 0.16,
        "keyword_adherence": 0.12,
        "rhyme_cohesion": 0.10,
        "artifact_free": 0.10,
        "complete_ending": 0.08,
    }
    total = sum(components[name] * weights[name] for name in weights)
    return {
        "score": round(total, 6),
        "bar_count": len(lines),
        "word_count": len(tokens),
        "outside_soft_range": not (min_bars <= len(lines) <= max_bars),
        "hit_token_cap": bool(hit_token_cap),
        "components": {name: round(value, 6) for name, value in components.items()},
    }


def rank_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    min_bars: int,
    max_bars: int,
    keywords: str = "",
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        record = dict(candidate)
        record["metrics"] = score_candidate(
            str(record.get("lyrics") or ""),
            min_bars=min_bars,
            max_bars=max_bars,
            keywords=keywords,
            hit_token_cap=bool(record.get("hit_token_cap", False)),
        )
        ranked.append(record)
    ranked.sort(key=lambda item: (-float(item["metrics"]["score"]), int(item.get("candidate_index", 0))))
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank
    return ranked
