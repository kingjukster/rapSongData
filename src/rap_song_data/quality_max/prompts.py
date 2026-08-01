"""Prompt contracts for the quality-max research lane."""

from __future__ import annotations

from typing import Literal


Mode = Literal["generate", "style_transfer", "mutate", "improve"]

SYSTEM_PROMPT = """You are an expert rap lyricist and editor. Produce only original lyrics.
Never copy, quote, or closely paraphrase an existing song. When an artist is named,
translate the request into high-level musical and writing traits rather than imitating
recognizable lyrics, signature phrases, or a specific copyrighted song. Return lyrics
only, with one bar per line and no headings, analysis, or commentary."""


def _soft_length_instruction(min_bars: int, max_bars: int, target_bars: int | None) -> str:
    if min_bars < 1 or max_bars < min_bars:
        raise ValueError("Expected 1 <= min_bars <= max_bars")
    if target_bars is not None and target_bars < 1:
        raise ValueError("target_bars must be positive when supplied")
    center = f" Aim for about {target_bars} bars." if target_bars else ""
    return (
        f"Use roughly {min_bars}-{max_bars} bars, allowing the writing to end naturally."
        f"{center} This is a creative target, not an exact line-count requirement."
    )


def build_messages(
    *,
    mode: Mode,
    theme: str,
    style: str,
    keywords: str,
    source_text: str = "",
    artist_reference: str = "",
    min_bars: int = 12,
    max_bars: int = 36,
    target_bars: int | None = None,
) -> list[dict[str, str]]:
    """Build a model-neutral chat request with soft structural targets."""
    length_instruction = _soft_length_instruction(min_bars, max_bars, target_bars)
    shared = (
        f"Theme or intent: {theme.strip() or 'open-ended'}.\n"
        f"Writing traits: {style.strip() or 'vivid, coherent, technically strong rap writing'}.\n"
        f"Useful words or images: {keywords.strip() or 'none required'}.\n"
        f"{length_instruction}\n"
        "Favor concrete imagery, coherent progression, layered rhyme, natural cadence, "
        "memorable phrasing, and a complete ending. Avoid filler, scraped-text artifacts, "
        "bracket labels, and explanations."
    )

    if mode == "generate":
        task = f"Write a new original verse.\n{shared}"
    elif mode == "style_transfer":
        if not source_text.strip():
            raise ValueError("style_transfer requires source_text")
        reference = artist_reference.strip() or "the requested style description"
        task = (
            "Rewrite the source into a fresh original verse while preserving its core meaning. "
            f"Use only high-level traits associated with {reference}; do not reproduce signature "
            "phrases, recognizable lyrics, or the wording of the source.\n"
            f"{shared}\nSOURCE TO TRANSFORM:\n{source_text.strip()}"
        )
    elif mode == "mutate":
        if not source_text.strip():
            raise ValueError("mutate requires source_text")
        task = (
            "Create a substantial mutation of the source. Preserve its central premise while "
            "changing imagery, rhyme paths, syntax, punchlines, and cadence. The result must stand "
            f"alone as original writing.\n{shared}\nSOURCE TO MUTATE:\n{source_text.strip()}"
        )
    elif mode == "improve":
        if not source_text.strip():
            raise ValueError("improve requires source_text")
        task = (
            "Rewrite and improve the source. Keep its strongest ideas, repair weak or generic bars, "
            "tighten narrative continuity, strengthen rhyme and cadence, and finish cleanly. Do not "
            f"explain the edits.\n{shared}\nSOURCE TO IMPROVE:\n{source_text.strip()}"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def build_revision_messages(
    *,
    lyrics: str,
    theme: str,
    style: str,
    keywords: str,
    min_bars: int,
    max_bars: int,
    target_bars: int | None,
) -> list[dict[str, str]]:
    """Request a second-pass edit of a promising candidate."""
    return build_messages(
        mode="improve",
        theme=theme,
        style=style,
        keywords=keywords,
        source_text=lyrics,
        min_bars=min_bars,
        max_bars=max_bars,
        target_bars=target_bars,
    )
