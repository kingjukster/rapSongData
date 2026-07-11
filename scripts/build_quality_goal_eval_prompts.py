#!/usr/bin/env python3
"""Build locked development and confirmation prompt banks for quality evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEV_THEMES = [
    "catching the first bus before sunrise after a double shift",
    "fixing a bicycle chain in the rain before a delivery",
    "hearing an old voicemail while packing to move",
    "returning borrowed tools after finishing a difficult repair",
    "coaching a cousin before their first job interview",
    "waiting out a summer blackout on the front stoop",
    "cleaning an empty venue after a small hometown show",
    "finding a handwritten recipe after the family kitchen is sold",
    "refusing a shortcut that would betray a teammate",
    "watching the first snow from a late train",
    "making a face-to-face apology after months of silence",
    "saving rent money while passing up a flashy purchase",
]

CONFIRMATION_THEMES = [
    "returning a lost wallet on the last subway",
    "reopening a neighborhood basketball court after winter",
    "finishing night class while friends celebrate downtown",
    "helping a parent learn a new phone before an early flight",
    "taking a wrong road-trip exit that leads to a quiet lake",
    "closing a food truck after a rain-soaked festival",
]

DEV_FAMILIES = [
    (
        "continuous_scene",
        "Write exactly 12 lines of original rap lyrics about {theme}. Keep one continuous scene moving forward and include at least three concrete sensory details. End with a complete declarative payoff. No intro or commentary.",
    ),
    (
        "compact_melodic",
        "Write exactly 12 compact, melodic lines of original rap lyrics about {theme}. Keep the imagery specific, the phrasing natural, and the final line decisive. Return lyrics only.",
    ),
    (
        "internal_slant_rhyme",
        "Write exactly 12 lines of original rap lyrics about {theme}. Use clear internal and slant rhymes without sacrificing meaning or scene coherence. Finish with a strong declarative line. Return lyrics only.",
    ),
    (
        "clean_payoff",
        "Write exactly 12 clean, radio-safe lines of original rap lyrics about {theme}. Avoid profanity and slurs, use concrete images, and land a memorable but natural payoff. No intro or commentary.",
    ),
]

CONFIRMATION_FAMILIES = [
    (
        "scene_paraphrase",
        "Tell the story of {theme} as exactly 12 original rap lines. Let each bar advance the same moment, name three things that can be seen, heard, or felt, and close with a firm statement. Lyrics only.",
    ),
    (
        "melodic_paraphrase",
        "Give me exactly 12 short melodic rap lines centered on {theme}. Keep the language original and vivid, maintain one emotional thread, and resolve it in the last line. Output only the verse.",
    ),
    (
        "rhyme_paraphrase",
        "Create an original 12-line rap verse about {theme}. Weave internal or slant rhyme through clear sentences, hold the scene together, and make line 12 feel earned and final. No preface.",
    ),
    (
        "safe_paraphrase",
        "In exactly 12 radio-safe rap lines, portray {theme}. Use precise sensory details, no profanity or slurs, no generic motivation slogans, and finish on a complete payoff. Return only lyrics.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/prompts"))
    return parser.parse_args()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def build_bank(split: str, themes: list[str], families: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme_index, theme in enumerate(themes):
        theme_id = f"{split}-theme-{theme_index + 1:02d}-{stable_id(theme)[:6]}"
        for family, template in families:
            prompt = template.format(theme=theme)
            rows.append(
                {
                    "prompt_key": stable_id(prompt),
                    "prompt": prompt,
                    "theme_id": theme_id,
                    "theme": theme,
                    "instruction_family": family,
                    "prompt_family": f"quality_goal_{family}",
                    "target_line_count": 12,
                    "evaluation_split": split,
                    "samples_per_model": 2,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = {
        "quality_goal_dev_prompts.json": build_bank("development", DEV_THEMES, DEV_FAMILIES),
        "quality_goal_confirmation_prompts.json": build_bank(
            "confirmation", CONFIRMATION_THEMES, CONFIRMATION_FAMILIES
        ),
    }
    for name, rows in banks.items():
        path = args.output_dir / name
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"path": str(path), "prompts": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
