#!/usr/bin/env python3
"""Build fixed Qwen3 rap generation evaluation prompt sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THEMES = [
    "ambition after public failure",
    "pressure, loyalty, and temptation",
    "late-night city focus after a long shift",
    "family pride and clean success",
    "loneliness after a breakthrough",
    "discipline, patience, and technical growth",
    "rebuilding a name after rumors",
    "standing outside a closed corner store at 2 AM",
    "turning rejection into a sharper routine",
    "keeping promises while the city moves fast",
    "a first show in a half-empty room",
    "staying calm while friends chase shortcuts",
    "repairing trust after a public mistake",
    "walking home under elevated train tracks",
    "choosing craft over clout during a slow year",
]


PROMPT_VARIANTS = [
    {
        "style": "melodic but compact phrasing",
        "family": "focused_12line_melodic",
        "instruction": (
            "Write exactly 12 lines of original rap lyrics about {theme}. "
            "Use melodic but compact phrasing. Keep the ending complete and declarative. "
            "No intro, no commentary."
        ),
    },
    {
        "style": "gritty scene-driven storytelling",
        "family": "focused_12line_story",
        "instruction": (
            "Write exactly 12 lines of original rap lyrics about {theme}. "
            "Make every line move the same scene forward. Keep the ending complete and declarative. "
            "No intro, no commentary."
        ),
    },
    {
        "style": "technical internal rhyme with clear meaning",
        "family": "focused_12line_technical",
        "instruction": (
            "Write exactly 12 lines of original rap lyrics about {theme}. "
            "Use internal rhymes, but keep the meaning clear. Keep the ending complete and declarative. "
            "No intro, no commentary."
        ),
    },
    {
        "style": "clean radio-safe phrasing",
        "family": "focused_12line_clean",
        "instruction": (
            "Write exactly 12 lines of clean, radio-safe original rap lyrics about {theme}. "
            "Avoid profanity and slurs. Keep the ending complete and declarative. "
            "No intro, no commentary."
        ),
    },
]


def build_prompts() -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    seen: set[str] = set()
    for theme in THEMES:
        for variant in PROMPT_VARIANTS:
            prompt = str(variant["instruction"]).format(theme=theme)
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(
                {
                    "prompt": prompt,
                    "theme": theme,
                    "style": variant["style"],
                    "target_line_count": 12,
                    "prompt_family": variant["family"],
                }
            )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/prompts/qwen3_4b_12line_expanded_eval_prompts.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = build_prompts()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "prompt_count": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
