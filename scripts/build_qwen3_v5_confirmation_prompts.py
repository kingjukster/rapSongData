#!/usr/bin/env python3
"""Freeze the independent 24-prompt v5 family confirmation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THEMES = (
    "repairing a bicycle chain beneath an apartment stairwell",
    "waiting in a laundromat while a thunderstorm floods the street",
    "delivering a birthday cake across town on a crowded bus",
    "finding a stray dog beside a closed gas station at dawn",
    "helping a neighbor move furniture before an eviction deadline",
    "returning a borrowed camera after discovering undeveloped film",
)
FAMILIES = {
    "melodic": ("Use melodic but compact phrasing.", "focused_12line_melodic"),
    "story": ("Make every line move the same scene forward.", "focused_12line_story"),
    "technical": ("Use internal and multisyllabic rhymes, but keep the meaning clear.", "focused_12line_technical"),
    "clean": ("Use clean, radio-safe phrasing with no profanity or slurs.", "focused_12line_clean"),
}


def build_prompts() -> list[dict[str, object]]:
    rows = []
    for theme in THEMES:
        for family, (instruction, prompt_family) in FAMILIES.items():
            clean_prefix = "clean, radio-safe " if family == "clean" else ""
            rows.append({
                "prompt": (
                    f"Write exactly 12 lines of {clean_prefix}original rap lyrics about {theme}. "
                    f"{instruction} Keep the ending complete and declarative. No intro, no commentary."
                ),
                "theme": theme, "style": instruction.rstrip("."), "target_line_count": 12,
                "prompt_family": prompt_family,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/prompts/qwen3_4b_12line_v5_confirmation_prompts.json"))
    args = parser.parse_args()
    rows = build_prompts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prompts": len(rows), "themes": len(THEMES), "families": len(FAMILIES)}, indent=2))


if __name__ == "__main__":
    main()
