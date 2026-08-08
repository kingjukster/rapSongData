#!/usr/bin/env python3
"""Build heldout-safe technical and clean prompts for targeted Qwen3 data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THEMES = [
    ("bike-chain-rain", "repairing a bicycle chain in the rain behind an apartment building"),
    ("diner-neon-close", "closing a diner after midnight under a flickering neon sign"),
    ("bus-depot-audition", "waiting at a bus depot after missing an important audition"),
    ("childhood-room-boxes", "packing boxes in a childhood bedroom before dawn"),
    ("studio-cable-showcase", "fixing a broken studio cable minutes before a small showcase"),
    ("lost-wallet-train", "returning a lost wallet on a crowded train"),
    ("grandmother-snow-steps", "clearing snow from a grandmother's steps before sunrise"),
    ("night-class-whistle", "finishing night class while a factory whistle sounds outside"),
    ("community-wall-paint", "painting over graffiti on a community center wall"),
    ("blackout-laundromat", "walking a dog past a shuttered laundromat during a blackout"),
    ("coffee-tin-microphone", "saving work tips in a coffee tin for a first microphone"),
    ("courthouse-sibling", "meeting a younger sibling outside a courthouse after a long day"),
]

TECHNICAL_TEMPLATE = (
    "Write exactly 12 lines of original rap lyrics about {theme}. Keep one coherent scene and clear literal "
    "meaning on every line. Use natural internal and multisyllabic rhymes in at least four lines without forced "
    "syntax. Include at least three concrete sensory details or physical actions. Avoid abstract filler, generic "
    "motivation, dialogue, and bracket labels. End with a complete declarative line that resolves the theme. "
    "No intro or commentary."
)

CLEAN_TEMPLATE = (
    "Write exactly 12 lines of clean, radio-safe original rap lyrics about {theme}. Keep one specific scene, "
    "natural rap cadence, and clear rhyme connections. Include at least three concrete physical details or "
    "actions. Avoid profanity, slurs, abstract motivational filler, forced phrases, dialogue, and bracket labels. "
    "End with a complete declarative payoff that resolves the theme. No intro or commentary."
)


def build_prompts(samples_per_model: int = 8) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    for theme_id, theme in THEMES:
        for family, template, style in (
            (
                "technical",
                TECHNICAL_TEMPLATE,
                "clear internal and multisyllabic rhyme grounded in a concrete scene",
            ),
            (
                "clean",
                CLEAN_TEMPLATE,
                "radio-safe concrete scene writing with natural rap cadence",
            ),
        ):
            prompts.append(
                {
                    "prompt": template.format(theme=theme),
                    "theme": theme,
                    "theme_id": f"stage2-{theme_id}",
                    "style": style,
                    "target_line_count": 12,
                    "prompt_family": family,
                    "instruction_family": f"targeted_{family}_v1",
                    "evaluation_split": "training_candidate",
                    "samples_per_model": samples_per_model,
                }
            )
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/prompts/qwen3_4b_12line_targeted_stage2_prompts.json"),
    )
    parser.add_argument("--samples-per-model", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_model <= 0:
        raise ValueError("--samples-per-model must be positive")
    prompts = build_prompts(args.samples_per_model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "prompt_count": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
