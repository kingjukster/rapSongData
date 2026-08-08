from __future__ import annotations

import json
from pathlib import Path

from scripts.build_qwen3_targeted_stage2_prompts import THEMES, build_prompts


ROOT = Path(__file__).resolve().parents[1]


def test_targeted_prompts_are_balanced_and_heldout_safe() -> None:
    prompts = build_prompts()
    evaluation = json.loads(
        (ROOT / "configs/prompts/qwen3_4b_12line_expanded_eval_prompts.json").read_text(encoding="utf-8")
    )
    evaluation_themes = {str(row["theme"]).casefold() for row in evaluation}

    assert len(THEMES) == 12
    assert len(prompts) == 24
    assert len({str(row["theme"]) for row in prompts}) == 12
    assert all(str(row["theme"]).casefold() not in evaluation_themes for row in prompts)
    assert sum(row["prompt_family"] == "technical" for row in prompts) == 12
    assert sum(row["prompt_family"] == "clean" for row in prompts) == 12
    assert all("exactly 12 lines" in str(row["prompt"]).lower() for row in prompts)
    assert all(row["evaluation_split"] == "training_candidate" for row in prompts)


def test_targeted_prompts_encode_family_specific_quality_requirements() -> None:
    prompts = build_prompts()
    technical = [str(row["prompt"]).lower() for row in prompts if row["prompt_family"] == "technical"]
    clean = [str(row["prompt"]).lower() for row in prompts if row["prompt_family"] == "clean"]

    assert all("multisyllabic" in prompt and "concrete" in prompt and "declarative" in prompt for prompt in technical)
    assert all("radio-safe" in prompt and "concrete" in prompt and "payoff" in prompt for prompt in clean)
