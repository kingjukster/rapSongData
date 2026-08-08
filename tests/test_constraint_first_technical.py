from __future__ import annotations

from scripts.run_constraint_first_technical import validate_plan


def candidate() -> dict:
    endings = ["light", "night", "rain", "train", "steel", "wheel", "flame", "name", "wire", "fire", "stone", "home"]
    lines = [f"quick silver figures trigger bigger patterns in the {word}" for word in endings]
    return {
        "lyrics": "\n".join(lines),
        "plan": {
            "end_rhyme_groups": [
                {"lines": [1, 2], "anchors": ["light", "night"]},
                {"lines": [3, 4], "anchors": ["rain", "train"]},
                {"lines": [5, 6], "anchors": ["steel", "wheel"]},
                {"lines": [7, 8], "anchors": ["flame", "name"]},
            ],
            "multisyllabic_chains": [
                {"terms": ["silver figures", "trigger bigger"], "lines": [1, 2]},
                {"terms": ["bigger patterns", "silver figures"], "lines": [3, 4]},
                {"terms": ["trigger bigger", "bigger patterns"], "lines": [5, 6]},
            ],
            "internal_rhyme_lines": [1, 2, 3, 4, 5, 6],
            "narrative_movements": ["setup", "pressure", "payoff"],
            "final_payoff": "home",
        },
    }


def test_validate_plan_accepts_present_contract() -> None:
    valid, failures, evidence = validate_plan(candidate())
    assert valid, failures
    assert len(evidence["covered_end_lines"]) == 8
    assert evidence["present_chain_count"] == 3


def test_validate_plan_rejects_missing_anchor() -> None:
    item = candidate()
    item["plan"]["end_rhyme_groups"][0]["anchors"][0] = "missing"
    valid, failures, _ = validate_plan(item)
    assert not valid
    assert "planned_end_anchors" in failures
