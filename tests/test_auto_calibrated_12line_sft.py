from __future__ import annotations

from scripts.build_auto_calibrated_12line_sft import (
    consensus_eligible,
    preference_pairs,
    split_theme_groups,
)


def row(index: int, *, theme: int = 0, calibrated: float = 3.8, overall: int = 4) -> dict:
    lyrics = "\n".join(f"concrete scene {index} line {line} moves in rain" for line in range(12))
    return {
        "candidate_id": f"candidate-{index}",
        "prompt_key": f"prompt-{theme}",
        "prompt": f"Write exactly 12 lines about theme {theme}.",
        "theme": f"theme-{theme}",
        "prompt_family": "story",
        "lyrics": lyrics,
        "calibrated_review_score": calibrated,
        "structural_metrics": {
            "structural_pass": True,
            "line_count": 12,
            "slur_count": 0,
            "prompt_leakage": False,
        },
        "judge": {
            "overall_quality": overall,
            "usable_as_is": "yes",
            "dimension_scores": {
                "theme_adherence": 4,
                "imagery": 3,
                "rhyme_cadence": 3,
                "originality": 3,
                "scene_coherence": 4,
                "ending_payoff": 3,
                "naturalness": 4,
            },
        },
    }


def test_consensus_gate_is_explicitly_automated_and_strict() -> None:
    assert consensus_eligible(row(1), 3.5)
    weak = row(2)
    weak["judge"]["dimension_scores"]["imagery"] = 2
    assert not consensus_eligible(weak, 3.5)
    assert not consensus_eligible(row(3, overall=3), 3.5)


def test_theme_groups_do_not_cross_splits() -> None:
    rows = [row(index, theme=index // 2) for index in range(20)]
    splits = split_theme_groups(rows, 20260710)
    theme_splits: dict[str, set[str]] = {}
    for split, values in splits.items():
        for value in values:
            theme_splits.setdefault(value["theme"], set()).add(split)
    assert all(len(assignments) == 1 for assignments in theme_splits.values())


def test_preference_pairs_require_score_margin_and_distinct_text() -> None:
    chosen = row(1, calibrated=4.0)
    rejected = row(2, calibrated=2.9)
    rejected["prompt_key"] = chosen["prompt_key"]
    rejected["prompt"] = chosen["prompt"]
    pairs = preference_pairs(
        [chosen],
        [chosen, rejected],
        {chosen["candidate_id"]: "train"},
        margin=0.75,
        max_rejects=3,
    )
    assert len(pairs) == 1
    assert pairs[0]["metadata"]["label_source"] == "automated_consensus_score_margin"
    assert pairs[0]["metadata"]["score_delta"] >= 0.75
