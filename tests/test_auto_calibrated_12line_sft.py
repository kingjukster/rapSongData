from __future__ import annotations

import pytest

from scripts.build_auto_calibrated_12line_sft import (
    BALANCED_FAMILIES,
    DEFAULT_OUTPUT,
    DEFAULT_V3_OUTPUT,
    QUALITY_TIER_1,
    QUALITY_TIER_2,
    QUALITY_TIER_3,
    consensus_eligible,
    distribution_summary,
    ensure_unique_lyrics,
    parse_args,
    preference_pairs,
    quality_tier,
    select_balanced_examples,
    split_balanced_theme_groups,
    split_theme_groups,
)


def row(
    index: int,
    *,
    theme: int = 0,
    family: str = "story",
    calibrated: float = 3.8,
    overall: int = 4,
) -> dict:
    lyrics = "\n".join(f"concrete scene {index} line {line} moves in rain" for line in range(12))
    return {
        "candidate_id": f"candidate-{index}",
        "prompt_key": f"prompt-{theme}",
        "prompt": f"Write exactly 12 lines about theme {theme}.",
        "theme": f"theme-{theme}",
        "prompt_family": family,
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


def test_v2_defaults_are_preserved_and_v3_is_explicit() -> None:
    v2 = parse_args([])
    assert v2.dataset_version == "v2"
    assert v2.output_dir == DEFAULT_OUTPUT

    v3 = parse_args(["--dataset-version", "v3"])
    assert v3.output_dir == DEFAULT_V3_OUTPUT
    assert {family: getattr(v3, f"{family}_quota") for family in BALANCED_FAMILIES} == {
        family: 33 for family in BALANCED_FAMILIES
    }


def test_quality_tiers_follow_imagery_and_payoff_policy() -> None:
    tier_one = row(10)
    tier_one["judge"]["dimension_scores"].update({"imagery": 4, "ending_payoff": 4})
    assert quality_tier(tier_one) == QUALITY_TIER_1

    tier_two = row(11)
    tier_two["judge"]["dimension_scores"].update({"imagery": 4, "ending_payoff": 3})
    assert quality_tier(tier_two) == QUALITY_TIER_2

    assert quality_tier(row(12)) == QUALITY_TIER_3


def test_balanced_selection_uses_exact_quotas_and_stable_tier_order() -> None:
    rows = []
    for family_index, family in enumerate(BALANCED_FAMILIES):
        tier_three = row(family_index * 10 + 1, family=family, calibrated=4.5)
        tier_two = row(family_index * 10 + 2, family=family, calibrated=3.5)
        tier_two["judge"]["dimension_scores"]["imagery"] = 4
        tier_one = row(family_index * 10 + 3, family=family, calibrated=3.5)
        tier_one["judge"]["dimension_scores"].update({"imagery": 4, "ending_payoff": 4})
        rows.extend((tier_three, tier_two, tier_one))

    quotas = {family: 2 for family in BALANCED_FAMILIES}
    first = select_balanced_examples(rows, quotas)
    second = select_balanced_examples(list(reversed(rows)), quotas)
    assert [value["candidate_id"] for value in first] == [value["candidate_id"] for value in second]
    for family in BALANCED_FAMILIES:
        family_rows = [value for value in first if value["prompt_family"] == family]
        assert len(family_rows) == 2
        assert [value["selection_metadata"]["quality_tier"] for value in family_rows] == [
            QUALITY_TIER_1,
            QUALITY_TIER_2,
        ]


def test_balanced_theme_splits_preserve_groups_and_every_family() -> None:
    rows = [
        row(theme_index * 10 + family_index, theme=theme_index, family=family)
        for theme_index in range(6)
        for family_index, family in enumerate(BALANCED_FAMILIES)
    ]
    splits = split_balanced_theme_groups(rows, 20260710)
    theme_splits: dict[str, set[str]] = {}
    for split, values in splits.items():
        assert {value["prompt_family"] for value in values} == set(BALANCED_FAMILIES)
        for value in values:
            theme_splits.setdefault(value["theme"], set()).add(split)
    assert all(len(assignments) == 1 for assignments in theme_splits.values())
    assert len(splits["validation"]) == len(BALANCED_FAMILIES)
    assert len(splits["test"]) == len(BALANCED_FAMILIES)


def test_duplicate_selected_lyrics_fail_the_build() -> None:
    first = row(100)
    duplicate = row(101)
    duplicate["lyrics"] = first["lyrics"]
    with pytest.raises(ValueError, match="Duplicate normalized lyrics selected"):
        ensure_unique_lyrics([first, duplicate])


def test_distribution_summary_covers_selection_audit_dimensions() -> None:
    rows = [row(200, theme=1, family="story"), row(201, theme=2, family="clean")]
    summary = distribution_summary(rows)
    assert summary["family"] == {"clean": 1, "story": 1}
    assert summary["theme"] == {"theme-1": 1, "theme-2": 1}
    assert summary["quality_tier"] == {QUALITY_TIER_3: 2}
    assert summary["dimensions"]["imagery"] == {"3": 2}
    assert summary["issue_tag"] == {"none": 2}
