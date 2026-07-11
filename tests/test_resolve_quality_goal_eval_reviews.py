from __future__ import annotations

import pytest

from scripts.resolve_quality_goal_eval_reviews import resolve_reviews


def fixtures() -> tuple[dict, dict]:
    packet = {
        "comparisons": [
            {
                "comparison_id": "comparison-1",
                "review": {"winner_alias": "B", "ranking": ["B", "A"], "tie": False, "notes": ""},
            }
        ]
    }
    key = {
        "assignments": [
            {
                "comparison_id": "comparison-1",
                "source_row_id": "row-1",
                "candidates": [
                    {"candidate_alias": "A", "generation_label": "base"},
                    {"candidate_alias": "B", "generation_label": "e2"},
                ],
            }
        ]
    }
    return packet, key


def test_resolves_winner_and_ranking_without_lyrics() -> None:
    packet, key = fixtures()

    rows = resolve_reviews(packet, key)

    assert rows == [
        {
            "comparison_id": "comparison-1",
            "row_id": "row-1",
            "winner": "e2",
            "ranking": ["e2", "base"],
            "notes": "",
        }
    ]


def test_rejects_incomplete_review() -> None:
    packet, key = fixtures()
    packet["comparisons"][0]["review"]["winner_alias"] = None
    packet["comparisons"][0]["review"]["ranking"] = []

    with pytest.raises(ValueError, match="incomplete"):
        resolve_reviews(packet, key)
