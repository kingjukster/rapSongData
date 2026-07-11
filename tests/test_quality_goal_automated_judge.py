from __future__ import annotations

from scripts.judge_quality_goal_eval_packet_openai import judge_packet


def test_mock_automated_consensus_selects_adapter_without_human_review() -> None:
    packet = {
        "comparisons": [
            {
                "comparison_id": "comparison-1",
                "prompt_metadata": {"prompt": "Write exactly 12 lines about rain."},
                "candidates": [
                    {"candidate_alias": "A", "lyrics": "short repeated line"},
                    {
                        "candidate_alias": "B",
                        "lyrics": "rain on the rail and sparks in the alley with a final earned declaration",
                    },
                ],
            }
        ]
    }
    private_key = {
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

    judged, preferences, summary = judge_packet(
        packet,
        private_key,
        votes_per_comparison=3,
        model="unused",
        temperature=0.0,
        max_retries=0,
        sleep_seconds=0.0,
        mock=True,
    )

    assert judged[0]["consensus"]["vote_count"] == 3
    assert preferences[0]["winner"] == "e2"
    assert summary["selected_adapter"] == "e2"
    assert summary["human_review_used"] is False
