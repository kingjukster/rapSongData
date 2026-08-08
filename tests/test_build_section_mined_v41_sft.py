from pathlib import Path

from scripts.build_section_mined_v41_sft import choose_unused_strong_rows, theme_split_map


def judged(candidate: str, family: str, score: float = 3.6) -> dict:
    return {
        "candidate_id": candidate,
        "prompt_family": family,
        "lyrics": "\n".join(f"line {index}" for index in range(12)),
        "calibrated_review_score": score,
        "structural_metrics": {"structural_pass": True},
        "judge": {
            "overall_quality": 4,
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


def test_choose_unused_strong_rows_filters_used_and_weak_families() -> None:
    rows = [judged("m", "melodic"), judged("s", "story"), judged("t", "technical")]
    selected = choose_unused_strong_rows(rows, {"s"}, 3.5)
    assert [row["candidate_id"] for row in selected] == ["m"]


def test_theme_split_map_preserves_existing_assignments(tmp_path: Path) -> None:
    import json

    rows = {
        "train": [{"metadata": {"theme": "train theme"}}],
        "validation": [{"metadata": {"theme": "validation theme"}}],
        "test": [{"metadata": {"theme": "test theme"}}],
    }
    for split, values in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
        )
    assert theme_split_map(tmp_path) == {
        "train theme": "train",
        "validation theme": "validation",
        "test theme": "test",
    }
