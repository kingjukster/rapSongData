from scripts.build_qwen3_v5_confirmation_prompts import build_prompts


def test_v5_confirmation_is_balanced_and_unique() -> None:
    rows = build_prompts()
    assert len(rows) == 24
    assert len({row["prompt"] for row in rows}) == 24
    counts = {}
    for row in rows:
        counts[row["prompt_family"]] = counts.get(row["prompt_family"], 0) + 1
    assert set(counts.values()) == {6}
