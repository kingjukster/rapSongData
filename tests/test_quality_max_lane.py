from __future__ import annotations

import pytest

from rap_song_data.quality_max.prompts import build_messages
from rap_song_data.quality_max.scoring import clean_lyrics, rank_candidates, score_candidate
from rap_song_data.quality_max.runner import resolve_settings


def test_generation_prompt_uses_soft_bar_range() -> None:
    messages = build_messages(
        mode="generate",
        theme="recovery",
        style="cinematic",
        keywords="rain, train",
        min_bars=12,
        max_bars=36,
        target_bars=24,
    )
    prompt = messages[-1]["content"]
    assert "roughly 12-36 bars" in prompt
    assert "about 24 bars" in prompt
    assert "not an exact line-count requirement" in prompt
    assert "Write exactly" not in prompt


@pytest.mark.parametrize("mode", ["style_transfer", "mutate", "improve"])
def test_transformations_require_source_text(mode: str) -> None:
    with pytest.raises(ValueError, match="requires source_text"):
        build_messages(
            mode=mode,
            theme="test",
            style="test",
            keywords="",
            min_bars=12,
            max_bars=36,
        )


def test_style_transfer_forbids_recognizable_copying() -> None:
    messages = build_messages(
        mode="style_transfer",
        theme="same meaning",
        style="high imagery",
        keywords="",
        source_text="A source line",
        artist_reference="Example Artist",
        min_bars=12,
        max_bars=36,
    )
    text = "\n".join(message["content"] for message in messages)
    assert "high-level traits" in text
    assert "recognizable lyrics" in text


def test_cleaning_never_truncates_to_bar_range() -> None:
    raw = "\n".join(f"line {index} with a complete image" for index in range(40))
    assert len(clean_lyrics(raw).splitlines()) == 40


def test_length_is_a_soft_score_not_a_rejection() -> None:
    short = "\n".join(f"short original bar {index}" for index in range(8))
    result = score_candidate(short, min_bars=12, max_bars=36)
    assert result["outside_soft_range"] is True
    assert 0.0 < result["components"]["bar_range"] < 1.0


def test_ranker_preserves_every_candidate() -> None:
    candidates = [
        {"candidate_index": 1, "lyrics": "\n".join(f"same bar {index}" for index in range(12))},
        {"candidate_index": 2, "lyrics": "\n".join(f"distinct image {index} moves tonight" for index in range(16))},
    ]
    ranked = rank_candidates(candidates, min_bars=12, max_bars=36, keywords="image")
    assert len(ranked) == 2
    assert {row["candidate_index"] for row in ranked} == {1, 2}
    assert [row["rank"] for row in ranked] == [1, 2]


def test_smoke_overrides_compute_budget_without_changing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "task_defaults": {
            "mode": "generate",
            "theme": "test",
            "style": "test",
            "keywords": "",
            "artist_reference": "",
            "min_bars": 12,
            "max_bars": 36,
            "target_bars": 24,
        },
        "generation": {"candidates": 12, "batch_size": 4, "revision_rounds": 1},
        "smoke": {"candidates": 2, "batch_size": 2, "revision_rounds": 0},
        "output_root": "runs/test",
        "adapter_enabled": True,
        "local_files_only": False,
    }
    args = type(
        "Args",
        (),
        {
            "mode": None,
            "theme": None,
            "style": None,
            "keywords": None,
            "artist_reference": None,
            "min_bars": None,
            "max_bars": None,
            "target_bars": None,
            "candidates": None,
            "batch_size": None,
            "revision_rounds": None,
            "source_file": None,
            "run_dir": None,
            "smoke": True,
            "base_only": False,
            "local_files_only": True,
        },
    )()
    settings = resolve_settings(args, config)
    assert settings["generation"]["candidates"] == 2
    assert settings["generation"]["revision_rounds"] == 0
    assert settings["task"]["target_bars"] == 24
    assert settings["local_files_only"] is True
