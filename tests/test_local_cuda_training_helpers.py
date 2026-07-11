from __future__ import annotations

from pathlib import Path

import pytest

from rap_song_data.training.local_cuda import (
    assistant_only_labels,
    classify_training_completion,
    dataset_selection_cache_metadata,
    dataset_fingerprints,
    resolve_max_steps,
)


def test_assistant_only_labels_mask_prompt_and_keep_target() -> None:
    marker = [20, 21]
    input_ids = [1, 2, 3, 20, 21, 30, 31, 99]

    labels = assistant_only_labels(input_ids, marker)

    assert labels == [-100, -100, -100, -100, -100, 30, 31, 99]


def test_assistant_only_labels_fail_when_marker_was_truncated() -> None:
    with pytest.raises(ValueError, match="assistant marker"):
        assistant_only_labels([1, 2, 3], [20, 21])


def test_assistant_only_labels_fail_when_no_supervised_tokens_remain() -> None:
    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        assistant_only_labels([1, 2, 20, 21], [20, 21])


def test_dataset_fingerprint_changes_when_file_changes(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    manifest = tmp_path / "manifest.json"
    train.write_text("first\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    config = {
        "train_path": str(train),
        "validation_path": str(validation),
        "manifest_path": str(manifest),
    }

    before = dataset_fingerprints(config)
    train.write_text("second\n", encoding="utf-8")
    after = dataset_fingerprints(config)

    assert before["train"]["sha256"] != after["train"]["sha256"]
    assert before["validation"] == after["validation"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), (0, None), (-1, None), (45, 45)],
)
def test_resolve_max_steps_supports_true_epoch_mode(configured: int | None, expected: int | None) -> None:
    assert resolve_max_steps({"max_steps": configured}) == expected


def test_dataset_selection_settings_participate_in_cache_metadata() -> None:
    baseline = dataset_selection_cache_metadata(
        {
            "max_train_records": 100,
            "max_validation_records": 20,
            "subset_shuffle_seed": 7,
        }
    )
    changed_limit = dataset_selection_cache_metadata(
        {
            "max_train_records": 101,
            "max_validation_records": 20,
            "subset_shuffle_seed": 7,
        }
    )
    changed_seed = dataset_selection_cache_metadata(
        {
            "max_train_records": 100,
            "max_validation_records": 20,
            "subset_shuffle_seed": 8,
        }
    )

    assert baseline != changed_limit
    assert baseline != changed_seed


@pytest.mark.parametrize(
    ("configured_max_steps", "requested_epochs", "global_step", "actual_epochs", "expected_reason"),
    [
        (10, 3.0, 10, 0.5, "max_steps_reached"),
        (None, 3.0, 30, 3.0, "epoch_target_reached"),
    ],
)
def test_full_budget_wins_over_simultaneous_time_budget_trigger(
    configured_max_steps: int | None,
    requested_epochs: float,
    global_step: int,
    actual_epochs: float,
    expected_reason: str,
) -> None:
    status, reason = classify_training_completion(
        configured_max_steps=configured_max_steps,
        requested_epochs=requested_epochs,
        global_step=global_step,
        actual_epochs=actual_epochs,
        time_budget_triggered=True,
    )

    assert status == "complete_full_budget"
    assert reason == expected_reason


def test_time_budget_is_partial_when_epoch_target_was_not_reached() -> None:
    status, reason = classify_training_completion(
        configured_max_steps=None,
        requested_epochs=3.0,
        global_step=20,
        actual_epochs=2.0,
        time_budget_triggered=True,
    )

    assert status == "partial_time_budget_reached"
    assert reason == "time_budget_reached"
