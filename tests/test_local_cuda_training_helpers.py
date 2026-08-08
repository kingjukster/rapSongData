from __future__ import annotations

import json
from pathlib import Path

import pytest

from rap_song_data.training import local_cuda

from rap_song_data.training.local_cuda import (
    assistant_only_labels,
    classify_training_completion,
    dataset_selection_cache_metadata,
    dataset_fingerprints,
    emit_policy_warnings,
    filter_training_args_for_signature,
    resolve_max_steps,
    write_run_summary,
)


def test_training_timestamp_runtime_is_available() -> None:
    assert local_cuda.dt.datetime.now().tzinfo is None


def test_policy_warning_reports_actual_gradient_accumulation() -> None:
    warnings = emit_policy_warnings(
        {"sequence_length": 1024, "gradient_accumulation_steps": 2},
        {"format": "json", "train_path": "train.jsonl"},
    )

    assert warnings == [
        "gradient_accumulation_steps=2 is being used for faster iteration; "
        "for strict comparability, prefer a higher accumulation setting (8)."
    ]


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


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, "group_by_length"), (False, "random")],
)
def test_training_args_map_group_by_length_for_transformers_5_12(
    enabled: bool,
    expected: str,
) -> None:
    filtered = filter_training_args_for_signature(
        {
            "group_by_length": enabled,
            "eval_strategy": "epoch",
            "unsupported": "drop-me",
        },
        {
            "train_sampling_strategy": object(),
            "eval_strategy": object(),
        },
    )

    assert filtered == {
        "train_sampling_strategy": expected,
        "eval_strategy": "epoch",
    }


def test_training_args_retain_legacy_group_by_length_and_eval_name() -> None:
    filtered = filter_training_args_for_signature(
        {
            "group_by_length": True,
            "eval_strategy": "epoch",
        },
        {
            "group_by_length": object(),
            "evaluation_strategy": object(),
        },
    )

    assert filtered == {
        "group_by_length": True,
        "evaluation_strategy": "epoch",
    }


def test_run_summary_reports_epoch_mode_max_steps_as_null(tmp_path: Path) -> None:
    summary_path, _ = write_run_summary(
        output_dir=tmp_path,
        command=["python", "train.py"],
        run_started_at="2026-07-12T00:00:00-05:00",
        run_ended_at="2026-07-12T00:01:00-05:00",
        run_wall_seconds=60.0,
        base_model="Qwen/Qwen3-4B",
        dataset_cfg={
            "train_path": "train.jsonl",
            "validation_path": "validation.jsonl",
            "text_field": "training_text",
        },
        training_cfg={
            "max_steps": 0,
            "sequence_length": 512,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
        },
        timing_records=[],
        result={"status": "complete_full_budget"},
        policy_warnings=[],
        runtime={},
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["training"]["max_steps"] is None


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
