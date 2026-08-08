from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "training"
PINNED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET_DIR = "data/training/qwen3_4b_12line_auto_calibrated_v3"
CONFIG_PATHS = {
    "full": CONFIG_DIR / "local_cuda_qwen3_4b_12line_auto_v3_e1.json",
    "smoke": CONFIG_DIR / "local_cuda_qwen3_4b_12line_auto_v3_smoke.json",
}


def load_config(kind: str) -> dict:
    return json.loads(CONFIG_PATHS[kind].read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["full", "smoke"])
def test_v3_training_config_locks_conservative_qlora_settings(kind: str) -> None:
    config = load_config(kind)
    training = config["training"]

    assert config["base_model"] == "Qwen/Qwen3-4B"
    assert config["model_revision"] == PINNED_MODEL_REVISION
    assert config["add_structural_special_tokens"] is False
    assert config["dataset"] == {
        "train_path": f"{DATASET_DIR}/train.jsonl",
        "validation_path": f"{DATASET_DIR}/validation.jsonl",
        "test_path": f"{DATASET_DIR}/test.jsonl",
        "manifest_path": f"{DATASET_DIR}/manifest.json",
        "preference_path": f"{DATASET_DIR}/preference_pairs.jsonl",
        "text_field": "training_text",
        "format": "json",
    }
    assert training["num_train_epochs"] == 1.0
    assert training["learning_rate"] == pytest.approx(2e-5)
    assert training["sequence_length"] == 512
    assert training["per_device_train_batch_size"] == 2
    assert training["per_device_eval_batch_size"] == 2
    assert training["gradient_accumulation_steps"] == 2
    assert training["lora_rank"] == 8
    assert training["lora_alpha"] == 16
    assert training["lora_dropout"] == pytest.approx(0.05)
    assert training["assistant_only_loss"] is True
    assert training["fail_on_truncation"] is True
    assert training["seed"] == training["data_seed"] == training["subset_shuffle_seed"]
    assert training["full_determinism"] is True
    assert training["load_in_4bit"] is True
    assert training["bf16"] is True
    assert training["attn_implementation"] == "sdpa"
    assert training["optim"] == "paged_adamw_8bit"
    assert "qwen3_4b_12line_auto_v3" in training["tokenized_cache_dir"]


def test_v3_full_and_smoke_configs_have_safe_budgets_and_distinct_outputs() -> None:
    full = load_config("full")
    smoke = load_config("smoke")

    assert full["training"]["max_steps"] == 0
    assert full["training"]["max_wall_time_minutes"] is None
    assert full["training"]["eval_strategy"] == "epoch"
    assert full["training"]["save_strategy"] == "epoch"

    assert smoke["training"]["max_steps"] == 10
    assert smoke["training"]["max_wall_time_minutes"] == 10
    assert smoke["training"]["eval_strategy"] == "no"
    assert smoke["training"]["save_strategy"] == "no"

    assert full["output_dir"] != smoke["output_dir"]
    assert full["training"]["tokenized_cache_dir"] == smoke["training"]["tokenized_cache_dir"]
