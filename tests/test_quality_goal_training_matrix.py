from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_quality_goal_training_matrix.py"
SPEC = importlib.util.spec_from_file_location("quality_goal_training_matrix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_configs() -> list[tuple[Path, dict]]:
    return [(path, json.loads((ROOT / path).read_text(encoding="utf-8"))) for path in MODULE.DEFAULT_CONFIGS]


def test_locked_matrix_configs_are_comparable() -> None:
    result = MODULE.validate_matrix(load_configs())

    assert [row["epochs"] for row in result["runs"]] == [1.0, 2.0, 3.0]
    assert result["learning_rate"] == 5e-5
    assert result["seed"] == 20260710


def test_matrix_rejects_max_steps_override() -> None:
    configs = load_configs()
    changed = copy.deepcopy(configs)
    changed[1][1]["training"]["max_steps"] = 20

    with pytest.raises(ValueError, match="max_steps must be null"):
        MODULE.validate_matrix(changed)


def test_matrix_rejects_non_epoch_hyperparameter_drift() -> None:
    configs = load_configs()
    changed = copy.deepcopy(configs)
    changed[1][1]["training"]["lora_rank"] = 999

    with pytest.raises(ValueError, match="all settings except epochs"):
        MODULE.validate_matrix(changed)


def test_training_status_requires_explicit_full_budget_summary(tmp_path: Path) -> None:
    config = {"output_dir": str(tmp_path)}
    assert MODULE.training_status(config)[0] == "missing_training_summary"
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"result": {"status": "partial_time_budget_reached"}}),
        encoding="utf-8",
    )

    assert MODULE.training_status(config)[0] == "partial_time_budget_reached"
