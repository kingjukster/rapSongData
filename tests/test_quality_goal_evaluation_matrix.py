from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_quality_goal_evaluation_matrix.py"
SPEC = importlib.util.spec_from_file_location("quality_goal_evaluation_matrix", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG_PATH = Path("configs/evaluation/quality_goal_generation_v1.json")


def config() -> dict:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_development_plan_has_exact_raw_generation_commands(tmp_path: Path) -> None:
    locked = MODULE.validate_config(config())
    plan = MODULE.build_plan(
        CONFIG_PATH,
        locked,
        stage="development",
        winner=None,
        run_dir=tmp_path,
    )

    assert plan["labels"] == ["base", "e1", "e2", "e3"]
    assert plan["prompt_count"] == 48
    assert plan["samples_per_prompt"] == 2
    assert plan["expected_rows_per_model"] == 96
    assert plan["winner_selection"] == "disabled"
    assert plan["selected_development_winner"] is None
    assert all(run["expected_rows"] == 96 for run in plan["runs"])

    for run in plan["runs"]:
        command = run["command"]
        assert option(command, "--base-model") == "Qwen/Qwen3-4B"
        assert option(command, "--model-revision") == MODULE.PINNED_MODEL_REVISION
        assert option(command, "--seed") == "20260710"
        assert option(command, "--num-candidates") == "96"
        assert option(command, "--batch-size") == "1"
        assert option(command, "--underlength-retries") == "0"
        assert "--strict-row-seeds" in command
        assert "--no-resume" in command
        assert "--no-block-slurs" in command
        assert "--no-enforce-target-line-count" in command
        assert "--run-manifest" in command

    decoding_flags = (
        "--max-input-tokens",
        "--max-new-tokens",
        "--temperature",
        "--top-p",
        "--top-k",
        "--repetition-penalty",
        "--no-repeat-ngram-size",
    )
    reference = {flag: option(plan["runs"][0]["command"], flag) for flag in decoding_flags}
    for run in plan["runs"][1:]:
        assert {flag: option(run["command"], flag) for flag in decoding_flags} == reference

    packet_command = plan["packet"]["command"]
    assert packet_command.count("--generation") == 4
    assert packet_command.count("--train-jsonl") == 3
    assert option(packet_command, "--seed") == "20260710"


def test_confirmation_requires_and_uses_explicit_development_winner(tmp_path: Path) -> None:
    locked = MODULE.validate_config(config())

    with pytest.raises(ValueError, match="explicit --winner"):
        MODULE.build_plan(CONFIG_PATH, locked, stage="confirmation", winner=None, run_dir=tmp_path)
    with pytest.raises(ValueError, match="only valid for the confirmation"):
        MODULE.build_plan(CONFIG_PATH, locked, stage="development", winner="e2", run_dir=tmp_path)

    plan = MODULE.build_plan(
        CONFIG_PATH,
        locked,
        stage="confirmation",
        winner="e2",
        run_dir=tmp_path,
    )
    assert plan["labels"] == ["base", "e2"]
    assert plan["prompt_count"] == 24
    assert plan["samples_per_prompt"] == 2
    assert plan["expected_rows_per_model"] == 48
    assert plan["selected_development_winner"] == "e2"
    assert all(run["expected_rows"] == 48 for run in plan["runs"])
    assert plan["packet"]["command"].count("--generation") == 2


def test_config_rejects_postprocessing_or_automatic_selection() -> None:
    changed = copy.deepcopy(config())
    changed["generation"]["enforce_target_line_count"] = True
    with pytest.raises(ValueError, match="enforce_target_line_count"):
        MODULE.validate_config(changed)

    changed = copy.deepcopy(config())
    changed["selection_policy"]["auto_select_winner"] = True
    with pytest.raises(ValueError, match="auto_select_winner"):
        MODULE.validate_config(changed)

    changed = copy.deepcopy(config())
    changed["promotion"]["maximum_exact_line_rate_drop"] = 0.2
    with pytest.raises(ValueError, match="maximum_exact_line_rate_drop"):
        MODULE.validate_config(changed)


def test_adapter_gate_requires_full_budget_and_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "REPO", tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    summary_path = adapter_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "base_model": MODULE.PINNED_BASE_MODEL,
                "model_revision": MODULE.PINNED_MODEL_REVISION,
                "result": {"status": "partial_time_budget_reached"},
            }
        ),
        encoding="utf-8",
    )
    runtime_config = {"models": {"e1": {"adapter_dir": "adapter"}}}
    with pytest.raises(ValueError, match="complete_full_budget"):
        MODULE.validate_adapter("e1", runtime_config)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["result"]["status"] = "complete_full_budget"
    payload["model_revision"] = "floating"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong model revision"):
        MODULE.validate_adapter("e1", runtime_config)

    payload["model_revision"] = MODULE.PINNED_MODEL_REVISION
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    hashes = MODULE.validate_adapter("e1", runtime_config)
    assert set(hashes) == {"training_summary", "adapter_config", "adapter_weights"}
    assert all(len(record["sha256"]) == 64 for record in hashes.values())


def test_prepare_only_does_not_check_adapters_or_launch_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "parse_args",
        lambda: argparse.Namespace(
            config=CONFIG_PATH,
            stage="development",
            winner=None,
            run_dir=tmp_path,
            prepare_only=True,
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "run_logged",
        lambda *_args, **_kwargs: pytest.fail("prepare-only launched a subprocess"),
    )

    assert MODULE.main() == 0
    plan = json.loads((tmp_path / "development" / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "prepared"
    assert plan["input_hashes"]["adapter_artifacts"] == {}
    assert len(plan["runs"]) == 4
