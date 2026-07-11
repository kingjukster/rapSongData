#!/usr/bin/env python3
"""Validate and run the locked 1/2/3-epoch Qwen3 quality-training matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    Path("configs/training/local_cuda_qwen3_4b_12line_human_v1_e1.json"),
    Path("configs/training/local_cuda_qwen3_4b_12line_human_v1_e2.json"),
    Path("configs/training/local_cuda_qwen3_4b_12line_human_v1_e3.json"),
]
DEV_PROMPTS = Path("configs/prompts/quality_goal_dev_prompts.json")
CONFIRMATION_PROMPTS = Path("configs/prompts/quality_goal_confirmation_prompts.json")
AUDIT_SCRIPT = Path("scripts/audit_training_jsonl.py")
HELDOUT_AUDIT_SCRIPT = Path("scripts/audit_heldout_prompts.py")
TRAIN_SCRIPT = Path("model/train_local_cuda.py")
PINNED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", dest="configs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/qwen3_4b_12line_human_v1_matrix"),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate the matrix and write a plan without requiring data or launching training.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_matrix(configs: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if len(configs) != 3:
        raise ValueError("The locked comparison requires exactly three configs.")

    expected_epochs = [1.0, 2.0, 3.0]
    reference_dataset: dict[str, Any] | None = None
    reference_seed: tuple[int, int, int] | None = None
    reference_signature: dict[str, Any] | None = None
    output_dirs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, (path, config) in enumerate(configs):
        training = config.get("training") or {}
        dataset = config.get("dataset") or {}
        if config.get("base_model") != "Qwen/Qwen3-4B":
            raise ValueError(f"{path}: base_model must be Qwen/Qwen3-4B")
        if config.get("model_revision") != PINNED_MODEL_REVISION:
            raise ValueError(f"{path}: model_revision must be pinned to {PINNED_MODEL_REVISION}")
        if training.get("max_steps") is not None:
            raise ValueError(f"{path}: max_steps must be null for a true epoch comparison")
        if training.get("max_wall_time_minutes") is not None:
            raise ValueError(f"{path}: max_wall_time_minutes must be null for a complete run")
        if float(training.get("num_train_epochs", 0)) != expected_epochs[index]:
            raise ValueError(f"{path}: expected num_train_epochs={expected_epochs[index]}")
        if float(training.get("learning_rate", 0)) != 5e-5:
            raise ValueError(f"{path}: learning_rate must be 5e-5")
        if not training.get("assistant_only_loss"):
            raise ValueError(f"{path}: assistant_only_loss must be enabled")
        if not training.get("fail_on_truncation"):
            raise ValueError(f"{path}: fail_on_truncation must be enabled")

        comparable_dataset = {
            key: dataset.get(key)
            for key in ("train_path", "validation_path", "test_path", "manifest_path", "text_field", "format")
        }
        seed = (
            int(training.get("seed", -1)),
            int(training.get("data_seed", -1)),
            int(training.get("subset_shuffle_seed", -1)),
        )
        if reference_dataset is None:
            reference_dataset = comparable_dataset
            reference_seed = seed
        elif comparable_dataset != reference_dataset or seed != reference_seed:
            raise ValueError(f"{path}: dataset and all comparison seeds must match E1")
        signature = json.loads(json.dumps(config))
        output_dir = str(signature.pop("output_dir", ""))
        signature["training"].pop("num_train_epochs", None)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{path}: all settings except epochs and output_dir must match E1")
        if not output_dir or output_dir in output_dirs:
            raise ValueError(f"{path}: each comparison run requires a unique non-empty output_dir")
        output_dirs.add(output_dir)
        rows.append(
            {
                "config_path": str(path),
                "epochs": expected_epochs[index],
                "output_dir": output_dir,
            }
        )

    if reference_seed != (20260710, 20260710, 20260710):
        raise ValueError("The locked matrix seed must be 20260710 for model, data, and subsets.")
    return {
        "base_model": "Qwen/Qwen3-4B",
        "model_revision": PINNED_MODEL_REVISION,
        "dataset": reference_dataset,
        "seed": reference_seed[0] if reference_seed else None,
        "learning_rate": 5e-5,
        "runs": rows,
    }


def run_logged(command: list[str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "w", encoding="utf-8", buffering=1
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        "command": command,
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": round((ended - started).total_seconds(), 2),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def training_status(config: dict[str, Any]) -> tuple[str, dict[str, Any] | None, Path]:
    summary_path = Path(str(config["output_dir"])) / "training_summary.json"
    if not summary_path.exists():
        return "missing_training_summary", None, summary_path
    summary = read_json(summary_path)
    result = summary.get("result") if isinstance(summary.get("result"), dict) else {}
    return str(result.get("status") or "missing_result_status"), summary, summary_path


def main() -> int:
    args = parse_args()
    config_paths = args.configs or DEFAULT_CONFIGS
    configs = [(path, read_json(path)) for path in config_paths]
    matrix = validate_matrix(configs)
    plan = {
        "status": "matrix_validated" if args.prepare_only else "awaiting_data_gate",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **matrix,
    }
    write_json(args.run_dir / "run_plan.json", plan)
    print(json.dumps(plan, indent=2))
    if args.prepare_only:
        return 0

    dataset = configs[0][1]["dataset"]
    required_paths = [
        Path(dataset[key])
        for key in ("train_path", "validation_path", "test_path", "manifest_path")
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        final = {
            "status": "human_review_data_gate_not_met",
            "missing_paths": missing_paths,
            "message": "Build at least 100 eligible, provenance-verified human reviews before training.",
        }
        write_json(args.run_dir / "final_result.json", final)
        print(json.dumps(final, indent=2), file=sys.stderr)
        return 2

    manifest = read_json(required_paths[3])
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    if manifest.get("status") != "training_ready" or int(counts.get("verified_human_unique") or 0) < 100:
        final = {
            "status": "human_review_manifest_gate_failed",
            "manifest_status": manifest.get("status"),
            "verified_human_unique": counts.get("verified_human_unique"),
        }
        write_json(args.run_dir / "final_result.json", final)
        return 2
    non_clean_outputs = [
        str(Path(str(config["output_dir"])))
        for _, config in configs
        if Path(str(config["output_dir"])).exists()
        and (
            not Path(str(config["output_dir"])).is_dir()
            or any(Path(str(config["output_dir"])).iterdir())
        )
    ]
    if non_clean_outputs:
        final = {
            "status": "training_output_directory_not_clean",
            "non_clean_output_dirs": non_clean_outputs,
            "message": "Move or archive prior artifacts before starting a controlled comparison.",
        }
        write_json(args.run_dir / "final_result.json", final)
        return 2

    audit_results: list[dict[str, Any]] = []
    for name, left, right, include_manifest in [
        ("train_validation", required_paths[0], required_paths[1], True),
        ("train_test", required_paths[0], required_paths[2], False),
        ("validation_test", required_paths[1], required_paths[2], False),
    ]:
        command = [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--train",
            str(left),
            "--validation",
            str(right),
            "--out",
            str(args.run_dir / f"training_data_audit_{name}.json"),
            "--target-line-count",
            "12",
            "--require-explicit-target",
            "--require-provenance",
            "--require-human-review",
        ]
        if include_manifest:
            command.extend(["--manifest", str(required_paths[3])])
        result = run_logged(command, args.run_dir / f"data_audit_{name}")
        audit_results.append({"name": name, **result})
        if result["returncode"] != 0:
            write_json(
                args.run_dir / "final_result.json",
                {"status": "training_data_audit_failed", "audits": audit_results},
            )
            return int(result["returncode"] or 1)

    heldout_result = run_logged(
        [
            sys.executable,
            str(HELDOUT_AUDIT_SCRIPT),
            "--prompt-file",
            str(DEV_PROMPTS),
            "--prompt-file",
            str(CONFIRMATION_PROMPTS),
            "--training-jsonl",
            str(required_paths[0]),
            "--training-jsonl",
            str(required_paths[1]),
            "--training-jsonl",
            str(required_paths[2]),
            "--out",
            str(args.run_dir / "heldout_overlap_audit.json"),
        ],
        args.run_dir / "heldout_audit",
    )
    if heldout_result["returncode"] != 0:
        write_json(args.run_dir / "final_result.json", {"status": "heldout_overlap_audit_failed", **heldout_result})
        return int(heldout_result["returncode"] or 1)

    attempts: list[dict[str, Any]] = []
    for config_path, config in configs:
        run_name = f"e{int(config['training']['num_train_epochs'])}"
        result = run_logged(
            [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(config_path)],
            args.run_dir / run_name,
        )
        status, summary, summary_path = training_status(config)
        attempt = {
            "run": run_name,
            "config_path": str(config_path),
            **result,
            "training_status": status,
            "training_summary": str(summary_path),
            "result": summary.get("result") if summary else None,
        }
        attempts.append(attempt)
        write_json(args.run_dir / "attempts.json", {"attempts": attempts})
        if result["returncode"] != 0 or status != "complete_full_budget":
            final = {"status": "training_matrix_incomplete", "attempts": attempts}
            write_json(args.run_dir / "final_result.json", final)
            return int(result["returncode"] or 2)

    write_json(args.run_dir / "final_result.json", {"status": "complete_full_budget", "attempts": attempts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
