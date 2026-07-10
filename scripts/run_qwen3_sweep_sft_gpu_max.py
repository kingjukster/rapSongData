"""Build and run the Qwen3-4B sweep SFT training job with high GPU use.

The runner is local-only and Qwen3-4B-only. It builds train/validation JSONL
from packaged sweep files, writes a concrete training config for each profile,
and runs ``model/train_local_cuda.py``. In ``auto`` mode it starts with the
highest-use profile and falls back only when the failure looks like CUDA OOM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/training/local_cuda_qwen3_4b_sweep_sft_gpu_max.json")
TRAIN_SCRIPT = Path("model/train_local_cuda.py")
BUILD_SCRIPT = Path("scripts/build_qwen3_sft_training_files.py")

PROFILES = [
    {
        "name": "seq768_bs3_ga2",
        "sequence_length": 768,
        "per_device_train_batch_size": 3,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
    },
    {
        "name": "seq768_bs2_ga2",
        "sequence_length": 768,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
    },
    {
        "name": "seq512_bs3_ga2",
        "sequence_length": 512,
        "per_device_train_batch_size": 3,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
    },
    {
        "name": "seq512_bs2_ga2",
        "sequence_length": 512,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
    },
    {
        "name": "seq768_bs1_ga4_gc",
        "sequence_length": 768,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packaged-dir", type=Path, default=Path("data/packaged/qwen3_4b_sweep"))
    parser.add_argument("--training-dir", type=Path, default=Path("data/training/qwen3_4b_sweep_sft"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qwen3_sweep_sft_gpu_max"))
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument(
        "--profile",
        choices=["auto"] + [profile["name"] for profile in PROFILES],
        default="auto",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wall-minutes", type=float, default=None)
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix appended to the model artifact directory name, e.g. 1epoch.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--include-manual-repairs", action="store_true")
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_logged(command: list[str], *, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as stderr:
        proc = subprocess.run(command, cwd=REPO, text=True, stdout=stdout, stderr=stderr, check=False)
    ended = dt.datetime.now()
    return {
        "command": command,
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": round((ended - started).total_seconds(), 2),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def looks_like_oom(log_dir: Path) -> bool:
    haystack = ""
    for name in ["stdout.txt", "stderr.txt"]:
        path = log_dir / name
        if path.exists():
            haystack += path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(
        phrase in haystack
        for phrase in [
            "cuda out of memory",
            "outofmemoryerror",
            "out of memory",
            "cublas_status_alloc_failed",
        ]
    )


def ensure_training_files(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_build:
        manifest_path = args.training_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"--skip-build requested but manifest is missing: {manifest_path}")
        return read_json(manifest_path)

    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--packaged-dir",
        str(args.packaged_dir),
        "--output-dir",
        str(args.training_dir),
    ]
    if args.include_manual_repairs:
        command.append("--include-manual-repairs")
    result = run_logged(command, log_dir=args.run_dir / "build_training_files")
    if result["returncode"] != 0:
        raise SystemExit(
            "Could not build SFT train/validation files. "
            f"See {args.run_dir / 'build_training_files' / 'stderr.txt'}"
        )
    return read_json(args.training_dir / "manifest.json")


def profile_sequence(selected: str) -> list[dict[str, Any]]:
    if selected == "auto":
        return list(PROFILES)
    return [profile for profile in PROFILES if profile["name"] == selected]


def config_for_profile(
    *,
    base: dict[str, Any],
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    if config.get("base_model") != "Qwen/Qwen3-4B":
        raise ValueError("This runner is locked to Qwen/Qwen3-4B.")
    profile_name = profile["name"]
    suffix = f"-{args.artifact_suffix.strip().strip('-')}" if args.artifact_suffix.strip() else ""
    config["output_dir"] = f"model/artifacts/qwen3-4b-sweep-sft-gpu-max-{profile_name}{suffix}"
    config["dataset"]["train_path"] = str(args.training_dir / "train.jsonl")
    config["dataset"]["validation_path"] = str(args.training_dir / "validation.jsonl")
    training = config["training"]
    for key in [
        "sequence_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
    ]:
        training[key] = profile[key]
    if profile["gradient_checkpointing"]:
        training["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.wall_minutes is not None:
        training["max_wall_time_minutes"] = args.wall_minutes
    training["tokenized_cache_dir"] = f"data/training/tokenized_cache/qwen3_4b_sweep_sft_{profile_name}{suffix}"
    return config


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest = ensure_training_files(args)
    base = read_json(args.base_config)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    for profile in profile_sequence(args.profile):
        config = config_for_profile(base=base, profile=profile, args=args)
        config_path = args.run_dir / "configs" / f"{profile['name']}.json"
        write_json(config_path, config)
        configs.append({"profile": profile, "config_path": config_path, "config": config})

    plan = {
        "status": "prepared" if args.prepare_only else "starting",
        "created_at": now(),
        "base_model": "Qwen/Qwen3-4B",
        "manifest": manifest,
        "profiles": [
            {
                "name": item["profile"]["name"],
                "config_path": str(item["config_path"]),
                "sequence_length": item["profile"]["sequence_length"],
                "per_device_train_batch_size": item["profile"]["per_device_train_batch_size"],
                "gradient_accumulation_steps": item["profile"]["gradient_accumulation_steps"],
                "gradient_checkpointing": item["profile"]["gradient_checkpointing"],
            }
            for item in configs
        ],
    }
    write_json(args.run_dir / "run_plan.json", plan)
    print(json.dumps(plan, indent=2))
    if args.prepare_only:
        return

    attempts: list[dict[str, Any]] = []
    for item in configs:
        profile_name = item["profile"]["name"]
        log_dir = args.run_dir / profile_name
        command = [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(item["config_path"])]
        result = run_logged(command, log_dir=log_dir)
        attempts.append({"profile": profile_name, **result, "oom": looks_like_oom(log_dir)})
        write_json(args.run_dir / "attempts.json", {"attempts": attempts})
        if result["returncode"] == 0:
            write_json(args.run_dir / "final_result.json", {"status": "complete", "attempts": attempts})
            return
        if not attempts[-1]["oom"] or args.profile != "auto":
            write_json(args.run_dir / "final_result.json", {"status": "failed", "attempts": attempts})
            raise SystemExit(result["returncode"])
        print(f"[oom-fallback] {profile_name} failed with CUDA OOM; trying next profile.")

    write_json(args.run_dir / "final_result.json", {"status": "failed_all_profiles", "attempts": attempts})
    raise SystemExit(1)


if __name__ == "__main__":
    main()
