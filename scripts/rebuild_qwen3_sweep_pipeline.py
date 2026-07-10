"""Rebuild the local Qwen3-4B sweep, curation, package, and SFT files.

This is the replacement path when the prior sweep files are unavailable. It is
local-only, uses Qwen/Qwen3-4B, and does not call OpenAI.
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
DEFAULT_ADAPTER = Path("model/artifacts/stage2-qwen3-4b-cleaned-chunks-512-60m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--sweep-dir", type=Path, default=Path("data/sweeps/qwen3_4b_rebuild"))
    parser.add_argument("--curation-dir", type=Path, default=Path("data/curation/qwen3_4b_rebuild"))
    parser.add_argument("--packaged-dir", type=Path, default=Path("data/packaged/qwen3_4b_sweep"))
    parser.add_argument("--training-dir", type=Path, default=Path("data/training/qwen3_4b_sweep_sft"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qwen3_rebuild_pipeline"))
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--training-profile", default="auto")
    parser.add_argument("--training-max-steps", type=int, default=None)
    parser.add_argument("--training-wall-minutes", type=float, default=None)
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], *, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as stderr:
        stdout.write(json.dumps({"event": "start", "time": started.isoformat(), "command": command}) + "\n")
        proc = subprocess.run(command, cwd=REPO, text=True, stdout=stdout, stderr=stderr, check=False)
        ended = dt.datetime.now()
        stdout.write(json.dumps({"event": "end", "time": ended.isoformat(), "returncode": proc.returncode}) + "\n")
    result = {
        "command": command,
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": round((ended - started).total_seconds(), 2),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See {stderr_path}")
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Qwen3-4B adapter not found: {args.adapter_dir}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    sweep_jsonl = args.sweep_dir / "sweep_raw.jsonl"
    sweep_summary = args.sweep_dir / "sweep_summary.json"
    results: list[dict[str, Any]] = []
    started = now()

    if not args.skip_sweep:
        results.append(
            run_command(
                [
                    sys.executable,
                    "scripts/run_qwen3_generation_sweep.py",
                    "--adapter-dir",
                    str(args.adapter_dir),
                    "--output-jsonl",
                    str(sweep_jsonl),
                    "--summary-json",
                    str(sweep_summary),
                    "--num-candidates",
                    str(args.num_candidates),
                    "--batch-size",
                    str(args.batch_size),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                log_dir=args.run_dir / "01_sweep",
            )
        )
    elif not sweep_jsonl.exists():
        raise FileNotFoundError(f"--skip-sweep requested but missing sweep file: {sweep_jsonl}")

    results.append(
        run_command(
            [
                sys.executable,
                "scripts/curate_qwen3_sweep.py",
                "--input",
                str(sweep_jsonl),
                "--output-dir",
                str(args.curation_dir),
            ],
            log_dir=args.run_dir / "02_curate",
        )
    )
    results.append(
        run_command(
            [
                sys.executable,
                "scripts/package_qwen3_sweep_datasets.py",
                "--input",
                str(args.curation_dir / "annotated_sweep.jsonl"),
                "--output-dir",
                str(args.packaged_dir),
            ],
            log_dir=args.run_dir / "03_package",
        )
    )
    results.append(
        run_command(
            [
                sys.executable,
                "scripts/build_qwen3_sft_training_files.py",
                "--packaged-dir",
                str(args.packaged_dir),
                "--output-dir",
                str(args.training_dir),
            ],
            log_dir=args.run_dir / "04_build_training_files",
        )
    )

    train_command = [
        sys.executable,
        "scripts/run_qwen3_sweep_sft_gpu_max.py",
        "--packaged-dir",
        str(args.packaged_dir),
        "--training-dir",
        str(args.training_dir),
        "--run-dir",
        str(args.run_dir / "05_training"),
        "--skip-build",
        "--profile",
        args.training_profile,
    ]
    if args.training_max_steps is not None:
        train_command += ["--max-steps", str(args.training_max_steps)]
    if args.training_wall_minutes is not None:
        train_command += ["--wall-minutes", str(args.training_wall_minutes)]
    if not args.start_training:
        train_command.append("--prepare-only")
    results.append(run_command(train_command, log_dir=args.run_dir / "05_training_prepare_or_run"))

    summary = {
        "status": "complete",
        "local_only": True,
        "base_model_scope": "Qwen/Qwen3-4B",
        "started_at": started,
        "ended_at": now(),
        "outputs": {
            "sweep_jsonl": str(sweep_jsonl),
            "curation_dir": str(args.curation_dir),
            "packaged_dir": str(args.packaged_dir),
            "training_dir": str(args.training_dir),
            "training_run_dir": str(args.run_dir / "05_training"),
        },
        "commands": results,
        "training_started": args.start_training,
    }
    write_json(args.run_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
