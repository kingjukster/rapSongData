"""Run a short three-model local CUDA train/generation benchmark."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO / "model" / "artifacts" / "three-model-1hr"
REPORTS = REPO / "reports"

GENERATION_ARGS = {
    "max_new_tokens": 120,
    "temperature": 0.78,
    "top_p": 0.88,
    "top_k": 50,
    "repetition_penalty": 1.18,
    "no_repeat_ngram_size": 5,
    "seed": 201,
}

STAGES = [
    {
        "name": "qwen2.5-7b",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "config": "model/configs/local_cuda_qwen2_5_7b_three_model_1hr.json",
        "output_dir": "model/artifacts/three-model-1hr/qwen2.5-7b-cleaned-chunks-256-rerun",
        "title": "Three-Model 1hr Qwen2.5-7B Eval",
    },
]


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def run_command(command: list[str], log_path: Path) -> dict[str, Any]:
    started = dt.datetime.now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"event": "start", "time": started.isoformat(), "command": command}) + "\n")
        proc = subprocess.run(command, cwd=REPO, stdout=log, stderr=log, text=True)
        ended = dt.datetime.now()
        log.write(json.dumps({"event": "end", "time": ended.isoformat(), "returncode": proc.returncode}) + "\n")
    return {
        "command": command,
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": round((ended - started).total_seconds(), 2),
        "log_path": str(log_path.relative_to(REPO)),
    }


def reused_result(command: list[str], log_path: Path, reason: str) -> dict[str, Any]:
    return {
        "command": command,
        "returncode": 0,
        "started_at": None,
        "ended_at": None,
        "wall_seconds": 0.0,
        "log_path": str(log_path.relative_to(REPO)),
        "reused": True,
        "reason": reason,
    }


def gpu_snapshot() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi.exe",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    return (proc.stdout or proc.stderr).strip()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def has_completed_adapter(output_dir: Path) -> bool:
    return (
        (output_dir / "training_summary.json").exists()
        and (output_dir / "adapter_model.safetensors").exists()
        and (output_dir / "adapter_config.json").exists()
    )


def has_generation_outputs(jsonl_path: Path, md_path: Path) -> bool:
    return md_path.exists() and len(read_jsonl(jsonl_path)) == 5


def summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    analyses = [record.get("analysis", {}) for record in records]
    timings = [record.get("timing", {}) for record in records]
    tokens_per_second = [
        float(item["tokens_per_second"])
        for item in timings
        if isinstance(item.get("tokens_per_second"), (int, float))
    ]
    return {
        "prompt_count": len(records),
        "slur_prompt_count": sum(1 for item in analyses if int(item.get("slur_count") or 0) > 0),
        "total_slurs": sum(int(item.get("slur_count") or 0) for item in analyses),
        "exact_line_match_count": sum(1 for item in analyses if item.get("exact_line_match") is True),
        "hook_pass_count": sum(1 for item in analyses if item.get("hook_line_cap_ok") is True),
        "avg_repeated_line_ratio": round(
            sum(float(item.get("repeated_line_ratio") or 0.0) for item in analyses) / max(1, len(analyses)),
            4,
        ),
        "avg_tokens_per_second": round(sum(tokens_per_second) / max(1, len(tokens_per_second)), 2),
    }


def stage_generation_command(
    stage: dict[str, str], md_path: Path, jsonl_path: Path, summary_json: Path, summary_md: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "model/run_fixed_generation_eval.py",
        "--base-model",
        stage["base_model"],
        "--adapter-dir",
        stage["output_dir"],
        "--output-md",
        str(md_path.relative_to(REPO)),
        "--output-jsonl",
        str(jsonl_path.relative_to(REPO)),
        "--max-new-tokens",
        str(GENERATION_ARGS["max_new_tokens"]),
        "--temperature",
        str(GENERATION_ARGS["temperature"]),
        "--top-p",
        str(GENERATION_ARGS["top_p"]),
        "--top-k",
        str(GENERATION_ARGS["top_k"]),
        "--repetition-penalty",
        str(GENERATION_ARGS["repetition_penalty"]),
        "--no-repeat-ngram-size",
        str(GENERATION_ARGS["no_repeat_ngram_size"]),
        "--seed",
        str(GENERATION_ARGS["seed"]),
        "--run-summary-json",
        str(summary_json),
        "--run-summary-md",
        str(summary_md),
        "--disable-thinking",
        "--title",
        stage["title"],
    ]
    return command


def write_suite_summary(stage_results: list[dict[str, Any]], suite_started: str, suite_ended: str) -> None:
    def stage_ok(stage: dict[str, Any]) -> bool:
        train = stage.get("train") or {}
        generation = stage.get("generation") or {}
        return train.get("returncode") == 0 and generation.get("returncode") == 0

    summary = {
        "status": "complete" if len(stage_results) == len(STAGES) and all(stage_ok(stage) for stage in stage_results) else "partial",
        "started_at": suite_started,
        "ended_at": suite_ended,
        "generation_args": GENERATION_ARGS,
        "stages": stage_results,
    }
    (RUN_ROOT / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Three-Model One-Hour Benchmark",
        "",
        f"- Started: {suite_started}",
        f"- Ended: {suite_ended}",
        f"- Status: {summary['status']}",
        "",
        "## Stages",
        "",
    ]
    for stage in stage_results:
        train_summary = stage.get("training_summary") or {}
        result = train_summary.get("result") or {}
        timing = train_summary.get("timing_summary") or {}
        gen = stage.get("generation_summary") or {}
        generation_result = stage.get("generation") or {}
        lines.extend(
            [
                f"### {stage['name']}",
                "",
                f"- Train return code: {stage['train']['returncode']}",
                f"- Generation return code: {generation_result.get('returncode')}",
                f"- Output: `{stage['output_dir']}`",
                f"- Train wall time: {result.get('train_runtime_minutes')} min",
                f"- Train loss: {result.get('trainer_metrics', {}).get('train_loss')}",
                f"- Avg train tokens/sec: {timing.get('average_estimated_tokens_per_second')}",
                f"- Peak allocated VRAM: {timing.get('peak_max_memory_allocated_gb')} GB",
                f"- Generation prompts: {gen.get('prompt_count')}",
                f"- Slur outputs: {gen.get('slur_prompt_count')} prompts, {gen.get('total_slurs')} terms",
                f"- Exact 16-line matches: {gen.get('exact_line_match_count')}",
                f"- Avg repeated-line ratio: {gen.get('avg_repeated_line_ratio')}",
                "",
            ]
        )
    (RUN_ROOT / "suite_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    suite_started = now()
    stage_results: list[dict[str, Any]] = []
    for stage in STAGES:
        output_dir = REPO / stage["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        stage_result: dict[str, Any] = {
            "name": stage["name"],
            "base_model": stage["base_model"],
            "config": stage["config"],
            "output_dir": stage["output_dir"],
            "gpu_before": gpu_snapshot(),
        }

        train_command = [sys.executable, "-u", "model/train_local_cuda.py", "--config", stage["config"]]
        if has_completed_adapter(output_dir):
            stage_result["train"] = reused_result(
                train_command,
                output_dir / "benchmark_train.log",
                "existing training_summary and adapter found",
            )
        else:
            stage_result["train"] = run_command(train_command, output_dir / "benchmark_train.log")
        stage_result["training_summary"] = read_json(output_dir / "training_summary.json")
        stage_result["gpu_after_train"] = gpu_snapshot()

        if stage_result["train"]["returncode"] == 0:
            safe_name = stage["name"].replace(".", "_").replace("-", "_")
            md_path = REPORTS / f"three_model_1hr_{safe_name}_generation.md"
            jsonl_path = REPORTS / f"three_model_1hr_{safe_name}_generation.jsonl"
            run_summary_json = REPORTS / f"three_model_1hr_{safe_name}_generation_run_summary.json"
            run_summary_md = REPORTS / f"three_model_1hr_{safe_name}_generation_run_summary.md"
            gen_command = stage_generation_command(
                stage,
                md_path,
                jsonl_path,
                run_summary_json,
                run_summary_md,
            )
            if has_generation_outputs(jsonl_path, md_path):
                stage_result["generation"] = reused_result(
                    gen_command,
                    output_dir / "benchmark_generation.log",
                    "existing generation report found",
                )
            else:
                stage_result["generation"] = run_command(gen_command, output_dir / "benchmark_generation.log")
            stage_result["generation_md"] = str(md_path.relative_to(REPO))
            stage_result["generation_jsonl"] = str(jsonl_path.relative_to(REPO))
            stage_result["generation_summary_json"] = str(run_summary_json.relative_to(REPO))
            stage_result["generation_summary_md"] = str(run_summary_md.relative_to(REPO))
            stage_result["generation_summary"] = summarize_generation(read_jsonl(jsonl_path))
            stage_result["gpu_after_generation"] = gpu_snapshot()
        else:
            stage_result["generation"] = None
            stage_result["generation_summary"] = None

        stage_results.append(stage_result)
        write_suite_summary(stage_results, suite_started, now())

    write_suite_summary(stage_results, suite_started, now())


if __name__ == "__main__":
    main()
