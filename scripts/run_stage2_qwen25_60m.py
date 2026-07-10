"""Run the Stage 2 practical Qwen2.5 local CUDA benchmark."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUN_TITLE = "Stage 2 Qwen2.5 60-Minute Practical Benchmark"
GENERATION_TITLE = "Stage 2 Qwen2.5 512 60m Generation Eval"
CONFIG = Path("configs/training/local_cuda_qwen2_5_7b_stage2_512_60m.json")
OUTPUT_DIR = Path("model/artifacts/stage2-qwen2.5-7b-cleaned-chunks-512-60m")
PROMPTS_FILE = Path("configs/prompts/stage2_fixed_prompts_12.txt")
GEN_MD = Path("reports/stage2_qwen2_5_7b_512_60m_generation.md")
GEN_JSONL = Path("reports/stage2_qwen2_5_7b_512_60m_generation.jsonl")

GENERATION_ARGS = {
    "max_new_tokens": 180,
    "temperature": 0.78,
    "top_p": 0.88,
    "top_k": 50,
    "repetition_penalty": 1.18,
    "no_repeat_ngram_size": 5,
    "seed": 301,
}


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
        "log_path": str(log_path if not log_path.is_absolute() else log_path.relative_to(REPO)),
    }


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


def has_adapter(output_dir: Path) -> bool:
    return (
        (output_dir / "training_summary.json").exists()
        and (output_dir / "adapter_model.safetensors").exists()
        and (output_dir / "adapter_config.json").exists()
    )


def has_generation() -> bool:
    return GEN_MD.exists() and len(read_jsonl(GEN_JSONL)) == len(
        [line for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    )


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


def write_summary(summary: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    training = summary.get("training_summary") or {}
    result = training.get("result") or {}
    timing = training.get("timing_summary") or {}
    generation = summary.get("generation_summary") or {}
    lines = [
        f"# {RUN_TITLE}",
        "",
        f"- Status: {summary['status']}",
        f"- Started: {summary['started_at']}",
        f"- Ended: {summary['ended_at']}",
        f"- Output: `{OUTPUT_DIR.as_posix()}`",
        f"- Generation report: `{GEN_MD.as_posix()}`",
        "",
        "## Training",
        "",
        f"- Train return code: {summary.get('train', {}).get('returncode')}",
        f"- Train wall time: {result.get('train_runtime_minutes')} min",
        f"- Status: {result.get('status')}",
        f"- Triggered time budget: {(result.get('time_budget') or {}).get('triggered')}",
        f"- Triggered step: {(result.get('time_budget') or {}).get('triggered_step')}",
        f"- Train loss: {(result.get('trainer_metrics') or {}).get('train_loss')}",
        f"- Avg estimated train tokens/sec: {timing.get('average_estimated_tokens_per_second')}",
        f"- Avg seconds/step: {timing.get('average_seconds_per_step')}",
        f"- Peak allocated VRAM: {timing.get('peak_max_memory_allocated_gb')} GB",
        "",
        "## Generation",
        "",
        f"- Generation return code: {summary.get('generation', {}).get('returncode')}",
        f"- Prompts: {generation.get('prompt_count')}",
        f"- Avg generation tokens/sec: {generation.get('avg_tokens_per_second')}",
        f"- Slur outputs: {generation.get('slur_prompt_count')} prompts, {generation.get('total_slurs')} terms",
        f"- Exact line-count matches: {generation.get('exact_line_match_count')}",
        f"- Hook line-cap passes: {generation.get('hook_pass_count')}",
        f"- Avg repeated-line ratio: {generation.get('avg_repeated_line_ratio')}",
    ]
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    started = now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage2_active_pid.txt").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    train_command = [sys.executable, "-u", "model/train_local_cuda.py", "--config", str(CONFIG)]
    gen_command = [
        sys.executable,
        "-u",
        "model/run_fixed_generation_eval.py",
        "--base-model",
        BASE_MODEL,
        "--adapter-dir",
        str(OUTPUT_DIR),
        "--output-md",
        str(GEN_MD),
        "--output-jsonl",
        str(GEN_JSONL),
        "--prompts-file",
        str(PROMPTS_FILE),
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
        "--disable-thinking",
        "--title",
        GENERATION_TITLE,
    ]

    if has_adapter(OUTPUT_DIR):
        train_result = {"command": train_command, "returncode": 0, "reused": True, "reason": "adapter exists"}
    else:
        train_result = run_command(train_command, OUTPUT_DIR / "stage2_train.log")

    generation_result: dict[str, Any] | None = None
    if train_result["returncode"] == 0:
        if has_generation():
            generation_result = {
                "command": gen_command,
                "returncode": 0,
                "reused": True,
                "reason": "generation outputs exist",
            }
        else:
            generation_result = run_command(gen_command, OUTPUT_DIR / "stage2_generation.log")

    generation_records = read_jsonl(GEN_JSONL)
    summary = {
        "status": "complete" if train_result["returncode"] == 0 and generation_result and generation_result["returncode"] == 0 else "partial",
        "started_at": started,
        "ended_at": now(),
        "config": str(CONFIG),
        "prompts_file": str(PROMPTS_FILE),
        "generation_args": GENERATION_ARGS,
        "train": train_result,
        "training_summary": read_json(OUTPUT_DIR / "training_summary.json"),
        "generation": generation_result,
        "generation_md": str(GEN_MD),
        "generation_jsonl": str(GEN_JSONL),
        "generation_summary": summarize_generation(generation_records),
    }
    write_summary(summary)


if __name__ == "__main__":
    main()
