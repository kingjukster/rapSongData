"""RTX 4090-oriented multi-candidate lyric generation and transformation runner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .prompts import build_messages, build_revision_messages
from .scoring import clean_lyrics, rank_candidates


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "quality_max" / "rtx4090_qwen3_14b_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["generate", "style_transfer", "mutate", "improve"])
    parser.add_argument("--theme")
    parser.add_argument("--style")
    parser.add_argument("--keywords")
    parser.add_argument("--artist-reference")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--min-bars", type=int)
    parser.add_argument("--max-bars", type=int)
    parser.add_argument("--target-bars", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--revision-rounds", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--base-only", action="store_true", help="Ignore the configured LoRA adapter.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and require a complete local base-model cache or model_path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Quality-max configuration must be a JSON object")
    return payload


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    task = dict(config["task_defaults"])
    generation = dict(config["generation"])
    for key in ("mode", "theme", "style", "keywords", "artist_reference", "min_bars", "max_bars", "target_bars"):
        value = getattr(args, key)
        if value is not None:
            task[key] = value
    for key in ("candidates", "batch_size", "revision_rounds"):
        value = getattr(args, key)
        if value is not None:
            generation[key] = value
    if args.smoke:
        generation.update(config["smoke"])
    source_text = ""
    if args.source_file:
        source_text = args.source_file.read_text(encoding="utf-8")
    output_root = args.run_dir or Path(config["output_root"])
    return {
        **config,
        "task": task,
        "generation": generation,
        "source_file": str(args.source_file) if args.source_file else None,
        "source_text": source_text,
        "output_root": str(output_root),
        "adapter_enabled": bool(config.get("adapter_enabled", True) and not args.base_only),
        "local_files_only": bool(config.get("local_files_only", False) or args.local_files_only),
    }


def validate_settings(settings: dict[str, Any]) -> None:
    task = settings["task"]
    generation = settings["generation"]
    if not 1 <= int(task["min_bars"]) <= int(task["max_bars"]):
        raise ValueError("Expected 1 <= min_bars <= max_bars")
    if int(generation["candidates"]) < 1 or int(generation["batch_size"]) < 1:
        raise ValueError("candidates and batch_size must be positive")
    if int(generation["revision_rounds"]) < 0:
        raise ValueError("revision_rounds must be non-negative")
    if task["mode"] != "generate" and not settings["source_text"].strip():
        raise ValueError(f"{task['mode']} requires --source-file")
    adapter_dir = Path(settings["adapter_dir"])
    if settings["adapter_enabled"] and not adapter_dir.exists():
        raise FileNotFoundError(f"Configured adapter does not exist: {adapter_dir}")
    model_path = settings.get("model_path")
    if model_path and not Path(model_path).exists():
        raise FileNotFoundError(f"Configured local model_path does not exist: {model_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _configure_runtime(torch: Any) -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        function = getattr(torch.backends.cuda, name, None)
        if function:
            function(True)
    return {
        "float32_matmul_precision": "high",
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _generate_pool(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    count: int,
    batch_size: int,
    decoding: dict[str, Any],
    seed: int,
    stage: str,
    starting_index: int,
) -> tuple[list[dict[str, Any]], float, int]:
    formatted = _format_chat(tokenizer, messages)
    records: list[dict[str, Any]] = []
    total_tokens = 0
    generation_seconds = 0.0
    for offset in range(0, count, batch_size):
        current_batch = min(batch_size, count - offset)
        batch_seed = seed + starting_index + offset
        torch.manual_seed(batch_seed)
        encoded = tokenizer(
            [formatted] * current_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(decoding["max_input_tokens"]),
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=int(decoding["max_new_tokens"]),
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                top_k=int(decoding["top_k"]),
                repetition_penalty=float(decoding["repetition_penalty"]),
                no_repeat_ngram_size=int(decoding["no_repeat_ngram_size"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generation_seconds += time.perf_counter() - started
        for row_index, output in enumerate(outputs):
            generated = output[input_tokens:]
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            generated_tokens = int(generated.shape[-1])
            total_tokens += generated_tokens
            records.append(
                {
                    "candidate_index": starting_index + offset + row_index,
                    "stage": stage,
                    "batch_seed": batch_seed,
                    "generated_tokens": generated_tokens,
                    "raw_output": raw,
                    "lyrics": clean_lyrics(raw),
                }
            )
    return records, generation_seconds, total_tokens


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    settings = resolve_settings(args, load_config(args.config))
    validate_settings(settings)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("The quality-max lane requires local CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The quality-max lane requires BF16-capable CUDA hardware.")

    from peft import PeftModel

    task = settings["task"]
    generation = settings["generation"]
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(settings["output_root"]) / f"{timestamp}_{task['mode']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    wall_started = time.perf_counter()
    command = [sys.executable, *sys.argv]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.config, run_dir / "source_config.json")
    previous_excepthook = sys.excepthook

    def record_failure(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        failure_text = "".join(traceback.format_exception(exc_type, exc, tb))
        failure_summary = {
            "schema_version": 1,
            "lane": settings["lane"],
            "status": "failed",
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_seconds": round(time.perf_counter() - wall_started, 3),
            "command": command,
            "base_model": settings["base_model"],
            "adapter_dir": settings["adapter_dir"] if settings["adapter_enabled"] else None,
            "error_type": exc_type.__name__,
            "error": str(exc),
        }
        (run_dir / "generation.log").write_text(failure_text, encoding="utf-8")
        (run_dir / "run_summary.json").write_text(
            json.dumps(failure_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_summary.md").write_text(
            "\n".join(
                [
                    f"# Quality-max run: {task['mode']}",
                    "",
                    "- Status: `failed`",
                    f"- Error type: `{exc_type.__name__}`",
                    f"- Error: {exc}",
                    "",
                    "See `generation.log` for the traceback.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = record_failure

    runtime = _configure_runtime(torch)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    adapter_dir = Path(settings["adapter_dir"])
    model_source = settings.get("model_path") or settings["base_model"]
    tokenizer_source = adapter_dir if settings["adapter_enabled"] else model_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=settings.get("model_revision") if not Path(str(tokenizer_source)).exists() else None,
        use_fast=True,
        local_files_only=settings["local_files_only"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=settings.get("model_revision") if not Path(str(model_source)).exists() else None,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=settings.get("attn_implementation", "sdpa"),
        quantization_config=quantization,
        local_files_only=settings["local_files_only"],
    )
    if settings["adapter_enabled"]:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.config.use_cache = True
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    messages = build_messages(
        mode=task["mode"],
        theme=task["theme"],
        style=task["style"],
        keywords=task["keywords"],
        source_text=settings["source_text"],
        artist_reference=task.get("artist_reference", ""),
        min_bars=int(task["min_bars"]),
        max_bars=int(task["max_bars"]),
        target_bars=task.get("target_bars"),
    )
    all_candidates, generation_seconds, total_tokens = _generate_pool(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        count=int(generation["candidates"]),
        batch_size=int(generation["batch_size"]),
        decoding=generation,
        seed=int(generation["seed"]),
        stage="draft",
        starting_index=1,
    )
    ranked = rank_candidates(
        all_candidates,
        min_bars=int(task["min_bars"]),
        max_bars=int(task["max_bars"]),
        keywords=task["keywords"],
    )

    for round_index in range(int(generation["revision_rounds"])):
        revision_messages = build_revision_messages(
            lyrics=ranked[0]["lyrics"],
            theme=task["theme"],
            style=task["style"],
            keywords=task["keywords"],
            min_bars=int(task["min_bars"]),
            max_bars=int(task["max_bars"]),
            target_bars=task.get("target_bars"),
        )
        revisions, seconds, tokens = _generate_pool(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            messages=revision_messages,
            count=int(generation["revision_candidates"]),
            batch_size=int(generation["batch_size"]),
            decoding=generation,
            seed=int(generation["seed"]) + (round_index + 1) * 10000,
            stage=f"revision_{round_index + 1}",
            starting_index=len(all_candidates) + 1,
        )
        all_candidates.extend(revisions)
        generation_seconds += seconds
        total_tokens += tokens
        ranked = rank_candidates(
            all_candidates,
            min_bars=int(task["min_bars"]),
            max_bars=int(task["max_bars"]),
            keywords=task["keywords"],
        )

    ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    wall_seconds = time.perf_counter() - wall_started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    summary = {
        "schema_version": 1,
        "lane": settings["lane"],
        "status": "complete",
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": round(wall_seconds, 3),
        "command": command,
        "command_sha256": hashlib.sha256("\n".join(command).encode()).hexdigest(),
        "config_sha256": _sha256(args.config),
        "base_model": settings["base_model"],
        "model_revision": settings.get("model_revision"),
        "adapter_dir": str(adapter_dir) if settings["adapter_enabled"] else None,
        "task": task,
        "generation": generation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "runtime": runtime,
        },
        "metrics": {
            "model_load_seconds": round(load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "generated_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / max(generation_seconds, 1e-9), 3),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "candidate_count": len(ranked),
            "best_score": ranked[0]["metrics"]["score"],
            "best_bar_count": ranked[0]["metrics"]["bar_count"],
        },
        "artifacts": {
            "candidates": str(run_dir / "candidates.jsonl"),
            "best_json": str(run_dir / "best.json"),
            "best_text": str(run_dir / "best.txt"),
        },
        "quality_note": "Heuristic scores are pre-ranking aids, not ground-truth quality judgments.",
    }
    _write_jsonl(run_dir / "candidates.jsonl", ranked)
    (run_dir / "best.json").write_text(json.dumps(ranked[0], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "best.txt").write_text(ranked[0]["lyrics"] + "\n", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "run_summary.md").write_text(
        "\n".join(
            [
                f"# Quality-max run: {task['mode']}",
                "",
                f"- Status: `{summary['status']}`",
                f"- Base model: `{settings['base_model']}`",
                f"- Adapter: `{summary['adapter_dir'] or 'none'}`",
                f"- Candidates: {len(ranked)}",
                f"- Best bars: {summary['metrics']['best_bar_count']}",
                f"- Best heuristic score: {summary['metrics']['best_score']}",
                f"- Wall time: {summary['wall_seconds']} seconds",
                f"- Generation throughput: {summary['metrics']['tokens_per_second']} tokens/second",
                f"- Peak allocated VRAM: {summary['metrics']['peak_vram_gb']} GB",
                "",
                "The bar range is a soft objective. Candidates are not truncated or rejected solely for length.",
                "Heuristic ranking must be confirmed by human or calibrated model review before promotion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "generation.log").write_text(
        "\n".join(
            [
                f"started_at={started_at}",
                f"ended_at={ended_at}",
                f"status={summary['status']}",
                f"model_load_seconds={summary['metrics']['model_load_seconds']}",
                f"generation_seconds={summary['metrics']['generation_seconds']}",
                f"tokens_per_second={summary['metrics']['tokens_per_second']}",
                f"peak_vram_gb={summary['metrics']['peak_vram_gb']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sys.excepthook = previous_excepthook
    print(json.dumps({"run_dir": str(run_dir), "best": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
