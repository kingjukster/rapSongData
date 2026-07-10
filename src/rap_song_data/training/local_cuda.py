"""Train the rap lyrics LoRA adapter on a local CUDA GPU.

This is the local replacement for the Runpod Flash training endpoint. It reads
JSONL files already built under ``model/data*`` and writes the adapter directly
to ``model/artifacts``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience for local auth.
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_CONFIG_PATH = Path("configs/training/local_cuda_config.example.json")
STRUCTURAL_SPECIAL_TOKENS = ["<|verse_start|>", "<|verse_end|>", "<|bar_start|>"]
CLEANED_CORPUS_DIR = Path("data/cleaned")
CLEANED_TRAIN_PATH = CLEANED_CORPUS_DIR / "categorized_rap_corpus_train.txt"
CLEANED_VALIDATION_PATH = CLEANED_CORPUS_DIR / "categorized_rap_corpus_validation.txt"
CLEANING_SUMMARY_PATH = CLEANED_CORPUS_DIR / "corpus_cleaning_summary.json"
CLEANING_REPORT_PATH = Path("reports/corpus_cleaning_report.md")
CLEANING_REPORT_SUMMARY_PATH = Path("reports/corpus_cleaning_summary.json")
DEFAULT_CONFIG_TRAIN_PATHS = {
    Path("model/data/train.jsonl"),
    Path("model/data_full_verses/train.jsonl"),
    Path("model/data_special_tokens/train.jsonl"),
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Local CUDA config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--timing-log-steps", type=int, default=None)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--add-structural-special-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.train_path is not None:
        config["dataset"]["train_path"] = str(args.train_path)
    if args.validation_path is not None:
        config["dataset"]["validation_path"] = str(args.validation_path)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.base_model is not None:
        config["base_model"] = args.base_model
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.sequence_length is not None:
        config["training"]["sequence_length"] = args.sequence_length
    if args.timing_log_steps is not None:
        config["training"]["timing_log_steps"] = args.timing_log_steps
    if args.load_in_4bit is not None:
        config["training"]["load_in_4bit"] = args.load_in_4bit
    if args.add_structural_special_tokens is not None:
        config["add_structural_special_tokens"] = args.add_structural_special_tokens
    if args.resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = str(args.resume_from_checkpoint)
    return config


def validate_paths(config: dict[str, Any]) -> None:
    dataset_cfg = config["dataset"]
    for key in ["train_path", "validation_path"]:
        path = Path(dataset_cfg[key])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found for {key}: {path}")


def infer_dataset_format(path: Path, dataset_cfg: dict[str, Any]) -> str:
    configured = str(dataset_cfg.get("format") or "").lower().strip()
    if configured:
        return configured
    if path.suffix.lower() in {".txt", ".text"}:
        return "text"
    return "json"


def prefer_cleaned_corpus(config: dict[str, Any], args: argparse.Namespace) -> None:
    dataset_cfg = config.setdefault("dataset", {})
    explicit_train = args.train_path is not None
    explicit_validation = args.validation_path is not None
    if explicit_train:
        return
    configured_train_path = Path(str(dataset_cfg.get("train_path", "")))
    if configured_train_path and configured_train_path not in DEFAULT_CONFIG_TRAIN_PATHS:
        print(f"[corpus] using configured dataset path: {configured_train_path}")
        return

    if CLEANED_TRAIN_PATH.exists():
        dataset_cfg["train_path"] = str(CLEANED_TRAIN_PATH)
        dataset_cfg["format"] = "text"
        dataset_cfg["text_field"] = "text"
        if not explicit_validation:
            if CLEANED_VALIDATION_PATH.exists():
                dataset_cfg["validation_path"] = str(CLEANED_VALIDATION_PATH)
            else:
                dataset_cfg["validation_path"] = str(CLEANED_TRAIN_PATH)
                print(
                    "[corpus] WARNING: cleaned train text exists but cleaned validation text is missing; "
                    "using the train text for validation loading because evaluation is disabled."
                )
        dataset_cfg["cleaning_summary_path"] = str(CLEANING_SUMMARY_PATH)
        dataset_cfg["cleaning_report_path"] = str(CLEANING_REPORT_PATH)
        dataset_cfg["cleaning_report_summary_path"] = str(CLEANING_REPORT_SUMMARY_PATH)
        print(f"[corpus] using cleaned corpus by default: {CLEANED_TRAIN_PATH}")
        return

    current_train = Path(dataset_cfg.get("train_path", ""))
    print(
        "[corpus] WARNING: cleaned corpus not found at "
        f"{CLEANED_TRAIN_PATH}. Training will use configured raw/model data path: {current_train}. "
        "Run `python scripts/build_categorized_rap_corpus.py --clean` before the next quality run."
    )


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def estimate_dataset_file(path: Path, dataset_format: str, text_field: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    record_count = 0
    char_count = 0
    word_count = 0
    if dataset_format == "text":
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                char_count += len(line)
                word_count += len(line.split())
                if line.strip() == "<|end|>":
                    record_count += 1
        if record_count == 0 and char_count:
            record_count = 1
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                record_count += 1
                try:
                    payload = json.loads(stripped)
                    text = str(payload.get(text_field) or payload.get("text") or "")
                except json.JSONDecodeError:
                    text = stripped
                char_count += len(text)
                word_count += len(text.split())
    return {
        "path": str(path),
        "exists": True,
        "record_count": record_count,
        "char_count": char_count,
        "estimated_tokens": int(max(word_count * 1.3, char_count / 4.0)),
    }


def read_delimited_text_records(path: Path) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line.strip() and not current:
                continue
            current.append(line)
            if line.strip() == "<|end|>":
                record = "\n".join(current).strip()
                if record:
                    records.append(record)
                current = []
    trailing = "\n".join(current).strip()
    if trailing:
        records.append(trailing)
    return records


def load_training_dataset(
    *,
    dataset_cfg: dict[str, Any],
    dataset_format: str,
    text_field: str,
    load_dataset_fn: Any,
    dataset_cls: Any,
    dataset_dict_cls: Any,
) -> Any:
    train_path = Path(dataset_cfg["train_path"])
    validation_path = Path(dataset_cfg["validation_path"])
    if dataset_format == "text":
        return dataset_dict_cls(
            {
                "train": dataset_cls.from_dict({text_field: read_delimited_text_records(train_path)}),
                "validation": dataset_cls.from_dict({text_field: read_delimited_text_records(validation_path)}),
            }
        )
    return load_dataset_fn(
        dataset_format,
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )


def apply_dataset_limits(dataset: Any, *, training_cfg: dict[str, Any]) -> Any:
    train_limit = training_cfg.get("max_train_records")
    validation_limit = training_cfg.get("max_validation_records")
    if train_limit is None and validation_limit is None:
        return dataset

    subset_seed = training_cfg.get("subset_shuffle_seed")
    for split, configured_limit in [
        ("train", train_limit),
        ("validation", validation_limit),
    ]:
        if configured_limit is None:
            continue
        limit = int(configured_limit)
        if limit <= 0:
            continue
        split_dataset = dataset[split]
        if subset_seed is not None:
            split_dataset = split_dataset.shuffle(seed=int(subset_seed))
        limit = min(limit, len(split_dataset))
        dataset[split] = split_dataset.select(range(limit))
        print(f"[dataset] using {limit} {split} records for this run")
    return dataset


def corpus_training_log(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    train_path = Path(dataset_cfg["train_path"])
    validation_path = Path(dataset_cfg["validation_path"])
    dataset_format = infer_dataset_format(train_path, dataset_cfg)
    text_field = dataset_cfg.get("text_field", "text" if dataset_format == "text" else "training_text")
    summary_path = Path(dataset_cfg.get("cleaning_summary_path") or CLEANING_SUMMARY_PATH)
    report_summary_path = Path(dataset_cfg.get("cleaning_report_summary_path") or CLEANING_REPORT_SUMMARY_PATH)
    if not summary_path.exists() and report_summary_path.exists():
        summary_path = report_summary_path
    summary = load_json_if_exists(summary_path)
    train_estimate = estimate_dataset_file(train_path, dataset_format, text_field)
    report_path = dataset_cfg.get("cleaning_report_path") or str(CLEANING_REPORT_PATH)
    chunk_manifest_path = dataset_cfg.get("chunk_manifest_path")
    chunk_manifest = load_json_if_exists(Path(chunk_manifest_path)) if chunk_manifest_path else None

    log_payload: dict[str, Any] = {
        "dataset_format": dataset_format,
        "text_field": text_field,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "record_count": train_estimate.get("record_count"),
        "estimated_tokens": train_estimate.get("estimated_tokens"),
        "included_quality_tiers": None,
        "report_path": report_path if Path(report_path).exists() else None,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "report_summary_path": str(report_summary_path) if report_summary_path.exists() else None,
        "chunk_manifest_path": str(chunk_manifest_path) if chunk_manifest_path and Path(chunk_manifest_path).exists() else None,
        "raw_only_warning": not summary_path.exists(),
    }
    if summary:
        counts = summary.get("counts", {})
        quality = summary.get("quality", {})
        log_payload.update(
            {
                "record_count": counts.get("train_records", log_payload["record_count"]),
                "estimated_tokens": counts.get("estimated_train_tokens", log_payload["estimated_tokens"]),
                "included_quality_tiers": quality.get("included_quality_tiers"),
                "cleaning_counts": counts,
                "raw_only_warning": False,
            }
        )
    if chunk_manifest:
        train_chunk_summary = chunk_manifest.get("train", {})
        log_payload.update(
            {
                "record_count": train_chunk_summary.get("chunk_count", log_payload["record_count"]),
                "estimated_tokens": train_chunk_summary.get("estimated_tokens", log_payload["estimated_tokens"]),
                "chunk_summary": train_chunk_summary,
            }
        )
    print("[corpus] " + json.dumps(log_payload, indent=2))
    return log_payload


def cuda_summary(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    gb = 1024**3
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "capability": f"{props.major}.{props.minor}",
        "total_vram_gb": round(props.total_memory / gb, 2),
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def configure_torch_runtime(torch: Any, training_cfg: dict[str, Any]) -> dict[str, Any]:
    """Enable safe RTX 50-series speed paths before model allocation."""
    precision = str(training_cfg.get("float32_matmul_precision", "high"))
    allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    cudnn_benchmark = bool(training_cfg.get("cudnn_benchmark", True))
    try:
        torch.set_float32_matmul_precision(precision)
    except Exception:
        precision = "unavailable"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        try:
            torch.backends.cuda.enable_flash_sdp(bool(training_cfg.get("enable_flash_sdp", True)))
            torch.backends.cuda.enable_mem_efficient_sdp(bool(training_cfg.get("enable_mem_efficient_sdp", True)))
            torch.backends.cuda.enable_math_sdp(bool(training_cfg.get("enable_math_sdp", True)))
        except Exception:
            pass
    return {
        "float32_matmul_precision": precision,
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "enable_flash_sdp": bool(training_cfg.get("enable_flash_sdp", True)),
        "enable_mem_efficient_sdp": bool(training_cfg.get("enable_mem_efficient_sdp", True)),
    }


def parse_timing_step(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        step_text = value.split("/", 1)[0].strip()
        if step_text.isdigit():
            return int(step_text)
    return None


def summarize_timing_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"intervals_logged": 0}

    normalized: list[dict[str, Any]] = []
    for record in records:
        step_number = parse_timing_step(record.get("step"))
        if step_number is None:
            continue
        normalized.append(
            {
                "step": step_number,
                "seconds_per_step": float(record["seconds_per_step"]),
                "estimated_tokens_per_second": float(record["estimated_tokens_per_second"]),
                "max_memory_allocated_gb": float(record["max_memory_allocated_gb"])
                if "max_memory_allocated_gb" in record
                else None,
            }
        )

    if not normalized:
        return {"intervals_logged": 0}

    def avg(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    normalized.sort(key=lambda record: record["step"])
    window_size = min(3, len(normalized))
    first_window = normalized[:window_size]
    last_window = normalized[-window_size:]
    seconds_per_step_values = [record["seconds_per_step"] for record in normalized]
    token_values = [record["estimated_tokens_per_second"] for record in normalized]
    max_memory_values = [
        record["max_memory_allocated_gb"]
        for record in normalized
        if record["max_memory_allocated_gb"] is not None
    ]
    first_window_avg = avg([record["seconds_per_step"] for record in first_window])
    last_window_avg = avg([record["seconds_per_step"] for record in last_window])
    slowdown_percent = ((last_window_avg - first_window_avg) / first_window_avg * 100.0) if first_window_avg else 0.0

    return {
        "intervals_logged": len(normalized),
        "first_logged_step": normalized[0]["step"],
        "last_logged_step": normalized[-1]["step"],
        "average_seconds_per_step": round(avg(seconds_per_step_values), 3),
        "average_estimated_tokens_per_second": round(avg(token_values), 1),
        "best_estimated_tokens_per_second": round(max(token_values), 1),
        "worst_estimated_tokens_per_second": round(min(token_values), 1),
        "first_window_average_seconds_per_step": round(first_window_avg, 3),
        "last_window_average_seconds_per_step": round(last_window_avg, 3),
        "slowdown_percent_first_to_last_window": round(slowdown_percent, 1),
        "peak_max_memory_allocated_gb": round(max(max_memory_values), 2) if max_memory_values else None,
    }


def write_training_summary(
    *,
    output_dir: Path,
    base_model: str,
    dataset_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    result: dict[str, Any],
    timing_records: list[dict[str, Any]],
    corpus_log: dict[str, Any],
) -> Path:
    summary_path = output_dir / "training_summary.json"
    summary_payload = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "dataset": {
            "train_path": dataset_cfg["train_path"],
            "validation_path": dataset_cfg["validation_path"],
            "text_field": dataset_cfg.get("text_field", "training_text"),
            "max_train_records": training_cfg.get("max_train_records"),
            "max_validation_records": training_cfg.get("max_validation_records"),
            "subset_shuffle_seed": training_cfg.get("subset_shuffle_seed"),
        },
        "training": {
            "max_steps": int(training_cfg["max_steps"]),
            "sequence_length": int(training_cfg["sequence_length"]),
            "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
            "learning_rate": float(training_cfg["learning_rate"]),
            "timing_log_steps": int(training_cfg.get("timing_log_steps", 10)),
            "max_wall_time_minutes": training_cfg.get("max_wall_time_minutes"),
            "load_in_4bit": bool(training_cfg.get("load_in_4bit", True)),
            "bf16": bool(training_cfg.get("bf16", True)),
            "tokenized_cache_dir": training_cfg.get("tokenized_cache_dir"),
        },
        "result": result,
        "timing_summary": summarize_timing_records(timing_records),
        "corpus": corpus_log,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)
        file.write("\n")
    return summary_path


def main() -> None:
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import torch
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing local training dependencies. Install them with:\n"
            "  pip install -r requirements-local-cuda.txt"
        ) from exc

    args = parse_args()
    config = resolve_config(args)
    prefer_cleaned_corpus(config, args)
    validate_paths(config)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch. Check your PyTorch CUDA install.")

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    base_model = config["base_model"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = configure_torch_runtime(torch, training_cfg)
    corpus_log = corpus_training_log(dataset_cfg)
    summary_source = corpus_log.get("summary_path")
    if summary_source:
        shutil.copyfile(summary_source, output_dir / "corpus_cleaning_summary.json")

    print(json.dumps({"cuda": cuda_summary(torch), "runtime": runtime_cfg, "base_model": base_model}, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    structural_token_ids: list[int] = []
    if bool(config.get("add_structural_special_tokens", False)):
        tokenizer.add_special_tokens({"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS})
        structural_token_ids = tokenizer.convert_tokens_to_ids(STRUCTURAL_SPECIAL_TOKENS)
        if any(token_id is None or token_id < 0 for token_id in structural_token_ids):
            raise ValueError(f"Failed to add structural special tokens: {STRUCTURAL_SPECIAL_TOKENS}")

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if training_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = training_cfg["attn_implementation"]
    if bool(training_cfg.get("load_in_4bit", True)):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bool(training_cfg.get("bf16", True)) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if bool(training_cfg.get("bf16", True)) else torch.float16

    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        except (TypeError, ValueError) as exc:
            if "attn_implementation" not in model_kwargs:
                raise
            print(
                "[runtime] WARNING: configured attn_implementation was rejected; "
                "retrying model load with the model default attention implementation."
            )
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as exc:
        if bool(training_cfg.get("load_in_4bit", True)):
            raise SystemExit(
                "Could not load the model in 4-bit mode. On a 12 GB RTX 5070, 4-bit QLoRA is the expected path.\n"
                "Make sure bitsandbytes is installed and supports your local CUDA/PyTorch build."
            ) from exc
        raise

    if structural_token_ids:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if bool(training_cfg.get("gradient_checkpointing", True)):
        gradient_checkpointing_kwargs = training_cfg.get("gradient_checkpointing_kwargs")
        if isinstance(gradient_checkpointing_kwargs, dict):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                print("[runtime] WARNING: gradient_checkpointing_kwargs unsupported; using default checkpointing.")
                model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    dataset_format = infer_dataset_format(Path(dataset_cfg["train_path"]), dataset_cfg)
    text_field = dataset_cfg.get("text_field", "text" if dataset_format == "text" else "training_text")
    dataset = load_training_dataset(
        dataset_cfg=dataset_cfg,
        dataset_format=dataset_format,
        text_field=text_field,
        load_dataset_fn=load_dataset,
        dataset_cls=Dataset,
        dataset_dict_cls=DatasetDict,
    )
    dataset = apply_dataset_limits(dataset, training_cfg=training_cfg)
    sequence_length = int(training_cfg["sequence_length"])

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=sequence_length,
            padding=False,
        )

    tokenized_cache_dir = training_cfg.get("tokenized_cache_dir")
    tokenized_cache_path = Path(tokenized_cache_dir) if tokenized_cache_dir else None
    tokenized_cache_meta_path = (
        tokenized_cache_path / "tokenized_cache_meta.json" if tokenized_cache_path else None
    )
    expected_cache_meta = {
        "base_model": base_model,
        "train_path": str(dataset_cfg["train_path"]),
        "validation_path": str(dataset_cfg["validation_path"]),
        "text_field": text_field,
        "dataset_format": dataset_format,
        "sequence_length": sequence_length,
        "tokenizer_length": len(tokenizer),
    }
    tokenized = None
    if tokenized_cache_path and tokenized_cache_path.exists() and tokenized_cache_meta_path:
        cached_meta = load_json_if_exists(tokenized_cache_meta_path)
        if cached_meta == expected_cache_meta:
            print(f"[dataset] loading tokenized dataset cache: {tokenized_cache_path}")
            tokenized = load_from_disk(str(tokenized_cache_path))
        else:
            print(f"[dataset] tokenized cache metadata mismatch; rebuilding: {tokenized_cache_path}")

    if tokenized is None:
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing training text",
            num_proc=int(training_cfg.get("tokenize_num_proc") or max(1, min(4, os.cpu_count() or 1))),
        )
        if tokenized_cache_path and tokenized_cache_meta_path:
            tokenized_cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[dataset] saving tokenized dataset cache: {tokenized_cache_path}")
            tokenized.save_to_disk(str(tokenized_cache_path))
            with tokenized_cache_meta_path.open("w", encoding="utf-8") as file:
                json.dump(expected_cache_meta, file, indent=2)
                file.write("\n")

    lora_kwargs: dict[str, Any] = {
        "r": int(training_cfg["lora_rank"]),
        "lora_alpha": int(training_cfg["lora_alpha"]),
        "lora_dropout": float(training_cfg["lora_dropout"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    if structural_token_ids:
        lora_kwargs["trainable_token_indices"] = structural_token_ids
    model = get_peft_model(model, LoraConfig(**lora_kwargs))

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": int(training_cfg["max_steps"]),
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "bf16": bool(training_cfg.get("bf16", True)),
        "fp16": not bool(training_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "logging_steps": max(10, min(50, int(training_cfg["max_steps"]) // 20)),
        "eval_strategy": "no",
        "save_steps": max(1, min(int(training_cfg.get("save_steps", 250)), int(training_cfg["max_steps"]))),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "save_safetensors": True,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 0)),
    }
    for optional_key in [
        "optim",
        "gradient_checkpointing_kwargs",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "group_by_length",
        "length_column_name",
        "max_grad_norm",
        "lr_scheduler_type",
        "warmup_ratio",
        "warmup_steps",
        "torch_empty_cache_steps",
        "torch_compile",
    ]:
        if optional_key in training_cfg:
            training_args_kwargs[optional_key] = training_cfg[optional_key]
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs = {
        key: value
        for key, value in training_args_kwargs.items()
        if key in parameters
    }
    if "save_only_model" in parameters:
        training_args_kwargs["save_only_model"] = True

    class TimingCallback(TrainerCallback):
        def __init__(
            self,
            *,
            log_steps: int,
            sequence_length: int,
            batch_size: int,
            gradient_accumulation_steps: int,
            torch_module: Any,
        ) -> None:
            self.log_steps = max(1, log_steps)
            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.torch = torch_module
            self.start_time = 0.0
            self.last_time = 0.0
            self.last_step = 0
            self.records: list[dict[str, Any]] = []

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            if self.torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            self.last_time = self.start_time
            self.last_step = int(state.global_step or 0)
            print("[timing] training timer started")

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            step = int(state.global_step or 0)
            if step <= 0:
                return
            if step - self.last_step < self.log_steps and step < int(state.max_steps or step):
                return
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            now = time.perf_counter()
            interval_steps = max(1, step - self.last_step)
            interval_seconds = max(now - self.last_time, 1e-9)
            total_seconds = max(now - self.start_time, 1e-9)
            examples_per_step = self.batch_size * self.gradient_accumulation_steps
            interval_examples = interval_steps * examples_per_step
            interval_tokens = interval_examples * self.sequence_length
            avg_steps_per_second = step / total_seconds
            payload = {
                "step": f"{step}/{int(state.max_steps or 0)}",
                "interval_seconds": round(interval_seconds, 2),
                "seconds_per_step": round(interval_seconds / interval_steps, 3),
                "steps_per_second": round(interval_steps / interval_seconds, 4),
                "avg_steps_per_second": round(avg_steps_per_second, 4),
                "estimated_examples_per_second": round(interval_examples / interval_seconds, 3),
                "estimated_tokens_per_second": round(interval_tokens / interval_seconds, 1),
                "elapsed_minutes": round(total_seconds / 60, 2),
            }
            if self.torch.cuda.is_available():
                payload["memory_allocated_gb"] = round(self.torch.cuda.memory_allocated() / 1024**3, 2)
                payload["max_memory_allocated_gb"] = round(self.torch.cuda.max_memory_allocated() / 1024**3, 2)
            self.records.append(dict(payload))
            print("[timing] " + json.dumps(payload))
            self.last_time = now
            self.last_step = step

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self, *, max_wall_time_minutes: float | None) -> None:
            self.max_wall_time_seconds = (
                max_wall_time_minutes * 60.0
                if max_wall_time_minutes is not None and max_wall_time_minutes > 0
                else None
            )
            self.start_time = 0.0
            self.triggered = False
            self.triggered_step: int | None = None

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
            self.start_time = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if self.max_wall_time_seconds is None or self.triggered:
                return
            elapsed = time.perf_counter() - self.start_time
            if elapsed >= self.max_wall_time_seconds:
                self.triggered = True
                self.triggered_step = int(state.global_step or 0)
                control.should_training_stop = True
                control.should_save = True
                print(
                    "[time_budget] "
                    + json.dumps(
                        {
                            "max_wall_time_minutes": round(self.max_wall_time_seconds / 60.0, 2),
                            "elapsed_minutes": round(elapsed / 60.0, 2),
                            "step": self.triggered_step,
                        }
                    )
                )

    timing_callback = TimingCallback(
        log_steps=int(training_cfg.get("timing_log_steps", 10)),
        sequence_length=sequence_length,
        batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        torch_module=torch,
    )
    time_budget_callback = TimeBudgetCallback(
        max_wall_time_minutes=(
            float(training_cfg["max_wall_time_minutes"])
            if training_cfg.get("max_wall_time_minutes") is not None
            else None
        )
    )
    callbacks: list[TrainerCallback] = [timing_callback]
    if time_budget_callback.max_wall_time_seconds is not None:
        callbacks.append(time_budget_callback)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )
    train_started_at = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint") or None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started_at
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    result = {
        "status": "time_budget_reached" if time_budget_callback.triggered else "complete",
        "output_dir": str(output_dir),
        "train_rows": len(dataset["train"]),
        "validation_rows": len(dataset["validation"]),
        "train_runtime_seconds": round(train_seconds, 2),
        "train_runtime_minutes": round(train_seconds / 60, 2),
        "trainer_metrics": train_output.metrics,
        "time_budget": {
            "max_wall_time_minutes": training_cfg.get("max_wall_time_minutes"),
            "triggered": time_budget_callback.triggered,
            "triggered_step": time_budget_callback.triggered_step,
        },
        "cuda": cuda_summary(torch),
    }
    summary_path = write_training_summary(
        output_dir=output_dir,
        base_model=base_model,
        dataset_cfg=dataset_cfg,
        training_cfg=training_cfg,
        result=result,
        timing_records=timing_callback.records,
        corpus_log=corpus_log,
    )
    result["summary_path"] = str(summary_path)
    print(f"[summary] wrote training summary to {summary_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
