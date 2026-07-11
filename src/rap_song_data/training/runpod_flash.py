"""Launch a QLoRA training job through Runpod Flash.

The control flow runs locally. The decorated function runs on Runpod. Provide
public or signed URLs for the JSONL train/validation files; the remote worker
downloads those files before training.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    from runpod_flash import DataCenter, Endpoint, GpuType, NetworkVolume
except ImportError as exc:  # pragma: no cover - only hit before setup
    raise SystemExit(
        "runpod-flash is not installed. Run: pip install runpod-flash"
    ) from exc

from rap_song_data.integrations.slack import notify_slack, slack_webhook_from_env


DEFAULT_CONFIG_PATH = Path("configs/training/runpod_flash_config.example.json")
load_dotenv()
STRUCTURAL_SPECIAL_TOKENS = ["<|verse_start|>", "<|verse_end|>", "<|bar_start|>"]
_ALLOWED_BASE_MODELS = {
    "qwen/qwen2.5-7b",
    "qwen/qwen2.5-7b-instruct",
}


def validate_qwen25_7b_only(model_name: str, *, scope: str) -> str:
    normalized = (model_name or "").split("@", 1)[0].strip().lower()
    if normalized not in _ALLOWED_BASE_MODELS:
        raise ValueError(
            f"{scope} policy requires Qwen2.5-7B. "
            f"Use Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen2.5-7B. Received: {model_name!r}"
        )
    return normalized


def configure_runtime(torch: Any) -> dict[str, object]:
    try:
        torch.set_float32_matmul_precision("high")
        precision = "high"
    except Exception:
        precision = "unavailable"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    return {
        "float32_matmul_precision": precision,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Runpod Flash config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


ENDPOINT_CONFIG = load_config(DEFAULT_CONFIG_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-url", default=None)
    parser.add_argument("--validation-url", default=None)
    parser.add_argument("--hf-dataset-repo", default=None)
    parser.add_argument("--hf-train-path", default="train.jsonl")
    parser.add_argument("--hf-validation-path", default="validation.jsonl")
    parser.add_argument("--hf-output-repo", default=None)
    parser.add_argument("--hf-checkpoint-repo", default=None)
    parser.add_argument("--checkpoint-upload-steps", type=int, default=None)
    parser.add_argument("--resume-from-hf-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--slack-webhook-url", default=None)
    parser.add_argument("--slack-progress-steps", type=int, default=None)
    return parser.parse_args()


def gpu_from_name(name: str):
    try:
        return getattr(GpuType, name)
    except AttributeError as exc:
        valid = [item for item in dir(GpuType) if item.startswith("NVIDIA_")]
        raise ValueError(f"Unknown GPU type {name!r}. Valid examples: {valid[:12]}") from exc


def datacenter_from_name(name: str):
    if not name:
        return None
    try:
        return getattr(DataCenter, name)
    except AttributeError as exc:
        valid = [item for item in dir(DataCenter) if item.isupper()]
        raise ValueError(f"Unknown datacenter {name!r}. Valid examples: {valid[:12]}") from exc


def hf_token_for_remote() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token() or ""
    except ImportError:
        return ""


def ensure_remote_packages() -> None:
    """Install training packages inside the remote worker if Flash did not bundle them."""
    packages = [
        "datasets",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "requests",
    ]
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


TRAIN_VOLUME = (
    NetworkVolume(
        name=ENDPOINT_CONFIG["volume_name"],
        size=int(ENDPOINT_CONFIG["volume_size_gb"]),
        datacenter=datacenter_from_name(ENDPOINT_CONFIG["datacenter"]),
    )
    if ENDPOINT_CONFIG.get("use_network_volume", False)
    else None
)


@Endpoint(
    name=ENDPOINT_CONFIG["endpoint_name"],
    gpu=gpu_from_name(ENDPOINT_CONFIG["gpu"]),
    workers=tuple(ENDPOINT_CONFIG["workers"]),
    datacenter=datacenter_from_name(ENDPOINT_CONFIG.get("datacenter")),
    volume=TRAIN_VOLUME,
    dependencies=[],
    execution_timeout_ms=int(ENDPOINT_CONFIG["execution_timeout_ms"]),
)
async def train_qlora(job_config: dict[str, Any]) -> dict[str, Any]:
    """Train a LoRA adapter from remote JSONL dataset URLs."""
    ensure_remote_packages()

    import os
    import shutil
    import inspect
    from pathlib import Path

    import requests
    import torch
    from datasets import load_dataset
    from huggingface_hub import create_repo, hf_hub_download, snapshot_download, upload_folder
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        TrainerCallback,
        Trainer,
        TrainingArguments,
    )

    slack_webhook_url = job_config.get("slack_webhook_url") or os.getenv("SLACK_WEBHOOK_URL") or ""
    run_command = ["runpod-flash", "train_qlora"]
    run_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    command_hash = hashlib.md5(json.dumps(job_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    runtime_cfg = configure_runtime(torch)

    dataset_cfg = job_config["dataset"]
    train_url = dataset_cfg.get("train_url")
    validation_url = dataset_cfg.get("validation_url")
    hf_dataset_repo = dataset_cfg.get("hf_dataset_repo")
    if not hf_dataset_repo and not train_url:
        raise ValueError("Missing dataset source. Pass --hf-dataset-repo or --train-url.")
    if not hf_dataset_repo and not validation_url:
        raise ValueError("Missing validation dataset source. Pass --hf-dataset-repo or --validation-url.")

    work_dir = Path("/runpod-volume/rap-lyrics") if Path("/runpod-volume").exists() else Path("/tmp/rap-lyrics")
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"

    def download_url(url: str, path: Path) -> None:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    training_cfg = job_config["training"]
    base_model = validate_qwen25_7b_only(job_config["base_model"], scope="Runpod Flash training")
    output_dir = job_config["output_dir"]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    text_field = dataset_cfg.get("text_field", "training_text")
    max_steps = int(training_cfg["max_steps"])
    hf_output_repo = job_config.get("hf_output_repo")
    hf_checkpoint_repo = job_config.get("hf_checkpoint_repo") or hf_output_repo
    resume_from_hf_checkpoint = job_config.get("resume_from_hf_checkpoint") or ""
    checkpoint_upload_steps = int(job_config.get("checkpoint_upload_steps") or 250)
    add_structural_special_tokens = bool(job_config.get("add_structural_special_tokens", False))

    def disk_report(path: Path) -> dict[str, str]:
        usage = shutil.disk_usage(path)
        gb = 1024**3
        return {
            "path": str(path),
            "free_gb": f"{usage.free / gb:.1f}",
            "used_gb": f"{usage.used / gb:.1f}",
            "total_gb": f"{usage.total / gb:.1f}",
        }

    def remove_checkpoint_dirs(output_path: Path, keep: set[str] | None = None) -> int:
        keep = keep or set()
        removed = 0
        for checkpoint_path in output_path.glob("checkpoint-*"):
            if checkpoint_path.name in keep:
                continue
            if checkpoint_path.is_dir():
                shutil.rmtree(checkpoint_path, ignore_errors=True)
                removed += 1
        return removed

    notify_slack(
        slack_webhook_url,
        "Rap LoRA training started",
        {
            "base_model": base_model,
            "dataset": hf_dataset_repo or "url",
            "max_steps": max_steps,
            "output_repo": job_config.get("hf_output_repo"),
            "checkpoint_repo": hf_checkpoint_repo,
            "resume_from": resume_from_hf_checkpoint or None,
            "disk": disk_report(work_dir),
        },
    )

    try:
        token = (
            dataset_cfg.get("hf_token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or None
        )
        if hf_output_repo:
            create_repo(
                repo_id=hf_output_repo,
                repo_type="model",
                private=True,
                exist_ok=True,
                token=token,
            )
        if hf_checkpoint_repo and hf_checkpoint_repo != hf_output_repo:
            create_repo(
                repo_id=hf_checkpoint_repo,
                repo_type="model",
                private=True,
                exist_ok=True,
                token=token,
            )

        resume_checkpoint_path = None
        if resume_from_hf_checkpoint:
            if not hf_checkpoint_repo:
                raise ValueError(
                    "Missing checkpoint repo. Pass --hf-output-repo or --hf-checkpoint-repo "
                    "when using --resume-from-hf-checkpoint."
                )
            resume_dir = work_dir / "resume"
            notify_slack(
                slack_webhook_url,
                "Rap LoRA checkpoint download started",
                {"checkpoint": resume_from_hf_checkpoint, "repo": hf_checkpoint_repo},
            )
            snapshot_download(
                repo_id=hf_checkpoint_repo,
                repo_type="model",
                allow_patterns=[f"{resume_from_hf_checkpoint}/**"],
                local_dir=str(resume_dir),
                token=token,
            )
            resume_checkpoint_path = resume_dir / resume_from_hf_checkpoint
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint {resume_from_hf_checkpoint!r} was not found in {hf_checkpoint_repo!r}."
                )

        if hf_dataset_repo:
            downloaded_train = hf_hub_download(
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                filename=dataset_cfg.get("hf_train_path", "train.jsonl"),
                token=token,
                local_dir=str(data_dir),
            )
            downloaded_validation = hf_hub_download(
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                filename=dataset_cfg.get("hf_validation_path", "validation.jsonl"),
                token=token,
                local_dir=str(data_dir),
            )
            train_path = Path(downloaded_train)
            validation_path = Path(downloaded_validation)
        else:
            download_url(train_url, train_path)
            download_url(validation_url, validation_path)
        notify_slack(
            slack_webhook_url,
            "Rap LoRA dataset ready",
            {"train_path": str(train_path), "validation_path": str(validation_path)},
        )

        load_started_at = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        structural_token_ids: list[int] = []
        if add_structural_special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS})
            structural_token_ids = tokenizer.convert_tokens_to_ids(STRUCTURAL_SPECIAL_TOKENS)
            if any(token_id is None or token_id < 0 for token_id in structural_token_ids):
                raise ValueError(f"Failed to add structural special tokens: {STRUCTURAL_SPECIAL_TOKENS}")
        notify_slack(
            slack_webhook_url,
            "Rap LoRA tokenizer loaded",
            {
                "base_model": base_model,
                "add_structural_special_tokens": add_structural_special_tokens,
                "structural_markers": STRUCTURAL_SPECIAL_TOKENS,
            },
        )

        load_in_4bit = bool(training_cfg.get("load_in_4bit", True))
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": compute_dtype,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started_at
        if add_structural_special_tokens:
            model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        checkpoint_kwargs = training_cfg.get("gradient_checkpointing_kwargs")
        if isinstance(checkpoint_kwargs, dict):
            checkpoint_kwargs = dict(checkpoint_kwargs)
        else:
            checkpoint_kwargs = {}
        checkpoint_kwargs.setdefault("use_reentrant", False)
        if bool(training_cfg.get("gradient_checkpointing", True)):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            except TypeError:
                print("[runtime] WARNING: non-reentrant checkpointing unsupported; using default checkpointing.")
                model.gradient_checkpointing_enable()
        else:
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
        notify_slack(
            slack_webhook_url,
            "Rap LoRA base model loaded",
            {"cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        )

        dataset = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(validation_path)},
        )

        lora_kwargs = {
            "r": int(training_cfg["lora_rank"]),
            "lora_alpha": int(training_cfg["lora_alpha"]),
            "lora_dropout": float(training_cfg["lora_dropout"]),
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        if structural_token_ids:
            lora_kwargs["trainable_token_indices"] = structural_token_ids
        lora = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora)

        sequence_length = int(training_cfg["sequence_length"])

        def tokenize_batch(batch):
            return tokenizer(
                batch[text_field],
                truncation=True,
                max_length=sequence_length,
                padding=False,
            )

        remove_columns = dataset["train"].column_names
        tokenize_num_proc = int(training_cfg.get("tokenize_num_proc") or max(1, min(4, os.cpu_count() or 1)))
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing training text",
            num_proc=tokenize_num_proc,
        )
        notify_slack(
            slack_webhook_url,
            "Rap LoRA tokenization complete",
            {"train_rows": len(dataset["train"]), "validation_rows": len(dataset["validation"])},
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args_kwargs = {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "num_train_epochs": float(training_cfg["num_train_epochs"]),
            "learning_rate": float(training_cfg["learning_rate"]),
            "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
            "bf16": bool(training_cfg.get("bf16", True)),
            "fp16": not bool(training_cfg.get("bf16", True)),
            "optim": str(training_cfg.get("optim", "paged_adamw_8bit")),
            "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
            "logging_steps": max(10, min(50, max_steps // 20)),
            "eval_strategy": "no",
            "save_steps": max(1, min(checkpoint_upload_steps, max_steps)),
            "save_total_limit": 1,
            "save_safetensors": True,
            "report_to": [],
            "remove_unused_columns": False,
            "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 2)),
        }
        training_args_parameters = inspect.signature(TrainingArguments.__init__).parameters
        if "save_only_model" in training_args_parameters:
            training_args_kwargs["save_only_model"] = bool(job_config.get("save_only_model_checkpoints", True))
        args = TrainingArguments(**training_args_kwargs)

        class TrainingTimingCallback(TrainerCallback):
            def __init__(
                self,
                *,
                log_steps: int,
                batch_size: int,
                gradient_accumulation_steps: int,
                torch_module: Any,
            ) -> None:
                self.log_steps = max(1, log_steps)
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
                interval_examples = interval_steps * self.batch_size * self.gradient_accumulation_steps
                payload = {
                    "step": f"{step}/{int(state.max_steps or 0)}",
                    "interval_seconds": round(interval_seconds, 3),
                    "seconds_per_step": round(interval_seconds / interval_steps, 4),
                    "steps_per_second": round(interval_steps / interval_seconds, 4),
                    "avg_steps_per_second": round(step / total_seconds, 4),
                    "estimated_examples_per_second": round(interval_examples / interval_seconds, 3),
                    "elapsed_minutes": round(total_seconds / 60, 2),
                }
                if self.torch.cuda.is_available():
                    payload["memory_allocated_gb"] = round(self.torch.cuda.memory_allocated() / 1024**3, 2)
                    payload["max_memory_allocated_gb"] = round(self.torch.cuda.max_memory_allocated() / 1024**3, 2)
                self.records.append(dict(payload))
                self.last_time = now
                self.last_step = step

        class SlackProgressCallback(TrainerCallback):
            def __init__(self, webhook_url: str, progress_steps: int, total_steps: int) -> None:
                self.webhook_url = webhook_url
                self.progress_steps = max(1, progress_steps)
                self.total_steps = total_steps
                self.last_reported_step = 0

            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
                if not self.webhook_url or state.global_step <= 0:
                    return
                if state.global_step - self.last_reported_step < self.progress_steps and state.global_step < self.total_steps:
                    return
                self.last_reported_step = state.global_step
                notify_slack(
                    self.webhook_url,
                    "Rap LoRA training progress",
                    {
                        "step": f"{state.global_step}/{self.total_steps}",
                        "loss": (logs or {}).get("loss"),
                        "learning_rate": (logs or {}).get("learning_rate"),
                    },
                )

        class HuggingFaceCheckpointCallback(TrainerCallback):
            def __init__(
                self,
                checkpoint_repo: str | None,
                upload_steps: int,
                auth_token: str | None,
                webhook_url: str,
            ) -> None:
                self.checkpoint_repo = checkpoint_repo
                self.upload_steps = max(1, upload_steps)
                self.auth_token = auth_token
                self.webhook_url = webhook_url
                self.last_uploaded_step = 0

            def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
                if not self.checkpoint_repo or state.global_step <= 0:
                    return
                if state.global_step - self.last_uploaded_step < self.upload_steps and state.global_step < max_steps:
                    return

                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if not checkpoint_dir.exists():
                    return

                notify_slack(
                    self.webhook_url,
                    "Rap LoRA checkpoint upload started",
                    {
                        "checkpoint": checkpoint_dir.name,
                        "repo": self.checkpoint_repo,
                        "disk": disk_report(Path(args.output_dir)),
                    },
                )
                try:
                    upload_folder(
                        repo_id=self.checkpoint_repo,
                        repo_type="model",
                        folder_path=str(checkpoint_dir),
                        path_in_repo=checkpoint_dir.name,
                        token=self.auth_token,
                        commit_message=f"Upload training checkpoint {checkpoint_dir.name}",
                    )
                    self.last_uploaded_step = state.global_step
                    removed = remove_checkpoint_dirs(Path(args.output_dir))
                    notify_slack(
                        self.webhook_url,
                        "Rap LoRA checkpoint uploaded",
                        {
                            "checkpoint": checkpoint_dir.name,
                            "repo": self.checkpoint_repo,
                            "removed_local_checkpoints": removed,
                            "disk": disk_report(Path(args.output_dir)),
                        },
                    )
                except Exception as exc:  # pragma: no cover - remote-only recovery path
                    notify_slack(
                        self.webhook_url,
                        "Rap LoRA checkpoint upload failed",
                        {"checkpoint": checkpoint_dir.name, "error": str(exc)[:500]},
                    )

        progress_steps = int(job_config.get("slack_progress_steps") or max(50, max_steps // 10))
        callbacks = [
            TrainingTimingCallback(
                log_steps=int(training_cfg.get("timing_log_steps", 10)),
                batch_size=int(training_cfg["per_device_train_batch_size"]),
                gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
                torch_module=torch,
            )
        ]
        if slack_webhook_url:
            callbacks.append(SlackProgressCallback(slack_webhook_url, progress_steps, max_steps))
        if hf_checkpoint_repo:
            callbacks.append(
                HuggingFaceCheckpointCallback(
                    hf_checkpoint_repo,
                    checkpoint_upload_steps,
                    token,
                    slack_webhook_url,
                )
            )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=collator,
            callbacks=callbacks,
        )
        train_started = time.perf_counter()
        notify_slack(
            slack_webhook_url,
            "Rap LoRA trainer started",
            {"resume_from": str(resume_checkpoint_path) if resume_checkpoint_path else None},
        )
        train_output = trainer.train(resume_from_checkpoint=str(resume_checkpoint_path) if resume_checkpoint_path else None)
        train_seconds = time.perf_counter() - train_started
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        remove_checkpoint_dirs(Path(output_dir))
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        else:
            peak_vram_gb = None
        timing_records = callbacks[0].records if callbacks else []
        avg_seconds_per_step = (
            round(sum(item["seconds_per_step"] for item in timing_records if "seconds_per_step" in item) / len(timing_records), 4)
            if timing_records
            else None
        )
        run_ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        run_wall_seconds = round(time.perf_counter() - run_started, 2)
        run_summary = {
            "command": run_command,
            "command_hash": command_hash,
            "started_at": run_started_at,
            "ended_at": run_ended_at,
            "wall_seconds": run_wall_seconds,
            "base_model": base_model,
            "output_dir": str(output_dir),
            "dataset": {"train_path": str(train_path), "validation_path": str(validation_path)},
            "training_cfg": training_cfg,
            "runtime": runtime_cfg,
            "environment": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "timing": {
                "model_load_seconds": round(load_seconds, 2),
                "train_seconds": round(train_seconds, 2),
                "avg_seconds_per_step": avg_seconds_per_step,
                "train_rows": len(dataset["train"]),
                "validation_rows": len(dataset["validation"]),
                "peak_vram_gb": peak_vram_gb,
                "metrics": train_output.metrics if hasattr(train_output, "metrics") else {},
            },
            "timing_records": timing_records,
        }
        run_summary_json = output_dir_path / "run_summary.json"
        run_summary_md = output_dir_path / "run_summary.md"
        run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        run_summary_md.write_text(
            "\n".join(
                [
                    "# Runpod Flash QLoRA Run Summary",
                    "",
                    f"- Command hash: {command_hash}",
                    f"- Command: {' '.join(run_command)}",
                    f"- Started: {run_started_at}",
                    f"- Ended: {run_ended_at}",
                    f"- Wall seconds: {run_wall_seconds}",
                    f"- Train seconds: {run_summary['timing']['train_seconds']}",
                    f"- Avg seconds/step: {avg_seconds_per_step}",
                    f"- Peak VRAM (GB): {peak_vram_gb}",
                ]
            ),
            encoding="utf-8",
        )

        uploaded_adapter = None
        if hf_output_repo:
            commit = upload_folder(
                repo_id=hf_output_repo,
                repo_type="model",
                folder_path=output_dir,
                token=token,
                commit_message=f"Upload rap lyrics LoRA adapter ({max_steps} steps)",
            )
            uploaded_adapter = str(commit.commit_url)

        result = {
            "status": "complete",
            "command": run_command,
            "command_hash": command_hash,
            "base_model": base_model,
            "output_dir": output_dir,
            "hf_output_repo": hf_output_repo,
            "uploaded_adapter": uploaded_adapter,
            "train_rows": len(dataset["train"]),
            "validation_rows": len(dataset["validation"]),
            "run_summary_json": str(run_summary_json),
            "run_summary_md": str(run_summary_md),
            "run_started_at": run_started_at,
            "run_ended_at": run_ended_at,
            "run_wall_seconds": run_wall_seconds,
            "train_seconds": round(train_seconds, 2),
            "model_load_seconds": round(load_seconds, 2),
            "avg_seconds_per_step": avg_seconds_per_step,
            "peak_vram_gb": peak_vram_gb,
            "environment": run_summary["environment"],
            "runtime": runtime_cfg,
            "timing_records": timing_records,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        notify_slack(
            slack_webhook_url,
            "Rap LoRA training completed",
            {
                "max_steps": max_steps,
                "run_wall_seconds": run_wall_seconds,
                "avg_seconds_per_step": avg_seconds_per_step,
                "peak_vram_gb": peak_vram_gb,
                "train_rows": len(dataset["train"]),
                "validation_rows": len(dataset["validation"]),
                "adapter": uploaded_adapter or output_dir,
                "run_summary_json": str(run_summary_json),
            },
        )
        return result
    except Exception as exc:
        notify_slack(
            slack_webhook_url,
            "Rap LoRA training failed",
            {
                "max_steps": max_steps,
                "error": str(exc)[:500],
            },
        )
        raise
async def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_url is not None:
        config["dataset"]["train_url"] = args.train_url
    if args.validation_url is not None:
        config["dataset"]["validation_url"] = args.validation_url
    if args.hf_dataset_repo is not None:
        config["dataset"]["hf_dataset_repo"] = args.hf_dataset_repo
        config["dataset"]["hf_train_path"] = args.hf_train_path
        config["dataset"]["hf_validation_path"] = args.hf_validation_path
        config["dataset"]["hf_token"] = hf_token_for_remote()
    if args.hf_output_repo is not None:
        config["hf_output_repo"] = args.hf_output_repo
    if args.hf_checkpoint_repo is not None:
        config["hf_checkpoint_repo"] = args.hf_checkpoint_repo
    elif args.hf_output_repo is not None:
        config["hf_checkpoint_repo"] = args.hf_output_repo
    if args.checkpoint_upload_steps is not None:
        config["checkpoint_upload_steps"] = args.checkpoint_upload_steps
    if args.resume_from_hf_checkpoint is not None:
        config["resume_from_hf_checkpoint"] = args.resume_from_hf_checkpoint
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    config["slack_webhook_url"] = args.slack_webhook_url or slack_webhook_from_env()
    if args.slack_progress_steps is not None:
        config["slack_progress_steps"] = args.slack_progress_steps

    result = await train_qlora(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
