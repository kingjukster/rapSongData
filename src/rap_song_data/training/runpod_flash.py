"""Launch a QLoRA training job through Runpod Flash.

The control flow runs locally. The decorated function runs on Runpod. Provide
public or signed URLs for the JSONL train/validation files; the remote worker
downloads those files before training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
    base_model = job_config["base_model"]
    output_dir = job_config["output_dir"]
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

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if add_structural_special_tokens:
            model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
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
            "bf16": True,
            "gradient_checkpointing": True,
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
        callbacks = []
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
        notify_slack(
            slack_webhook_url,
            "Rap LoRA trainer started",
            {"resume_from": str(resume_checkpoint_path) if resume_checkpoint_path else None},
        )
        trainer.train(resume_from_checkpoint=str(resume_checkpoint_path) if resume_checkpoint_path else None)
        remove_checkpoint_dirs(Path(output_dir))
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

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
            "base_model": base_model,
            "output_dir": output_dir,
            "hf_output_repo": hf_output_repo,
            "uploaded_adapter": uploaded_adapter,
            "train_rows": len(dataset["train"]),
            "validation_rows": len(dataset["validation"]),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        notify_slack(
            slack_webhook_url,
            "Rap LoRA training completed",
            {
                "max_steps": max_steps,
                "train_rows": len(dataset["train"]),
                "validation_rows": len(dataset["validation"]),
                "adapter": uploaded_adapter or output_dir,
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
