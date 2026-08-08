from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .common import PRIVATE_RESEARCH_POLICY, command_record, hash_file, read_json, utc_now, write_json
from .modeling import ScratchModelSpec


@dataclass
class TrainingSpec:
    sequence_length: int = 512
    microbatch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.02
    target_tokens: int = 600_000_000
    max_epochs: float = 2.0
    max_hours: float = 20.0
    eval_every_steps: int = 500
    eval_batches: int = 100
    checkpoint_minutes: float = 30.0
    dataloader_workers: int = 4
    seed: int = 20260713
    gradient_checkpointing: bool = False

    @classmethod
    def from_json(cls, path: Path | None, *, mode: str) -> "TrainingSpec":
        spec = cls()
        if mode == "sft":
            spec.learning_rate = 5e-5
            spec.final_learning_rate = 5e-6
            spec.max_hours = 2.0
            spec.max_epochs = 1.0
            spec.target_tokens = 10**18
        if path is not None:
            payload = read_json(path)
            values = payload.get("training", payload)
            for field in asdict(spec):
                if field in values:
                    setattr(spec, field, values[field])
        return spec


class BinaryBlockDataset:
    def __init__(self, split_manifest: dict[str, Any], sequence_length: int):
        self.sequence_length = sequence_length
        self.paths = [Path(item["path"]) for item in split_manifest["shards"]]
        counts = [int(item["blocks"]) for item in split_manifest["shards"]]
        self.cumulative = np.cumsum(counts).tolist()
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return int(self.cumulative[-1]) if self.cumulative else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def __getitem__(self, index: int):
        import torch

        if index < 0:
            index += len(self)
        shard = bisect.bisect_right(self.cumulative, index)
        prior = self.cumulative[shard - 1] if shard else 0
        local = index - prior
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.paths[shard], mode="r", dtype=np.uint16)
        start = local * self.sequence_length
        block = np.asarray(self._maps[shard][start : start + self.sequence_length], dtype=np.int64).copy()
        return torch.from_numpy(block)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int, final_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def cuda_snapshot(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "total_vram_gb": round(props.total_memory / 1024**3, 3),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def create_model(tokenizer: Any, *, tiny: bool = False, model_spec: ScratchModelSpec | None = None):
    from transformers import LlamaForCausalLM

    if tiny:
        spec = ScratchModelSpec(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=172,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
    elif model_spec is not None:
        spec = ScratchModelSpec.from_mapping({**model_spec.to_dict(), "vocab_size": len(tokenizer)})
    else:
        spec = ScratchModelSpec(vocab_size=len(tokenizer))
    model = LlamaForCausalLM(spec.build_config())
    return model, spec


def save_checkpoint(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    state: dict[str, Any],
) -> Path:
    step = int(state["global_step"])
    destination = output_dir / f"checkpoint-{step:08d}"
    if destination.exists():
        return destination
    temporary = output_dir / f".checkpoint-{step:08d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    import torch

    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    write_json(temporary / "trainer_state.json", state)
    os.replace(temporary, destination)
    return destination


def load_resume_state(checkpoint: Path, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    import torch

    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
    return read_json(checkpoint / "trainer_state.json")


def evaluate_loss(model: Any, loader: Any, device: Any, *, max_batches: int, torch: Any) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            input_ids = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def endless(loader: Any) -> Iterator[Any]:
    while True:
        yield from loader


def markdown_summary(summary: dict[str, Any]) -> str:
    result = summary["result"]
    return "\n".join(
        [
            f"# Scratch {summary['mode'].upper()} Run Summary",
            "",
            f"- Status: {result['status']}",
            f"- Stop reason: {result['stop_reason']}",
            f"- Global step: {result['global_step']}",
            f"- Tokens processed: {result['tokens_processed']}",
            f"- Wall time: {summary['wall_seconds']:.2f} seconds",
            f"- Average tokens/sec: {result['average_tokens_per_second']:.2f}",
            f"- Peak allocated VRAM: {result['peak_allocated_vram_gb']:.3f} GB",
            f"- Best validation loss: {result['best_validation_loss']}",
            f"- Output: `{summary['output_dir']}`",
            "",
        ]
    )


def train(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The cuda extra is required for scratch training.") from exc

    started = time.monotonic()
    started_at = utc_now()
    spec = TrainingSpec.from_json(Path(args.config) if args.config else None, mode=mode)
    configured_model_spec: ScratchModelSpec | None = None
    if args.config:
        config_payload = read_json(Path(args.config))
        configured_model_spec = ScratchModelSpec.from_mapping(config_payload.get("model"))
    if args.smoke:
        spec.max_hours = min(spec.max_hours, 0.25)
        spec.eval_every_steps = min(spec.eval_every_steps, 10)
        spec.checkpoint_minutes = min(spec.checkpoint_minutes, 5)
        spec.dataloader_workers = 0
    if args.max_steps is not None:
        max_steps_override = args.max_steps
    elif args.smoke:
        max_steps_override = 20
    else:
        max_steps_override = None
    seed_everything(spec.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bf16.")

    data_dir = Path(args.data_dir)
    token_manifest = read_json(data_dir / "tokenization_manifest.json")
    if not args.smoke and not args.ignore_data_gate and not token_manifest["acceptance"]["full_pilot_allowed"]:
        raise RuntimeError("Corpus/token acceptance gates failed; full scratch training is blocked.")
    tokenizer = AutoTokenizer.from_pretrained(data_dir / "tokenizer", use_fast=True)
    split_info = token_manifest["splits"][mode]
    train_dataset = BinaryBlockDataset(split_info["train"], spec.sequence_length)
    validation_dataset = BinaryBlockDataset(split_info["validation"], spec.sequence_length)
    if not len(train_dataset):
        raise RuntimeError("No complete training blocks were produced.")
    loader = DataLoader(
        train_dataset,
        batch_size=spec.microbatch_size,
        shuffle=True,
        num_workers=spec.dataloader_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=spec.dataloader_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=spec.microbatch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    resume = Path(args.resume) if args.resume else None
    if resume:
        model = AutoModelForCausalLM.from_pretrained(resume)
        model_spec = {"resumed_model": str(resume)}
    elif mode == "sft":
        if not args.model_dir:
            raise ValueError("--model-dir is required for SFT.")
        model = AutoModelForCausalLM.from_pretrained(args.model_dir)
        model_spec = {"base_checkpoint": str(args.model_dir)}
    else:
        model, scratch_spec = create_model(tokenizer, tiny=args.tiny_model, model_spec=configured_model_spec)
        model_spec = scratch_spec.to_dict()
    model.config.use_cache = False
    if spec.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    optimizer_kwargs = {
        "lr": spec.learning_rate,
        "betas": (spec.adam_beta1, spec.adam_beta2),
        "weight_decay": spec.weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    effective_tokens = spec.microbatch_size * spec.gradient_accumulation_steps * spec.sequence_length
    available_tokens = int(split_info["train"]["written_tokens"])
    target_tokens = min(spec.target_tokens, int(available_tokens * spec.max_epochs))
    total_steps = max(1, math.ceil(target_tokens / effective_tokens))
    if max_steps_override is not None:
        total_steps = min(total_steps, max_steps_override)
    warmup_steps = int(round(total_steps * spec.warmup_ratio))
    final_ratio = spec.final_learning_rate / spec.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, warmup_steps, final_ratio),
    )
    state = {
        "global_step": 0,
        "micro_step": 0,
        "tokens_processed": 0,
        "best_validation_loss": None,
        "best_checkpoint": None,
        "first_validation_loss": None,
        "validation_history": [],
    }
    if resume:
        state.update(load_resume_state(resume, optimizer, scheduler))
    initial_tokens_processed = int(state["tokens_processed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exact_command.txt").write_text(" ".join(command_record()) + "\n", encoding="utf-8")
    write_json(output_dir / "resolved_config.json", {"training": asdict(spec), "model": model_spec})
    telemetry_path = output_dir / "run_metrics_timeseries.csv"
    telemetry_exists = telemetry_path.exists()
    telemetry_handle = telemetry_path.open("a", encoding="utf-8", newline="")
    telemetry = csv.DictWriter(
        telemetry_handle,
        fieldnames=["timestamp", "step", "loss", "learning_rate", "seconds_per_step", "tokens_per_second", "validation_loss", "peak_vram_gb"],
    )
    if not telemetry_exists:
        telemetry.writeheader()
    stop_reason = "target_steps_reached"
    status = "complete"
    first_eval: float | None = state.get("first_validation_loss")
    validation_history: list[float] = list(state.get("validation_history", []))
    last_checkpoint_time = time.monotonic()
    train_iterator = endless(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_start = time.monotonic()
    step_start = run_start
    last_loss = float("nan")
    try:
        while state["global_step"] < total_steps:
            if (time.monotonic() - run_start) / 3600 >= spec.max_hours:
                stop_reason = "wall_time_limit"
                break
            accumulated_loss = 0.0
            for _ in range(spec.gradient_accumulation_steps):
                batch = next(train_iterator).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(input_ids=batch, labels=batch, use_cache=False)
                    loss = output.loss / spec.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += float(loss.detach().cpu())
                state["micro_step"] += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["tokens_processed"] += effective_tokens
            last_loss = accumulated_loss
            now = time.monotonic()
            seconds = now - step_start
            step_start = now
            validation_loss: float | None = None
            if state["global_step"] % spec.eval_every_steps == 0 or state["global_step"] == total_steps:
                validation_loss = evaluate_loss(
                    model, validation_loader, device, max_batches=spec.eval_batches, torch=torch
                )
                validation_history.append(validation_loss)
                if first_eval is None:
                    first_eval = validation_loss
                state["first_validation_loss"] = first_eval
                state["validation_history"] = validation_history
                if state["best_validation_loss"] is None or validation_loss < state["best_validation_loss"]:
                    state["best_validation_loss"] = validation_loss
                    state["best_checkpoint"] = str(
                        output_dir / f"checkpoint-{state['global_step']:08d}"
                    )
                    checkpoint = save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, state)
                    state["best_checkpoint"] = str(checkpoint)
            peak_vram = (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            )
            telemetry.writerow(
                {
                    "timestamp": utc_now(),
                    "step": state["global_step"],
                    "loss": round(last_loss, 8),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds_per_step": round(seconds, 6),
                    "tokens_per_second": round(effective_tokens / max(seconds, 1e-9), 3),
                    "validation_loss": "" if validation_loss is None else round(validation_loss, 8),
                    "peak_vram_gb": round(peak_vram, 6),
                }
            )
            telemetry_handle.flush()
            if (now - last_checkpoint_time) / 60 >= spec.checkpoint_minutes:
                save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, state)
                last_checkpoint_time = now
        final_checkpoint = save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, state)
    except torch.cuda.OutOfMemoryError:
        status = "failed_oom"
        stop_reason = "cuda_out_of_memory"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise
    finally:
        telemetry_handle.close()
    wall_seconds = time.monotonic() - started
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "global_step": state["global_step"],
        "tokens_processed": state["tokens_processed"],
        "run_tokens_processed": state["tokens_processed"] - initial_tokens_processed,
        "average_tokens_per_second": round(
            (state["tokens_processed"] - initial_tokens_processed) / max(time.monotonic() - run_start, 1e-9), 3
        ),
        "peak_allocated_vram_gb": round(peak_vram, 6),
        "first_validation_loss": first_eval,
        "final_validation_loss": validation_history[-1] if validation_history else None,
        "validation_history": validation_history,
        "best_validation_loss": state["best_validation_loss"],
        "best_checkpoint": state["best_checkpoint"],
        "final_checkpoint": str(final_checkpoint),
        "last_train_loss": last_loss,
        "parameter_count": parameter_count,
        "vram_fallback_recommended": peak_vram > 10.5,
    }
    summary = {
        **PRIVATE_RESEARCH_POLICY,
        "mode": mode,
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_seconds": round(wall_seconds, 3),
        "command": command_record(),
        "output_dir": str(output_dir),
        "data_manifest_sha256": hash_file(data_dir / "tokenization_manifest.json"),
        "training": asdict(spec),
        "model": model_spec,
        "cuda": cuda_snapshot(torch),
        "result": result,
    }
    write_json(output_dir / "run_summary.json", summary)
    (output_dir / "run_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    return summary


def add_training_arguments(parser: argparse.ArgumentParser, *, mode: str) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model-dir", type=Path, help="Pretrained scratch checkpoint used for SFT.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--ignore-data-gate", action="store_true")
