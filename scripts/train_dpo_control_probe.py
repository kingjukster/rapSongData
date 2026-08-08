from __future__ import annotations

import argparse
import json
import os
import time
import gc
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative DPO probe trainer for rap adapter.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base HF model to load.",
    )
    parser.add_argument(
        "--base-adapter-path",
        default="model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-384-500-simple/checkpoint-500",
        help="Existing LoRA adapter (mixed SFT checkpoint-500).",
    )
    parser.add_argument(
        "--pair-path",
        default="data/preferences/rap_dpo_control_pairs.jsonl",
        help="DPO JSONL with prompt/chosen/rejected.",
    )
    parser.add_argument(
        "--output-dir",
        default="model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-500-dpo-control-probe",
        help="DPO output directory.",
    )
    parser.add_argument("--run-name", default="qwen2.5-7b-rap-lora-mixed-sft-500-dpo-control-probe")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--run-summary", default=None)
    parser.add_argument(
        "--disable-ref-model",
        action="store_true",
        help="Use a single-model DPO path to keep memory under 12 GB.",
    )
    return parser.parse_args()


def _read_pairs(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    required = {"prompt", "chosen", "rejected"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"Invalid row in {path}: missing required fields")
    return rows


def _load_model_and_tokenizer(base_model: str, adapter_path: str | None, *, trainable: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        try:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
        except TypeError:
            model = PeftModel.from_pretrained(model, adapter_path)
            if trainable:
                for _, param in model.named_parameters():
                    if "lora" in _.lower():
                        param.requires_grad_(True)
            else:
                for _, param in model.named_parameters():
                    param.requires_grad_(False)

    return model, tokenizer


def _pair_stats(pairs):
    return {
        "pairs": len(pairs),
        "avg_prompt_len": sum(len(r["prompt"]) for r in pairs) / max(len(pairs), 1),
        "avg_chosen_len": sum(len(r["chosen"]) for r in pairs) / max(len(pairs), 1),
        "avg_rejected_len": sum(len(r["rejected"]) for r in pairs) / max(len(pairs), 1),
        "source_counts": {
            source: len([r for r in pairs if r.get("source") == source]) for source in set(r.get("source", "unknown") for r in pairs)
        },
    }


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = Path(args.run_summary or output_dir / "run_summary.json")

    pairs = _read_pairs(Path(args.pair_path))
    if not pairs:
        raise RuntimeError(f"No DPO pairs found in {args.pair_path}")

    dataset = Dataset.from_list(pairs)
    gc.collect()
    model, tokenizer = _load_model_and_tokenizer(
        args.base_model,
        args.base_adapter_path,
        trainable=True,
    )
    ref_model = None
    if not args.disable_ref_model:
        try:
            ref_model, _ = _load_model_and_tokenizer(
                args.base_model,
                args.base_adapter_path,
                trainable=False,
            )
        except Exception as exc:
            print(f"Ref model load failed; falling back to single-model mode: {exc}")
            ref_model = None
            torch.cuda.empty_cache()
            gc.collect()

    trainer_cfg = DPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
    )

    train_kwargs = dict(
        model=model,
        args=trainer_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if ref_model is not None:
        train_kwargs["ref_model"] = ref_model
    try:
        trainer = DPOTrainer(**train_kwargs)
    except TypeError:
        if "processing_class" in train_kwargs:
            train_kwargs["tokenizer"] = train_kwargs.pop("processing_class")
            trainer = DPOTrainer(**train_kwargs)
        else:
            raise

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    train_start = time.time()
    train_result = trainer.train()
    train_end = time.time()

    trainer.save_state()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    torch.cuda.empty_cache()

    summary = {
        "script": str(Path(__file__).name),
        "command": " ".join([arg for arg in os.sys.argv]),
        "base_model": args.base_model,
        "base_adapter_path": args.base_adapter_path,
        "pair_path": args.pair_path,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(train_start)),
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(train_end)),
        "wall_sec": round(train_end - train_start, 3),
        "train_metrics": train_result.metrics if hasattr(train_result, "metrics") else {},
        "trainer_config": {k: _json_safe(getattr(trainer_cfg, k)) for k in vars(trainer_cfg)},
        "pair_stats": _pair_stats(pairs),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3)
        if torch.cuda.is_available()
        else None,
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved(0) / (1024**3), 3)
        if torch.cuda.is_available()
        else None,
    }

    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved DPO adapter to: {output_dir / 'adapter'}")
    print(f"Run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
