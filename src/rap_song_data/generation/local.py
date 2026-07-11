"""Generate lyrics locally with the saved LoRA adapter.

This loads the base model plus the adapter saved under
`model/artifacts/qwen2.5-7b-rap-lora-local`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience for local auth.
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = Path("model/artifacts/qwen2.5-7b-rap-lora-local")
VERSE_START_TOKEN = "<|verse_start|>"
VERSE_END_TOKEN = "<|verse_end|>"
BAR_START_TOKEN = "<|bar_start|>"
STRUCTURAL_SPECIAL_TOKENS = [VERSE_START_TOKEN, VERSE_END_TOKEN, BAR_START_TOKEN]
ARTIFACT_PATTERNS = [
    r"\[?\s*lyrics taken from\b.*",
    r"\[?\s*lyrics from\b.*",
    r"\[?\s*source:\b.*",
    r"\[?\s*embed\b.*",
    r"\[?\s*you might also like\b.*",
    r"https?://\S+.*",
    r"\bgenius\.com\b.*",
]
BLOCKED_PHRASES = [
    "Lyrics taken from",
    "lyrics taken from",
    "Genius",
    "genius.com",
    "https://",
    "http://",
    "You might also like",
    "Embed",
]


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


def configure_torch_runtime(torch) -> dict[str, object]:
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--task", choices=["generate_song", "generate_verse"], default="generate_verse")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-structural-special-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--artist", default="Original")
    parser.add_argument("--rap-family", default="Melodic / Emo / Cloud")
    parser.add_argument("--rap-category", default="Emo Rap / Melodic Rap")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--views-log", default="0.0000")
    parser.add_argument("--structure", default="Exactly 12 lines. Each line should be one short rap bar. No chorus. No bracket labels. No explanations.")
    parser.add_argument("--theme", default="working late, loneliness, ambition, city lights")
    parser.add_argument("--keywords", default="night, shift, city, lights, work, ambition")
    parser.add_argument("--rules", default="Write only original lyrics. Keep line breaks. Do not explain. Stay focused on the requested theme and keywords.")
    parser.add_argument("--target-bars", type=int, default=12)
    parser.add_argument("--bar-count-range", default="10-25")
    parser.add_argument("--max-words-per-bar", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.22)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-summary-dir",
        type=Path,
        default=None,
        help="Directory to write run_summary.json and run_summary.md.",
    )
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
    if args.task == "generate_verse":
        return (
            "<|task|>generate_verse\n"
            f"<|title|>{args.title}\n"
            f"<|artist|>{args.artist}\n"
            f"<|rap_family|>{args.rap_family}\n"
            f"<|rap_category|>{args.rap_category}\n"
            f"<|year|>{args.year}\n"
            f"<|views_log|>{args.views_log}\n"
            f"<|structure|>{args.structure}\n"
            f"<|target_bars|>{args.target_bars}\n"
            f"<|bar_count_range|>{args.bar_count_range}\n"
            f"<|max_words_per_bar|>{args.max_words_per_bar}\n"
            f"<|theme|>{args.theme}\n"
            f"<|keywords|>{args.keywords}\n"
            f"<|rules|>{args.rules}\n"
            "<|lyrics|>\n"
            f"{VERSE_START_TOKEN}\n"
            f"{BAR_START_TOKEN}"
        )
    return (
        "<|task|>generate_song\n"
        f"<|title|>{args.title}\n"
        f"<|artist|>{args.artist}\n"
        f"<|rap_family|>{args.rap_family}\n"
        f"<|rap_category|>{args.rap_category}\n"
        f"<|year|>{args.year}\n"
        f"<|views_log|>{args.views_log}\n"
        "<|lyrics|>\n"
    )


def blocked_phrase_ids(tokenizer) -> list[list[int]]:
    blocked = []
    for phrase in BLOCKED_PHRASES:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            blocked.append(ids)
    return blocked


def clean_generated_lyrics(text: str) -> str:
    if "<|lyrics|>" in text:
        text = text.split("<|lyrics|>", 1)[1]
    if VERSE_END_TOKEN in text:
        text = text.split(VERSE_END_TOKEN, 1)[0]
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]

    text = text.replace(VERSE_START_TOKEN, "")
    text = text.replace(VERSE_END_TOKEN, "")
    text = text.replace(BAR_START_TOKEN, "\n")
    text = re.sub(r"<\|\s*bar_?start\s*\|?\s*[=:\-]*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|\s*verse_?start\s*\|?\s*[=:\-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|\s*verse_?end[^>\n]{0,80}>?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|.*?\|>", "", text)

    cleaned_lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in ARTIFACT_PATTERNS):
            break
        cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def main() -> None:
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    args = parse_args()
    validate_qwen25_7b_only(args.base_model, scope="Generation run")
    command = [sys.executable, *sys.argv]
    command_hash = hashlib.md5(" ".join(str(item) for item in command).encode("utf-8")).hexdigest()
    run_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    runtime = configure_torch_runtime(torch)
    load_started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    device_map = "auto" if use_cuda else None

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if args.add_structural_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": device_map, "low_cpu_mem_usage": True}
    if use_cuda and args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.add_structural_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = True
    model.eval()
    if not use_cuda:
        model.to("cpu")
    if use_cuda:
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started_at

    prompt = build_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt")
    if use_cuda:
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    generation_started_at = time.perf_counter()
    generation_start_memory_bytes = torch.cuda.max_memory_allocated() if use_cuda else 0
    with torch.inference_mode():
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "bad_words_ids": blocked_phrase_ids(tokenizer),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.min_new_tokens > 0:
            generate_kwargs["min_new_tokens"] = args.min_new_tokens
        output_ids = model.generate(**inputs, **generate_kwargs)
    if use_cuda:
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started_at

    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    generated_tokens = int(output_ids[0].shape[-1] - inputs["input_ids"].shape[-1])
    if args.clean_output:
        text = clean_generated_lyrics(text)
    timing = {
        "runtime": runtime,
        "model_load_seconds": round(load_seconds, 2),
        "generation_seconds": round(generation_seconds, 2),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / max(generation_seconds, 1e-9), 2),
    }
    if use_cuda:
        timing["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    run_ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_wall_seconds = round(time.perf_counter() - run_started, 2)
    run_summary_dir = args.run_summary_dir or args.adapter_dir
    run_summary_dir.mkdir(parents=True, exist_ok=True)
    run_summary_json = run_summary_dir / "run_summary.json"
    run_summary_md = run_summary_dir / "run_summary.md"
    run_result_json = run_summary_dir / "generation_result.json"
    run_result_txt = run_summary_dir / "generation_result.txt"
    run_log_path = run_summary_dir / "generation.log"
    generated_token_count = generated_tokens
    summary_payload = {
        "command": command,
        "command_hash": command_hash,
        "started_at": run_started_at,
        "ended_at": run_ended_at,
        "wall_seconds": run_wall_seconds,
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "output_dir": str(run_summary_dir),
        "settings": {
            "task": args.task,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
            "target_bars": args.target_bars,
            "max_words_per_bar": args.max_words_per_bar,
            "load_in_4bit": args.load_in_4bit,
            "add_structural_special_tokens": args.add_structural_special_tokens,
            "clean_output": args.clean_output,
            "use_cuda": use_cuda,
        },
        "environment": {
            "runtime": runtime,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "bf16_supported": bool(use_cuda and torch.cuda.is_bf16_supported()),
        },
        "timing": timing,
        "artifacts": {
            "run_summary_json": str(run_summary_json),
            "run_summary_md": str(run_summary_md),
            "generation_result_json": str(run_result_json),
            "generation_result_txt": str(run_result_txt),
            "run_log": str(run_log_path),
        },
    }
    summary_payload["metrics"] = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_token_count": generated_token_count,
        "generation_seconds": round(generation_seconds, 2),
        "tokens_per_second": round(generated_tokens / max(generation_seconds, 1e-9), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_cuda else None,
        "generation_start_allocated_gb": round(generation_start_memory_bytes / 1024**3, 2) if use_cuda else None,
    }
    run_summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    run_result_json.write_text(
        json.dumps(
            {
                "command": command,
                "command_hash": command_hash,
                "prompt": prompt,
                "raw_output": tokenizer.decode(output_ids[0], skip_special_tokens=False),
                "cleaned_output": text,
                "timing": timing,
                "settings": summary_payload["settings"],
                "environment": summary_payload["environment"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_result_txt.write_text(text, encoding="utf-8")
    run_log_path.write_text(
        "\n".join(
            [
                f"started_at={run_started_at}",
                f"command_hash={command_hash}",
                f"model_load_seconds={timing['model_load_seconds']}",
                f"generation_seconds={timing['generation_seconds']}",
                f"generated_tokens={generated_token_count}",
                f"tokens_per_second={timing['tokens_per_second']}",
                f"peak_vram_gb={timing.get('max_memory_allocated_gb', 0)}",
                f"ended_at={run_ended_at}",
                f"wall_seconds={run_wall_seconds}",
            ]
        ),
        encoding="utf-8",
    )
    run_summary_md.write_text(
        "\n".join(
            [
                "# Local Generation Run Summary",
                "",
                f"- Command: {' '.join(summary_payload['command'])}",
                f"- Started: {run_started_at}",
                f"- Ended: {run_ended_at}",
                f"- Wall seconds: {run_wall_seconds}",
                f"- Base model: {args.base_model}",
                f"- Adapter: {args.adapter_dir}",
                f"- Quantization: {'4-bit nf4' if args.load_in_4bit and use_cuda else 'non-4-bit fallback'}",
                f"- Command hash: {command_hash}",
                "",
                "```json",
                json.dumps(summary_payload["environment"], indent=2),
                "```",
                "",
                "```json",
                json.dumps(summary_payload["timing"], indent=2),
                "```",
                "",
                "## Output artifacts",
                "",
                f"- `run_summary_json`: {run_summary_json}",
                f"- `run_summary_md`: {run_summary_md}",
                f"- `generation_result_json`: {run_result_json}",
                f"- `generation_result_txt`: {run_result_txt}",
                f"- `run_log`: {run_log_path}",
            ]
        ),
        encoding="utf-8",
    )
    print("[timing] " + json.dumps(timing))
    print(f"[summary] wrote run summary to {run_summary_json}")
    print(text)


if __name__ == "__main__":
    main()
