"""Run a fixed generation eval for a local LoRA adapter."""

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
except ImportError:  # pragma: no cover - optional local convenience.
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_PROMPTS = [
    "Write a 16-bar rap verse about ambition after failure. Use internal rhymes.",
    "Write a gritty storytelling verse about a late-night train ride.",
    "Write a hook about loyalty and pressure. Keep it catchy.",
    "Write a battle rap verse with clever punchlines but no slurs.",
    "Write a reflective verse about success feeling lonely.",
]

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

BLOCKED_SLUR_TERMS = [
    "nigga",
    "niggas",
    "nigger",
    "niggers",
    "faggot",
    "faggots",
    "fag",
    "fags",
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

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SLUR_RE = re.compile(
    r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.22)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--block-slurs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--title", default="Fixed Prompt Generation Eval")
    parser.add_argument(
        "--run-summary-md",
        type=Path,
        default=None,
        help="Optional run_summary.md path for run metadata.",
    )
    parser.add_argument(
        "--run-summary-json",
        type=Path,
        default=None,
        help="Optional run_summary.json path for run metadata.",
    )
    parser.add_argument(
        "--run-summary-dir",
        type=Path,
        default=None,
        help="Directory to store run summary files.",
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError("JSON prompt file must be a list of strings")
        prompts = [item.strip() for item in payload if item.strip()]
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in prompt file: {path}")
    return prompts


def configure_runtime(torch) -> dict[str, object]:
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


def blocked_phrase_ids(tokenizer, *, block_slurs: bool) -> list[list[int]]:
    blocked = []
    phrases = list(BLOCKED_PHRASES)
    if block_slurs:
        phrases.extend(BLOCKED_SLUR_TERMS)
    for phrase in phrases:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            blocked.append(ids)
    return blocked


def clean_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<\|channel\|>\s*thought\s*<channel\|>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<\|channel\|>\s*thought.*?<channel\|>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<\|/?channel.*?\|>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<channel\|>", "", text, flags=re.IGNORECASE)
    text = text.split("<|end|>", 1)[0]
    text = re.sub(r"<\|.*?\|>", "", text)
    cleaned_lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in ARTIFACT_PATTERNS):
            break
        cleaned_lines.append(line)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def repeated_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines if line.strip()]
    if not normalized:
        return 0.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(normalized)


def extract_requested_line_count(prompt: str) -> int | None:
    match = re.search(r"\b(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def analyze_generation(prompt: str, text: str) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    requested_lines = extract_requested_line_count(prompt)
    slur_terms = [match.group(0).lower() for match in SLUR_RE.finditer(text)]
    unique_slurs = sorted(set(slur_terms))
    analysis: dict[str, object] = {
        "line_count": len(lines),
        "word_count": len(words(text)),
        "repeated_line_ratio": round(repeated_line_ratio(lines), 4),
        "non_ascii_count": sum(1 for char in text if ord(char) > 127),
        "requested_line_count": requested_lines,
        "line_count_delta": (len(lines) - requested_lines) if requested_lines is not None else None,
        "exact_line_match": requested_lines == len(lines) if requested_lines is not None else None,
        "is_hook_prompt": "hook" in prompt.lower(),
        "hook_line_cap_ok": (len(lines) <= 8) if "hook" in prompt.lower() else None,
        "no_slurs_requested": "no slurs" in prompt.lower(),
        "slur_count": len(slur_terms),
        "slur_terms": unique_slurs,
        "no_slurs_passed": (len(slur_terms) == 0) if "no slurs" in prompt.lower() else None,
    }
    return analysis


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    analyses = [record["analysis"] for record in records]
    exact_line_records = [item for item in analyses if item["exact_line_match"] is not None]
    hook_records = [item for item in analyses if item["hook_line_cap_ok"] is not None]
    slur_requested = [item for item in analyses if item["no_slurs_requested"]]
    return {
        "prompt_count": len(records),
        "exact_line_match_count": sum(1 for item in exact_line_records if item["exact_line_match"]),
        "exact_line_match_rate": round(
            sum(1 for item in exact_line_records if item["exact_line_match"]) / len(exact_line_records), 2
        )
        if exact_line_records
        else None,
        "hook_line_cap_pass_count": sum(1 for item in hook_records if item["hook_line_cap_ok"]),
        "hook_line_cap_pass_rate": round(
            sum(1 for item in hook_records if item["hook_line_cap_ok"]) / len(hook_records), 2
        )
        if hook_records
        else None,
        "slur_violation_prompt_count": sum(1 for item in analyses if item["slur_count"] > 0),
        "no_slur_prompt_pass_count": sum(1 for item in slur_requested if item["no_slurs_passed"]),
        "no_slur_prompt_pass_rate": round(
            sum(1 for item in slur_requested if item["no_slurs_passed"]) / len(slur_requested), 2
        )
        if slur_requested
        else None,
        "avg_repeated_line_ratio": round(
            sum(float(item["repeated_line_ratio"]) for item in analyses) / len(analyses), 4
        )
        if analyses
        else 0.0,
    }


def write_run_summary(
    *,
    command: list[str],
    command_hash: str,
    run_started_at: str,
    run_ended_at: str,
    wall_seconds: float,
    summary: dict[str, object],
    records: list[dict[str, object]],
    settings: dict[str, object],
    env: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    summary_dir = args.run_summary_dir or args.output_jsonl.parent
    run_summary_json = args.run_summary_json
    run_summary_md = args.run_summary_md
    summary_dir.mkdir(parents=True, exist_ok=True)

    if run_summary_json is None:
        run_summary_json = summary_dir / "run_summary.json"
    if run_summary_md is None:
        run_summary_md = run_summary_json.with_name("run_summary.md")
    run_summary_log = summary_dir / "run_summary.log"
    run_summary_json.parent.mkdir(parents=True, exist_ok=True)
    run_summary_md.parent.mkdir(parents=True, exist_ok=True)

    timings = [record.get("timing", {}) for record in records]
    generated_tokens = [float(item.get("generated_tokens", 0)) for item in timings if isinstance(item.get("generated_tokens"), (int, float))]
    generation_seconds = [float(item.get("generation_seconds", 0)) for item in timings if isinstance(item.get("generation_seconds"), (int, float))]
    peak_memory = [
        float(item["max_memory_allocated_gb"])
        for item in timings
        if isinstance(item.get("max_memory_allocated_gb"), (int, float))
    ]

    payload = {
        "command": command,
        "command_hash": command_hash,
        "started_at": run_started_at,
        "ended_at": run_ended_at,
        "wall_seconds": wall_seconds,
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "settings": settings,
        "environment": env,
        "records": len(records),
        "summary": summary,
        "output": {
            "output_md": str(args.output_md),
            "output_jsonl": str(args.output_jsonl),
            "run_summary_json": str(run_summary_json),
            "run_summary_md": str(run_summary_md),
        },
        "artifacts": {
            "run_summary_json": str(run_summary_json),
            "run_summary_md": str(run_summary_md),
            "run_summary_log": str(run_summary_log),
        },
        "run_metrics": {
            "total_generated_tokens": round(sum(generated_tokens), 2) if generated_tokens else 0.0,
            "total_generation_seconds": round(sum(generation_seconds), 2) if generation_seconds else 0.0,
            "avg_tokens_per_second": round(
                sum(generated_tokens) / max(sum(generation_seconds), 1e-9),
                4,
            )
            if generated_tokens and generation_seconds
            else 0.0,
            "peak_memory_gb": round(max(peak_memory), 4) if peak_memory else None,
        },
    }
    run_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        "# Fixed Generation Run Summary",
        "",
        f"- Command: {' '.join(command)}",
        f"- Started: {run_started_at}",
        f"- Ended: {run_ended_at}",
        f"- Wall seconds: {wall_seconds}",
        f"- Base model: {args.base_model}",
        f"- Adapter: {args.adapter_dir}",
        f"- Output JSONL: {args.output_jsonl}",
        "",
        "## Settings",
        "",
        f"```json\n{json.dumps(settings, indent=2)}\n```",
        "",
        "## Environment",
        "",
        f"```json\n{json.dumps(env, indent=2)}\n```",
        "",
        "## Prompt adherence summary",
        "",
        f"```json\n{json.dumps(summary, indent=2)}\n```",
    ]
    run_summary_md.write_text("\n".join(md_lines), encoding="utf-8")
    run_log_lines = [
        f"run_started_at={run_started_at}",
        f"command={json.dumps(command)}",
        f"command_hash={command_hash}",
        f"wall_seconds={wall_seconds}",
        f"ended={run_ended_at}",
        f"total_generated_tokens={payload['run_metrics'].get('total_generated_tokens', 0.0)}",
        f"avg_tokens_per_second={payload['run_metrics'].get('avg_tokens_per_second', 0.0)}",
        f"peak_memory_gb={payload['run_metrics'].get('peak_memory_gb')}",
    ]
    run_summary_log.write_text("\n".join(run_log_lines), encoding="utf-8")
    return run_summary_json, run_summary_md


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    system = (
        "Write only original rap lyrics for the user's prompt. "
        "Keep line breaks. Do not explain. Do not include metadata, source ids, labels, URLs, or scraped text."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


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
    validate_qwen25_7b_only(args.base_model, scope="Fixed generation eval")
    run_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    command = [sys.executable, *sys.argv]
    command_hash = hashlib.md5(" ".join(command).encode("utf-8")).hexdigest()
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    runtime = configure_runtime(torch)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    device_map = "auto" if use_cuda else None

    load_started_at = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    except Exception as exc:
        print(
            "[tokenizer] WARNING: could not load tokenizer from adapter; "
            f"falling back to base model tokenizer. Error: {exc}",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
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
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = True
    model.eval()
    if not use_cuda:
        model.to("cpu")
    if use_cuda:
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started_at

    settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "seed": args.seed,
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_dir),
        "load_in_4bit": args.load_in_4bit,
        "enable_thinking": not args.disable_thinking,
        "block_slurs": args.block_slurs,
        "quantization": {
            "mode": "4bit_nf4" if use_cuda and args.load_in_4bit else "fp16_fp32_fallback",
            "compute_dtype": str(dtype),
            "bnb_4bit_use_double_quant": bool(use_cuda and args.load_in_4bit),
        },
        "command_hash": command_hash,
    }
    env = {
        "runtime": runtime,
        "model_load_seconds": round(load_seconds, 2),
        "cuda_available": use_cuda,
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }

    prompts = load_prompts(args.prompts_file)
    records = []
    bad_words = blocked_phrase_ids(tokenizer, block_slurs=args.block_slurs)
    settings["blocked_token_sequence_count"] = len(bad_words)
    for index, prompt in enumerate(prompts, start=1):
        torch.manual_seed(args.seed + index - 1)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        chat_template_kwargs = {}
        if args.disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_to_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                prompt_to_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        if use_cuda:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        started_at = time.perf_counter()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_memory = torch.cuda.max_memory_allocated()
        else:
            start_memory = 0
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                bad_words_ids=bad_words,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if use_cuda:
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started_at

        input_length = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_length:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_text = clean_text(raw_text)
        analysis = analyze_generation(prompt, generated_text)
        generated_tokens = int(generated_ids.shape[-1])
        timing = {
            "generation_seconds": round(generation_seconds, 2),
            "generated_tokens": generated_tokens,
            "tokens_per_second": round(generated_tokens / max(generation_seconds, 1e-9), 2),
            "prompt_tokens": prompt_token_count,
        }
        if use_cuda:
            timing["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            timing["start_memory_allocated_gb"] = round(start_memory / 1024**3, 2)
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "seed": args.seed + index - 1,
                "settings": settings,
                "environment": env,
                "timing": timing,
                "raw_text": raw_text,
                "generated_text": generated_text,
                "analysis": analysis,
            }
        )

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    run_ended_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_wall_seconds = round(time.perf_counter() - run_started, 2)
    run_summary_json, run_summary_md = write_run_summary(
        command=command,
        command_hash=command_hash,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        wall_seconds=run_wall_seconds,
        summary=summary,
        records=records,
        settings={
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
            "base_model": args.base_model,
            "adapter_path": str(args.adapter_dir),
            "load_in_4bit": args.load_in_4bit,
            "enable_thinking": not args.disable_thinking,
            "disable_thinking": args.disable_thinking,
            "block_slurs": args.block_slurs,
            "quantization": {
                "mode": "4bit_nf4" if use_cuda and args.load_in_4bit else "fp16_fp32_fallback",
                "compute_dtype": str(dtype),
                "bnb_4bit_use_double_quant": bool(use_cuda and args.load_in_4bit),
            },
            "prompts_file": str(args.prompts_file) if args.prompts_file else None,
            "title": args.title,
            "command_hash": command_hash,
        },
        env={
            "runtime": runtime,
            "model_load_seconds": round(load_seconds, 2),
            "cuda_available": use_cuda,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        args=args,
    )
    print(f"[summary] wrote run summary to {run_summary_json}")
    print(f"[summary] wrote run summary markdown to {run_summary_md}")

    md_lines = [
        f"# {args.title}",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps({"settings": settings, "environment": env}, indent=2),
        "```",
        "",
        "## Prompt Adherence Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    for record in records:
        md_lines.extend(
            [
                f"## Prompt {record['index']}",
                "",
                record["prompt"],
                "",
                "```text",
                record["generated_text"],
                "```",
                "",
                "```json",
                json.dumps(record["timing"], indent=2),
                "```",
                "",
                "```json",
                json.dumps(record["analysis"], indent=2),
                "```",
                "",
            ]
        )
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "complete",
                "records": len(records),
                "settings": settings,
                "environment": env,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
