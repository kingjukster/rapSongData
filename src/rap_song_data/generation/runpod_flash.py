"""Generate lyrics on Runpod Flash using the trained LoRA adapter."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuType

from rap_song_data.integrations.slack import notify_slack, slack_webhook_from_env


load_dotenv()

MODEL_CACHE: dict[str, Any] = {}

ARTIFACT_PATTERNS = [
    r"\[?\s*lyrics taken from\b.*",
    r"\[?\s*lyrics from\b.*",
    r"\[?\s*source:\b.*",
    r"\[?\s*embed\b.*",
    r"\[?\s*you might also like\b.*",
    r"https?://\S+.*",
    r"\bgenius\.com\b.*",
]
SECTION_LABEL_RE = re.compile(r"^\s*\[(intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus).*?\]\s*$", re.I)
CURLY_SECTION_LABEL_RE = re.compile(r"^\s*\{(?:intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus).*?\}\s*$", re.I)
STAGE_DIRECTION_RE = re.compile(r"\*[^*\n]{1,80}\*")
MALFORMED_CONTROL_RE = re.compile(r"<\|\s*(?:begin|end|bar|verse)[^>\n]{0,80}>?", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
BAD_CONTENT_RE = re.compile(
    r"\b(?:dick|suck|fuck(?:in|ing)?|bitch(?:es)?|hoe(?:s)?|shoot|kill|die|fight|murder|gun|strap|club|shots?|skin)\b",
    re.I,
)
INCOMPLETE_END_RE = re.compile(
    r"\b(?:and|but|then|so|for|what|if|when|with|from|to|in|on|of|that|this|these|those|the|a|an|my|your|our|their|call|tell|let|need|want|gotta|gonna|gon'|end|up)$",
    re.I,
)
BLOCKED_PHRASES = [
    "Lyrics taken from",
    "lyrics taken from",
    "Genius",
    "genius.com",
    "https://",
    "http://",
    "You might also like",
    "Embed",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
]
VERSE_START_TOKEN = "<|verse_start|>"
VERSE_END_TOKEN = "<|verse_end|>"
BAR_START_TOKEN = "<|bar_start|>"
STRUCTURAL_SPECIAL_TOKENS = [VERSE_START_TOKEN, VERSE_END_TOKEN, BAR_START_TOKEN]
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
        "tf32_matmul": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
        "cudnn_benchmark": bool(torch.cuda.is_available() and torch.backends.cudnn.benchmark),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


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
    packages = [
        "transformers",
        "accelerate",
        "peft",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "safetensors",
    ]
    try:
        import peft  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


@Endpoint(
    name="rap-lyrics-generate-v5",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),
    dependencies=[],
    execution_timeout_ms=900_000,
)
async def generate_lyrics(job_config: dict[str, Any]) -> dict[str, Any]:
    ensure_remote_packages()

    import torch
    from huggingface_hub import HfApi, login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = validate_qwen25_7b_only(
        job_config.get("base_model", "Qwen/Qwen2.5-7B-Instruct"),
        scope="Runpod Flash generation",
    )
    adapter_repo = job_config["adapter_repo"]
    adapter_subfolder = job_config.get("adapter_subfolder") or None
    add_structural_special_tokens = bool(job_config.get("add_structural_special_tokens", False))
    use_4bit = bool(job_config.get("load_in_4bit", True))
    slack_webhook_url = job_config.get("slack_webhook_url") or os.getenv("SLACK_WEBHOOK_URL") or ""
    run_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run_started = time.perf_counter()
    run_hash = hashlib.md5(json.dumps(job_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    run_summary_dir = job_config.get("run_summary_dir")
    notify_slack(
        slack_webhook_url,
        "Rap lyric generation started",
        {
            "title": job_config.get("title", "Untitled"),
            "adapter": adapter_repo,
            "subfolder": adapter_subfolder,
            "add_structural_special_tokens": add_structural_special_tokens,
        },
    )
    token = job_config.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
    if not token:
        raise ValueError("Missing HF token. Set HF_TOKEN in .env or run hf auth login locally.")

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    login(token=token, add_to_git_credential=False)

    try:
        model_load_started = time.perf_counter()
        tokenizer, model, bad_words_ids = get_cached_generation_assets(
            base_model=base_model,
            adapter_repo=adapter_repo,
            adapter_subfolder=adapter_subfolder,
            add_structural_special_tokens=add_structural_special_tokens,
            load_in_4bit=use_4bit,
            token=token,
            torch=torch,
            HfApi=HfApi,
            PeftModel=PeftModel,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
        )
        model_load_seconds = time.perf_counter() - model_load_started
        batch = job_config.get("batch")
        if batch:
            samples = []
            timing_records: list[dict[str, Any]] = []
            prompt_tokens_total = 0
            generated_tokens_total = 0
            generation_seconds_total = 0.0
            for index, override in enumerate(batch, start=1):
                sample_config = {**job_config, **override}
                sample_config.pop("batch", None)
                sample = generate_one_sample(sample_config, tokenizer, model, bad_words_ids, torch)
                sample["batch_index"] = index
                timing_records.append(sample.get("timing", {}))
                prompt_tokens_total += int(sample.get("prompt_token_count", 0))
                generated_tokens_total += int(sample.get("generated_token_count", 0))
                generation_seconds_total += float(sample.get("timing", {}).get("generation_seconds", 0.0))
                samples.append(sample)
            result = {
                "command": ["runpod-flash", "generate_lyrics"],
                "command_hash": run_hash,
                "samples": samples,
                "run_started_at": run_started_at,
                "run_ended_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "run_wall_seconds": round(time.perf_counter() - run_started, 2),
                "base_model": base_model,
                "adapter_repo": adapter_repo,
                "adapter_subfolder": adapter_subfolder,
                "timing": {
                    "model_load_seconds": round(model_load_seconds, 2),
                    "total_generation_seconds": round(generation_seconds_total, 2),
                    "total_generated_tokens": generated_tokens_total,
                    "prompt_tokens_total": prompt_tokens_total,
                    "avg_tokens_per_second": round(
                        generated_tokens_total / max(generation_seconds_total, 1e-9),
                        2,
                    )
                    if generated_tokens_total and generation_seconds_total
                    else 0.0,
                },
                "timing_records": timing_records,
                "runtime": configure_runtime(torch),
            }
        else:
            result = generate_one_sample(job_config, tokenizer, model, bad_words_ids, torch)
            result["timing"]["model_load_seconds"] = round(model_load_seconds, 2)
            result.update(
                {
                    "command": ["runpod-flash", "generate_lyrics"],
                    "command_hash": run_hash,
                    "run_started_at": run_started_at,
                    "run_ended_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "run_wall_seconds": round(time.perf_counter() - run_started, 2),
                    "runtime": configure_runtime(torch),
                    "base_model": base_model,
                    "adapter_repo": adapter_repo,
                    "adapter_subfolder": adapter_subfolder,
                }
            )
        if run_summary_dir:
            run_summary_dir_path = Path(str(run_summary_dir))
            run_summary_dir_path.mkdir(parents=True, exist_ok=True)
            run_summary_json = run_summary_dir_path / "run_summary.json"
            run_summary_md = run_summary_dir_path / "run_summary.md"
            run_summary_log = run_summary_dir_path / "run_summary.log"
            run_summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            sample_count = len(result.get("samples", [result]))
            run_summary_md.write_text(
                "\n".join(
                    [
                        "# Runpod Flash Generation Run Summary",
                        "",
                        f"- Command: {' '.join(result['command'])}",
                        f"- Command hash: {result['command_hash']}",
                        f"- Started: {run_started_at}",
                        f"- Ended: {result['run_ended_at']}",
                        f"- Wall seconds: {result['run_wall_seconds']}",
                        f"- Base model: {base_model}",
                        f"- Adapter: {adapter_repo}",
                        f"- Generated samples: {sample_count}",
                        "- Timing:",
                        f"  - Model load seconds: {result['timing']['model_load_seconds']}",
                        f"  - Total generation seconds: {result['timing'].get('total_generation_seconds', result['timing'].get('generation_seconds', 0.0))}",
                    ]
                ),
                encoding="utf-8",
            )
            run_summary_log.write_text(
                "\n".join(
                    [
                        f"run_started_at={run_started_at}",
                        f"command={json.dumps(result['command'])}",
                        f"command_hash={result['command_hash']}",
                        f"wall_seconds={result['run_wall_seconds']}",
                        f"total_generated_tokens={result['timing'].get('total_generated_tokens', result['timing'].get('generated_token_count', 0))}",
                        f"avg_tokens_per_second={result['timing'].get('avg_tokens_per_second', result['timing'].get('tokens_per_second', 0.0))}",
                        f"ended={result['run_ended_at']}",
                    ]
                ),
                encoding="utf-8",
            )
            result["run_summary_json"] = str(run_summary_json)
            result["run_summary_md"] = str(run_summary_md)
            result["run_summary_log"] = str(run_summary_log)
            result["artifacts"] = {
                "run_summary_json": str(run_summary_json),
                "run_summary_md": str(run_summary_md),
                "run_summary_log": str(run_summary_log),
            }
        notify_slack(
            slack_webhook_url,
            "Rap lyric generation completed",
            {
                "title": job_config.get("title", "Untitled"),
                "samples": len(result.get("samples", [result])),
                "preview": (result.get("lyrics") or (result.get("samples") or [{}])[0].get("lyrics", ""))[:500],
            },
        )
        return result
    except Exception as exc:
        notify_slack(
            slack_webhook_url,
            "Rap lyric generation failed",
            {
                "title": job_config.get("title", "Untitled"),
                "error": str(exc)[:500],
            },
        )
        raise


def get_cached_generation_assets(
    *,
    base_model: str,
    adapter_repo: str,
    adapter_subfolder: str | None,
    add_structural_special_tokens: bool,
    load_in_4bit: bool,
    token: str,
    torch: Any,
    HfApi: Any,
    PeftModel: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
):
    cache_key = f"{base_model}|{adapter_repo}|{adapter_subfolder or ''}|special={add_structural_special_tokens}"
    if MODEL_CACHE.get("key") == cache_key:
        return MODEL_CACHE["tokenizer"], MODEL_CACHE["model"], MODEL_CACHE["bad_words_ids"]

    MODEL_CACHE.clear()
    HfApi().model_info(adapter_repo, token=token)
    adapter_kwargs = {"token": token, "subfolder": adapter_subfolder or ""}

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        token=token,
        use_fast=True,
    )
    added_tokens = 0
    if add_structural_special_tokens:
        added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": STRUCTURAL_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "token": token,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_repo, **adapter_kwargs)
    model.config.use_cache = True
    model.eval()

    bad_words_ids = []
    for phrase in BLOCKED_PHRASES:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            bad_words_ids.append(ids)

    MODEL_CACHE.update(
        {
            "key": cache_key,
            "tokenizer": tokenizer,
            "model": model,
            "bad_words_ids": bad_words_ids,
        }
    )
    return tokenizer, model, bad_words_ids


def build_prompt(job_config: dict[str, Any]) -> str:
    title = job_config.get("title", "Untitled")
    artist = job_config.get("artist", "Original")
    rap_family = job_config.get("rap_family", "Melodic / Emo / Cloud")
    rap_category = job_config.get("rap_category", "Emo Rap / Melodic Rap")
    year = int(job_config.get("year", 2026))
    views_log = job_config.get("views_log", "0.0000")
    structure = job_config.get(
        "structure",
        "Exactly 12 lines. Each line should be one short rap bar. No chorus. No bracket labels. No explanations. No paragraph-style lines.",
    )
    theme = job_config.get("theme", "working late, loneliness, ambition, city lights")
    keywords = job_config.get("keywords", "night, shift, city, lights, work, ambition")
    max_words_per_bar = int(job_config.get("max_words_per_line", job_config.get("max_words_per_bar", 9)))
    target_bars = int(job_config.get("target_bars", job_config.get("target_lines", 12)))
    bar_count_range = job_config.get("bar_count_range", "10-25")
    rules = job_config.get(
        "rules",
        "Write only original lyrics. Keep line breaks. Do not explain. Stay focused on the requested theme and keywords. Avoid random sexual content, threats, unrelated violent imagery, parenthetical ad-libs, bracket labels, dialogue, URLs, source notes, and copied song text.",
    )

    return (
        "<|task|>generate_verse\n"
        f"<|title|>{title}\n"
        f"<|artist|>{artist}\n"
        f"<|rap_family|>{rap_family}\n"
        f"<|rap_category|>{rap_category}\n"
        f"<|year|>{year}\n"
        f"<|views_log|>{views_log}\n"
        f"<|structure|>{structure}\n"
        f"<|target_bars|>{target_bars}\n"
        f"<|bar_count_range|>{bar_count_range}\n"
        f"<|max_words_per_bar|>{max_words_per_bar}\n"
        f"<|theme|>{theme}\n"
        f"<|keywords|>{keywords}\n"
        f"<|rules|>{rules}\n"
        "<|lyrics|>\n"
        f"{VERSE_START_TOKEN}\n"
        f"{BAR_START_TOKEN}"
    )


def generate_one_sample(
    job_config: dict[str, Any],
    tokenizer: Any,
    model: Any,
    bad_words_ids: list[list[int]],
    torch: Any,
) -> dict[str, Any]:
    prompt = build_prompt(job_config)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seed = int(job_config.get("seed", 42))
    torch.manual_seed(seed)
    prompt_token_count = int(inputs["input_ids"].shape[-1])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generation_start_memory = torch.cuda.max_memory_allocated()
    else:
        generation_start_memory = 0
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(job_config.get("max_new_tokens", 180)),
            min_new_tokens=int(job_config.get("min_new_tokens", 40)),
            do_sample=True,
            temperature=float(job_config.get("temperature", 0.75)),
            top_p=float(job_config.get("top_p", 0.85)),
            top_k=int(job_config.get("top_k", 50)),
            repetition_penalty=float(job_config.get("repetition_penalty", 1.35)),
            no_repeat_ngram_size=int(job_config.get("no_repeat_ngram_size", 5)),
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    lyrics = clean_generated_lyrics(text, max_lines=int(job_config.get("max_lines", 24)))
    if not lyrics:
        lyrics = clean_generated_lyrics(raw_completion, max_lines=int(job_config.get("max_lines", 24)))
    generated_token_count = int(generated_ids.shape[-1])
    timing = {
        "generation_seconds": round(generation_seconds, 2),
        "generated_token_count": generated_token_count,
        "tokens_per_second": round(generated_token_count / max(generation_seconds, 1e-9), 2),
    }
    if torch.cuda.is_available():
        timing["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        timing["start_memory_allocated_gb"] = round(generation_start_memory / 1024**3, 2)
    if job_config.get("force_line_count", True):
        lyrics = force_line_count(
            lyrics,
            target_lines=int(job_config.get("target_lines", 12)),
            max_words_per_line=int(job_config.get("max_words_per_line", 9)),
        )
    return {
        "lyrics": lyrics,
        "prompt": prompt,
        "raw_completion_preview": raw_completion[:1200],
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "timing": timing,
    }


def clean_generated_lyrics(text: str, max_lines: int = 24) -> str:
    if "<|lyrics|>" in text:
        text = text.split("<|lyrics|>", 1)[1]
    if VERSE_END_TOKEN in text:
        text = text.split(VERSE_END_TOKEN, 1)[0]
    if "<|end|>" in text:
        parts = text.split("<|end|>")
        text = max(parts, key=lambda part: len(WORD_RE.findall(part)), default=text)

    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:i|b|em|strong|p|mark|sub|sup|span|div)[^>]*>", "", text, flags=re.I)
    text = text.replace(VERSE_START_TOKEN, "")
    text = text.replace(VERSE_END_TOKEN, "")
    text = text.replace(BAR_START_TOKEN, "\n")
    text = re.sub(r"<\|\s*bar_?start\s*\|?\s*[=:\-]*", "\n", text, flags=re.I)
    text = re.sub(r"<\|\s*verse_?start\s*\|?\s*[=:\-]*", "", text, flags=re.I)
    text = re.sub(r"<\|\s*verse_?end[^>\n]{0,80}>?", "", text, flags=re.I)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<\|[^>\n]{0,80}>", "", text)
    text = re.sub(r"<\|[^|\n]{0,80}$", "", text)
    text = MALFORMED_CONTROL_RE.sub("\n", text)
    text = STAGE_DIRECTION_RE.sub("", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    replacements = {
        "feelingsgone": "feelings gone",
        "theyseein": "they seein",
        "its ": "it's ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n"):
        line = normalize_generated_line(raw_line)
        if not line:
            continue
        line = re.sub(r"\[[^\]]{0,80}\]", "", line).strip()
        line = re.sub(r"\{[^}]{0,80}\}", "", line).strip()
        line = strip_broken_parenthetical(line)
        if not line:
            continue
        if starts_like_continuation(line) and len(line.split()) <= 2:
            continue
        if SECTION_LABEL_RE.match(line) or CURLY_SECTION_LABEL_RE.match(line):
            continue
        if line.count('"') >= 2 and re.search(r"\b(?:said|asked|told|yo|hey|listen)\b", line, re.I):
            continue
        if BAD_CONTENT_RE.search(line):
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in ARTIFACT_PATTERNS):
            break
        if len(line) > 180:
            continue
        cleaned_lines.append(line)
        if len(cleaned_lines) >= max_lines:
            break
    cleaned_lines = merge_continuation_lines(cleaned_lines)
    cleaned_lines = drop_incomplete_tail(cleaned_lines)
    return "\n".join(cleaned_lines).strip()


def force_line_count(text: str, target_lines: int = 12, max_words_per_line: int = 9) -> str:
    source_lines = merge_continuation_lines([line.strip() for line in text.splitlines() if line.strip()])
    if not source_lines:
        return ""

    lines: list[str] = []
    for source_line in source_lines:
        for segment in phrase_segments(source_line, max_words_per_line):
            lines.extend(split_long_line(segment, max_words_per_line))
            if len(lines) >= target_lines:
                break
        if len(lines) >= target_lines:
            break

    lines = merge_short_tail(drop_incomplete_tail(lines[:target_lines]), max_words_per_line)
    return "\n".join(lines).strip()


def normalize_generated_line(line: str) -> str:
    line = re.sub(r"[^\x00-\x7F]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[.\-–—:;,\s]+", "", line).strip()
    line = re.sub(r"\([^)]{1,40}\)", "", line).strip()
    line = re.sub(
        r"\b(on|so|and|but|then|for|with|to|from|in) ([A-Z][a-z]+)\b",
        lambda match: f"{match.group(1)} {match.group(2).lower()}",
        line,
    )
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    return line.strip()


def strip_broken_parenthetical(line: str) -> str:
    if line.count("(") > line.count(")"):
        line = re.sub(r"\s*\([^)]*$", "", line).strip()
    if line.count(")") > line.count("("):
        line = re.sub(r"^[^(]*\)\s*", "", line).strip()
    return line


def drop_incomplete_tail(lines: list[str]) -> list[str]:
    cleaned = list(lines)
    while cleaned:
        tail = strip_broken_parenthetical(cleaned[-1]).strip()
        if not tail or INCOMPLETE_END_RE.search(tail):
            cleaned.pop()
            continue
        cleaned[-1] = tail
        break
    return cleaned


def merge_short_tail(lines: list[str], max_words_per_line: int) -> list[str]:
    if len(lines) < 2:
        return lines
    tail_words = lines[-1].split()
    previous_words = lines[-2].split()
    if len(tail_words) <= 2 and len(previous_words) + len(tail_words) <= max_words_per_line + 5:
        merged = list(lines[:-2])
        merged.append(normalize_generated_line(f"{lines[-2]} {lines[-1]}"))
        return merged
    return lines


def merge_continuation_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        line = strip_broken_parenthetical(line).strip()
        if not line:
            continue
        if merged and (INCOMPLETE_END_RE.search(merged[-1]) or starts_like_continuation(line)):
            merged[-1] = normalize_generated_line(f"{merged[-1]} {line}")
        else:
            merged.append(line)
    return merged


def phrase_segments(line: str, max_words_per_line: int) -> list[str]:
    line = normalize_generated_line(line)
    if len(line.split()) <= max_words_per_line:
        return [line]

    segments: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= max_words_per_line + 3:
            segments.append(sentence)
            continue
        segments.extend(split_on_soft_boundaries(sentence, max_words_per_line))
    return [segment for segment in segments if segment]


def split_on_soft_boundaries(line: str, max_words_per_line: int) -> list[str]:
    pieces = [piece.strip(" ,;") for piece in re.split(r"[,;]\s+", line) if piece.strip(" ,;")]
    if len(pieces) > 1 and all(len(piece.split()) >= 3 for piece in pieces):
        return pieces

    tokens = line.split()
    segments: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        end = min(cursor + max_words_per_line + 2, len(tokens))
        split_at = best_connector_split(tokens, cursor, end)
        if split_at <= cursor:
            split_at = end
        segment = " ".join(tokens[cursor:split_at]).strip(" ,;")
        if segment:
            segments.append(segment)
        cursor = split_at
    return segments


def best_connector_split(tokens: list[str], start: int, end: int) -> int:
    connectors = {"and", "but", "so", "then", "cause", "'cause", "because"}
    lower_bound = start + 4
    for index in range(end - 1, lower_bound, -1):
        token = re.sub(r"[^A-Za-z']+", "", tokens[index]).lower()
        if token in connectors:
            return index
    return end


def starts_like_continuation(line: str) -> bool:
    first = WORD_RE.search(line)
    if not first:
        return False
    token = first.group(0)
    return token[0].islower() or token.lower() in {"now", "then", "but", "and", "cause", "cuz"}


def split_long_line(line: str, max_words_per_line: int) -> list[str]:
    tokens = line.split()
    if len(tokens) <= max_words_per_line:
        return [line]

    chunks: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        end = min(cursor + max_words_per_line, len(tokens))
        while end < len(tokens) and end < cursor + max_words_per_line + 3:
            candidate = " ".join(tokens[cursor:end])
            next_token = tokens[end] if end < len(tokens) else ""
            if not should_extend_chunk(candidate, next_token):
                break
            end += 1
        chunk = " ".join(tokens[cursor:end]).strip(" ,;")
        if chunk:
            chunks.append(strip_broken_parenthetical(chunk))
        cursor = end
    return drop_incomplete_tail([chunk for chunk in chunks if chunk])


def should_extend_chunk(candidate: str, next_token: str) -> bool:
    if INCOMPLETE_END_RE.search(candidate):
        return True
    if not next_token:
        return False
    if re.search(r"[,.;:!?)]$", candidate):
        return False
    clean_next = re.sub(r"^[^A-Za-z0-9']+", "", next_token)
    if not clean_next:
        return False
    return clean_next[0].islower() or clean_next.lower() in {"on", "up", "air", "home", "me", "you", "it", "them"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-repo", default="kingjukster77/rap-lyrics-lora-adapter")
    parser.add_argument("--adapter-subfolder", default=None)
    parser.add_argument("--add-structural-special-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--title", default="Night Shift")
    parser.add_argument("--artist", default="Original")
    parser.add_argument("--rap-family", default="Melodic / Emo / Cloud")
    parser.add_argument("--rap-category", default="Emo Rap / Melodic Rap")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--structure", default="Exactly 12 lines. Each line should be one short rap bar. No chorus. No bracket labels. No explanations. No paragraph-style lines.")
    parser.add_argument("--theme", default="working late, loneliness, ambition, city lights")
    parser.add_argument("--keywords", default="night, shift, city, lights, work, ambition")
    parser.add_argument("--rules", default="Write only original lyrics. Keep line breaks. Do not explain. Stay focused on the requested theme and keywords. Avoid random sexual content, threats, unrelated violent imagery, parenthetical ad-libs, bracket labels, dialogue, URLs, source notes, and copied song text.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--min-new-tokens", type=int, default=40)
    parser.add_argument("--max-lines", type=int, default=24)
    parser.add_argument("--target-lines", type=int, default=12)
    parser.add_argument("--target-bars", type=int, default=None)
    parser.add_argument("--bar-count-range", default="10-25")
    parser.add_argument("--max-words-per-line", type=int, default=9)
    parser.add_argument("--force-line-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.35)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slack-webhook-url", default=None)
    parser.add_argument(
        "--run-summary-dir",
        type=Path,
        default=None,
        help="Optional directory to write run_summary.json/md/log for generation runs.",
    )
    parser.add_argument(
        "--batch-params-json",
        default=None,
        help="JSON list of per-sample generation overrides. Used by evaluate_generation.py to reduce remote calls.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    batch = json.loads(args.batch_params_json) if args.batch_params_json else None
    result = await generate_lyrics(
        {
            "adapter_repo": args.adapter_repo,
            "adapter_subfolder": args.adapter_subfolder,
            "add_structural_special_tokens": args.add_structural_special_tokens,
            "hf_token": hf_token_for_remote(),
            "title": args.title,
            "artist": args.artist,
            "rap_family": args.rap_family,
            "rap_category": args.rap_category,
            "year": args.year,
            "structure": args.structure,
            "theme": args.theme,
            "keywords": args.keywords,
            "rules": args.rules,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "max_lines": args.max_lines,
            "target_lines": args.target_lines,
            "target_bars": args.target_bars or args.target_lines,
            "bar_count_range": args.bar_count_range,
            "max_words_per_line": args.max_words_per_line,
            "force_line_count": args.force_line_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
            "load_in_4bit": args.load_in_4bit,
            "run_summary_dir": args.run_summary_dir,
            "slack_webhook_url": args.slack_webhook_url or slack_webhook_from_env(),
            "batch": batch,
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
