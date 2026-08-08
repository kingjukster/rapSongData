"""Run a local batched causal-LM generation sweep.

The sweep can use a supported local LoRA adapter or the base model alone,
writes raw and postprocessed outputs, and does not call OpenAI or any external
judge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rap_song_data.evaluation.fixed_generation import (
    generation_eos_token_ids,
    generation_pad_token_id,
)


BASE_MODEL = "Qwen/Qwen3-4B"
SUPPORTED_BASE_MODELS = {
    BASE_MODEL,
    "Qwen/Qwen3-14B",
    "allenai/Olmo-3-7B-Instruct",
}
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DEFAULT_ADAPTER = Path("model/artifacts/stage2-qwen3-4b-cleaned-chunks-512-60m")
OUTPUT_PROMPT_METADATA_KEYS = (
    "theme_id",
    "instruction_family",
    "prompt_family",
    "evaluation_split",
    "samples_per_model",
)

SYSTEM_PROMPT = (
    "Write only original rap lyrics for the user's prompt. Keep line breaks. "
    "Do not explain. Do not include metadata, source ids, labels, URLs, or scraped text."
)

ARTIFACT_PATTERNS = [
    r"\[?\s*lyrics taken from\b.*",
    r"\[?\s*lyrics from\b.*",
    r"\[?\s*source:\b.*",
    r"\[?\s*embed\b.*",
    r"\[?\s*you might also like\b.*",
    r"https?://\S+.*",
    r"\bgenius\.com\b.*",
]

BLOCKED_ARTIFACT_PHRASES = [
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

THEMES = [
    "ambition after public failure",
    "pressure, loyalty, and temptation",
    "late-night city focus after a long shift",
    "family pride and clean success",
    "loneliness after a breakthrough",
    "discipline, patience, and technical growth",
    "rebuilding a name after rumors",
    "standing outside a closed corner store at 2 AM",
    "turning rejection into a sharper routine",
    "keeping promises while the city moves fast",
]

STYLES = [
    "internal rhymes and crisp line breaks",
    "multisyllabic rhymes with a controlled cadence",
    "gritty storytelling where every line advances the scene",
    "melodic but compact phrasing",
    "punchline-heavy writing without slurs or hate speech",
]

PROMPT_TEMPLATES = [
    "Write exactly 16 lines of original rap lyrics about {theme}. Use {style}. No intro, no commentary.",
    "Write a 16-bar verse about {theme}. Keep it coherent, vivid, and original. Use {style}.",
    "Write a clean radio-safe verse about {theme}. No profanity and no slurs. Use {style}.",
    "Write a hook with 4 short lines about {theme}. Make it catchy, direct, and original.",
    "Write a battle rap verse about {theme}. Avoid slurs, hate speech, and sexual threats. Use {style}.",
    "Write a dark cinematic verse about {theme}. Keep every line grounded in the same scene.",
    "Write a reflective verse about {theme}. Avoid question endings and avoid dialogue.",
    "Write a technical rhyme-heavy verse about {theme}. Use {style}. No bracket labels.",
    "Write a storytelling verse about {theme}. Every line should move the image forward.",
    "Write exactly 12 lines about {theme}. Keep the ending complete and declarative.",
]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/sweeps/qwen3_4b_rebuild/sweep_raw.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/sweeps/qwen3_4b_rebuild/sweep_summary.json"))
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.82)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.16)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-slurs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-target-line-count", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--underlength-retries",
        type=int,
        default=0,
        help="Additional generation attempts only when the postprocessed output is under the requested line count.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-row-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force batch size 1 so every recorded row seed is the seed actually used.",
    )
    return parser.parse_args()


def stable_id(parts: list[str]) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_run_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_generation_run_manifest(
    path: Path,
    *,
    spec: dict[str, Any],
    resume: bool,
    output_exists: bool,
) -> str:
    fingerprint = generation_run_fingerprint(spec)
    if resume and output_exists:
        if not path.exists():
            raise RuntimeError(f"Resume refused because the generation run manifest is missing: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint or existing.get("spec") != spec:
            raise RuntimeError("Resume refused because model, adapter, prompts, seeds, or decoding settings changed.")
        return fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "fingerprint": fingerprint, "spec": spec}, indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint


def build_prompt_bank() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for theme in THEMES:
        for template_index, template in enumerate(PROMPT_TEMPLATES):
            style = STYLES[(len(prompts) + template_index) % len(STYLES)]
            prompt = template.format(theme=theme, style=style)
            prompts.append(
                {
                    "prompt_key": stable_id([prompt]),
                    "prompt": prompt,
                    "theme": theme,
                    "template_index": template_index,
                    "style": style,
                }
            )
    return prompts


def load_prompt_bank(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return build_prompt_bank()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    prompts: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON prompt file must contain a list")
        for item in payload:
            if isinstance(item, str):
                prompt = item.strip()
                metadata: dict[str, Any] = {}
            elif isinstance(item, dict):
                prompt = str(item.get("prompt") or item.get("instruction") or "").strip()
                metadata = {key: value for key, value in item.items() if key not in {"prompt", "instruction"}}
            else:
                continue
            if prompt:
                prompts.append({"prompt_key": stable_id([prompt]), "prompt": prompt, **metadata})
    else:
        for line in text.splitlines():
            prompt = line.strip()
            if prompt:
                prompts.append({"prompt_key": stable_id([prompt]), "prompt": prompt})
    if not prompts:
        raise ValueError(f"No prompts found in prompt file: {path}")
    return prompts


def build_sweep_plan(prompt_bank: list[dict[str, Any]], *, num_candidates: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prompts = list(prompt_bank)
    rng.shuffle(prompts)
    plan: list[dict[str, Any]] = []
    for index in range(num_candidates):
        base = prompts[index % len(prompts)]
        sample_index = index // len(prompts)
        plan.append(
            {
                **base,
                "candidate_index": index + 1,
                "sample_index": sample_index,
                "row_id": stable_id([base["prompt_key"], str(sample_index), str(seed)]),
                "seed": seed + index,
            }
        )
    return plan


def attach_prompt_metadata(record: dict[str, Any], plan_row: dict[str, Any]) -> dict[str, Any]:
    for key in OUTPUT_PROMPT_METADATA_KEYS:
        if key in plan_row:
            record[key] = plan_row[key]
    return record


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def blocked_phrase_ids(tokenizer: Any, *, block_slurs: bool) -> list[list[int]]:
    phrases = list(BLOCKED_ARTIFACT_PHRASES)
    if block_slurs:
        phrases.extend(BLOCKED_SLUR_TERMS)
    blocked: list[list[int]] = []
    for phrase in phrases:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            blocked.append(ids)
    return blocked


def count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def requested_line_count(prompt: str) -> int | None:
    match = re.search(r"\b(?:exactly\s+)?(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, re.I)
    return int(match.group(1)) if match else None


def clean_special_tokens(text: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    original = text
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
        actions.append("remove_thinking_trace")
    patterns = [
        r"<think>.*?</think>",
        r"<think>.*",
        r"<\|channel\|>\s*thought.*?<channel\|>",
        r"<\|/?channel.*?\|>",
        r"<\|.*?\|>",
    ]
    for pattern in patterns:
        new_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        if new_text != text:
            actions.append("clean_model_artifacts")
            text = new_text
    text = text.split("<|end|>", 1)[0]
    if text != original and "clean_model_artifacts" not in actions:
        actions.append("clean_model_artifacts")
    return text, sorted(set(actions))


def postprocess_text(raw_text: str, *, target_line_count: int | None = None) -> tuple[str, list[str]]:
    text, actions = clean_special_tokens(raw_text)
    normalized_text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
        .replace("\ufeff", "")
        .replace("\u200b", "")
    )
    if normalized_text != text:
        text = normalized_text
        actions.append("normalize_unicode_punctuation")
    cleaned_lines: list[str] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```") or line.lower() in {"text", "lyrics"}:
            actions.append("clean_model_artifacts")
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in ARTIFACT_PATTERNS):
            actions.append("clean_model_artifacts")
            break
        line = re.sub(r"^\s*(?:verse|hook|chorus|bridge|intro|outro)\s*:?\s*", "", line, flags=re.IGNORECASE)
        if line:
            cleaned_lines.append(line)

    if len(cleaned_lines) >= 3:
        last = cleaned_lines[-1].strip()
        last_words = WORD_RE.findall(last)
        weak_question = last.endswith("?") or re.search(r"\b(what do you think|should i|can you|right)\??$", last, re.I)
        weak_fragment = len(last_words) <= 3 and not re.search(r"[.!]$", last)
        if weak_question or weak_fragment:
            cleaned_lines.pop()
            actions.append("drop_dangling_final_line")

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    if target_line_count is not None and target_line_count > 0 and len(cleaned_lines) > target_line_count:
        cleaned_lines = cleaned_lines[:target_line_count]
        actions.append("trim_to_target_line_count")
    return "\n".join(cleaned_lines).strip(), sorted(set(actions))


def needs_underlength_retry(text: str, *, target_line_count: int | None) -> bool:
    return target_line_count is not None and target_line_count > 0 and count_lines(text) < target_line_count


def structural_failure_tags(text: str, *, target_line_count: int | None) -> list[str]:
    if target_line_count is None or target_line_count <= 0:
        return []
    line_count = count_lines(text)
    if line_count < target_line_count:
        return ["fixable_underlength"]
    if line_count > target_line_count:
        return ["overlength"]
    return []


def configure_runtime(torch: Any) -> dict[str, Any]:
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


def existing_ids(path: Path) -> set[str]:
    return set(inspect_output_jsonl(path)["row_ids"])


def inspect_output_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"row_ids": [], "duplicate_row_ids": [], "malformed_rows": []}
    row_ids: list[str] = []
    malformed_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_rows.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("row_id"), str):
                malformed_rows.append({"line": line_number, "error": "missing_string_row_id"})
                continue
            row_ids.append(payload["row_id"])
    counts = Counter(row_ids)
    return {
        "row_ids": row_ids,
        "duplicate_row_ids": sorted(row_id for row_id, count in counts.items() if count > 1),
        "malformed_rows": malformed_rows,
    }


def prepare_output_jsonl(path: Path, *, resume: bool) -> set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.write_text("", encoding="utf-8")
        return set()
    return existing_ids(path)


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cublas_status_alloc_failed" in text


def generation_finish_metadata(
    generated_ids: Any,
    tokenizer: Any,
    max_new_tokens: int,
    configured_eos_token_ids: int | list[int] | None = None,
) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in generated_ids.detach().cpu().tolist()]
    eos_token_ids = set()
    configured = configured_eos_token_ids
    if configured is None:
        configured = tokenizer.eos_token_id
    if configured is not None:
        if isinstance(configured, list):
            eos_token_ids.update(int(token_id) for token_id in configured)
        else:
            eos_token_ids.add(int(configured))
    first_eos_index = next((index for index, token_id in enumerate(token_ids) if token_id in eos_token_ids), None)
    hit_eos = first_eos_index is not None
    generated_token_count = len(token_ids)
    hit_token_cap = not hit_eos and generated_token_count >= max_new_tokens
    return {
        "generated_tokens": generated_token_count,
        "effective_generated_tokens": (first_eos_index + 1) if hit_eos else generated_token_count,
        "hit_eos": hit_eos,
        "first_eos_token_index": first_eos_index,
        "hit_token_cap": hit_token_cap,
        "finish_reason": "eos" if hit_eos else ("length" if hit_token_cap else "unknown"),
    }


def decode_content_before_eos(generated_ids: Any, tokenizer: Any, finish: dict[str, Any]) -> tuple[str, str | None]:
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    first_eos = finish.get("first_eos_token_index")
    if isinstance(first_eos, int):
        content_ids = generated_ids[:first_eos]
        content_text = tokenizer.decode(content_ids, skip_special_tokens=False)
        return content_text, full_text if full_text != content_text else None
    return full_text, None


def main() -> None:
    args = parse_args()
    if args.base_model not in SUPPORTED_BASE_MODELS:
        raise SystemExit(
            f"Unsupported base model {args.base_model!r}; supported models: "
            f"{sorted(SUPPORTED_BASE_MODELS)}"
        )
    if args.adapter and not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_dir}")
    if args.underlength_retries < 0:
        raise ValueError("--underlength-retries must be >= 0")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if args.adapter:
        from peft import PeftModel

    runtime = configure_runtime(torch)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This sweep is intended for local GPU generation.")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    prompt_bank = load_prompt_bank(args.prompt_file)
    full_plan = build_sweep_plan(prompt_bank, num_candidates=args.num_candidates, seed=args.seed)
    expected_ids = {row["row_id"] for row in full_plan}
    run_manifest_path = args.run_manifest or args.output_jsonl.with_name(
        f"{args.output_jsonl.stem}.run_manifest.json"
    )
    adapter_config = args.adapter_dir / "adapter_config.json"
    adapter_weights = args.adapter_dir / "adapter_model.safetensors"
    run_spec = {
        "generator_source_sha256": sha256_file(Path(__file__)),
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "adapter_enabled": args.adapter,
        "adapter_dir": str(args.adapter_dir) if args.adapter else None,
        "adapter_config_sha256": sha256_file(adapter_config) if args.adapter and adapter_config.exists() else None,
        "adapter_weights_sha256": sha256_file(adapter_weights) if args.adapter and adapter_weights.exists() else None,
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "prompt_bank_sha256": hashlib.sha256(
            json.dumps(prompt_bank, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "planned_row_ids_sha256": hashlib.sha256("\n".join(sorted(expected_ids)).encode("utf-8")).hexdigest(),
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "strict_row_seeds": args.strict_row_seeds,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "disable_thinking": args.disable_thinking,
        "block_slurs": args.block_slurs,
        "enforce_target_line_count": args.enforce_target_line_count,
        "underlength_retries": args.underlength_retries,
        "load_in_4bit": args.load_in_4bit,
    }
    run_fingerprint = prepare_generation_run_manifest(
        run_manifest_path,
        spec=run_spec,
        resume=args.resume,
        output_exists=args.output_jsonl.exists(),
    )
    done_ids = prepare_output_jsonl(args.output_jsonl, resume=args.resume)
    initial_output_audit = inspect_output_jsonl(args.output_jsonl)
    if initial_output_audit["malformed_rows"] or initial_output_audit["duplicate_row_ids"]:
        raise RuntimeError(f"Existing generation output is malformed or duplicated: {initial_output_audit}")
    unexpected_existing = done_ids - expected_ids
    if unexpected_existing:
        raise RuntimeError(
            f"Existing generation output contains {len(unexpected_existing)} row ids outside this plan. "
            "Use --no-resume or a fresh path."
        )
    plan = [row for row in full_plan if row["row_id"] not in done_ids]

    load_started = time.perf_counter()
    tokenizer_source = args.adapter_dir if args.adapter else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if args.adapter else args.model_revision,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        **model_kwargs,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.config.use_cache = True
    model.eval()
    eos_token_ids = generation_eos_token_ids(model, tokenizer)
    pad_token_id = generation_pad_token_id(tokenizer)
    bad_words = blocked_phrase_ids(tokenizer, block_slurs=args.block_slurs)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    generated = len(done_ids)
    batch_size = max(1, args.batch_size)
    if args.strict_row_seeds and batch_size != 1:
        print(f"[determinism] forcing batch_size=1 instead of {batch_size} for exact per-row seeds")
        batch_size = 1
    started = time.perf_counter()
    settings = {
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "adapter_dir": str(args.adapter_dir) if args.adapter else None,
        "adapter_enabled": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "seed": args.seed,
        "block_slurs": args.block_slurs,
        "enforce_target_line_count": args.enforce_target_line_count,
        "underlength_retries": args.underlength_retries,
        "load_in_4bit": args.load_in_4bit,
        "strict_row_seeds": args.strict_row_seeds,
        "run_fingerprint": run_fingerprint,
        "effective_batch_size": batch_size,
    }

    def prompt_text_for(row: dict[str, Any]) -> str:
        kwargs = {"enable_thinking": False} if args.disable_thinking else {}
        try:
            return tokenizer.apply_chat_template(
                prompt_to_messages(row["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                prompt_to_messages(row["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_for_rows(rows: list[dict[str, Any]], *, seed: int) -> tuple[Any, int, float]:
        torch.manual_seed(seed)
        inputs = tokenizer(
            [prompt_text_for(row) for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        batch_started = time.perf_counter()
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
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids,
            )
        torch.cuda.synchronize()
        return output_ids, inputs["input_ids"].shape[1], time.perf_counter() - batch_started

    def attempt_from_output(
        row: dict[str, Any],
        output: Any,
        *,
        input_width: int,
        batch_seconds: float,
        all_outputs: Any,
        attempt_index: int,
        seed: int,
        batch_row_count: int,
    ) -> dict[str, Any]:
        generated_ids = output[input_width:]
        finish = generation_finish_metadata(
            generated_ids,
            tokenizer,
            args.max_new_tokens,
            configured_eos_token_ids=eos_token_ids,
        )
        raw_text, raw_text_with_special = decode_content_before_eos(generated_ids, tokenizer, finish)
        target_line_count = requested_line_count(row["prompt"]) if args.enforce_target_line_count else None
        cleaned, actions = postprocess_text(raw_text, target_line_count=target_line_count)
        timing = {
            "batch_seconds": round(batch_seconds, 2),
            "generated_tokens": finish["generated_tokens"],
            "effective_generated_tokens": finish["effective_generated_tokens"],
            "max_new_tokens": args.max_new_tokens,
            "batch_size": batch_row_count,
            "estimated_tokens_per_second": round(
                sum(int(item.shape[-1] - input_width) for item in all_outputs) / max(batch_seconds, 1e-9),
                2,
            ),
        }
        return {
            "attempt_index": attempt_index,
            "seed": seed,
            "raw_generated_text": raw_text,
            "raw_generated_text_with_special": raw_text_with_special,
            "generated_text": cleaned,
            "postprocess_actions": actions,
            "raw_line_count": count_lines(raw_text),
            "postprocessed_line_count": count_lines(cleaned),
            "target_line_count": target_line_count,
            "hit_eos": finish["hit_eos"],
            "hit_token_cap": finish["hit_token_cap"],
            "finish_reason": finish["finish_reason"],
            "first_eos_token_index": finish["first_eos_token_index"],
            "timing": timing,
        }

    underlength_retry_triggered = 0
    underlength_retry_generations = 0
    underlength_retry_resolved = 0
    underlength_retry_exhausted = 0

    with args.output_jsonl.open("a", encoding="utf-8", buffering=1) as handle:
        index = 0
        while index < len(plan):
            current = plan[index : index + batch_size]
            try:
                generation_seed = current[0]["seed"] if args.strict_row_seeds else args.seed + generated
                output_ids, input_width, batch_seconds = generate_for_rows(current, seed=generation_seed)
            except RuntimeError as exc:
                if is_oom(exc) and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    print(f"[oom-fallback] reducing sweep batch size to {batch_size}")
                    continue
                raise

            for row, output in zip(current, output_ids):
                attempts = [
                    attempt_from_output(
                        row,
                        output,
                        input_width=input_width,
                        batch_seconds=batch_seconds,
                        all_outputs=output_ids,
                        attempt_index=1,
                        seed=row["seed"],
                        batch_row_count=len(current),
                    )
                ]
                if args.underlength_retries and needs_underlength_retry(
                    attempts[-1]["generated_text"],
                    target_line_count=attempts[-1]["target_line_count"],
                ):
                    underlength_retry_triggered += 1
                    for retry_index in range(1, args.underlength_retries + 1):
                        retry_seed = row["seed"] + retry_index * 100_003
                        retry_outputs, retry_input_width, retry_seconds = generate_for_rows([row], seed=retry_seed)
                        retry_attempt = attempt_from_output(
                            row,
                            retry_outputs[0],
                            input_width=retry_input_width,
                            batch_seconds=retry_seconds,
                            all_outputs=retry_outputs,
                            attempt_index=retry_index + 1,
                            seed=retry_seed,
                            batch_row_count=1,
                        )
                        attempts.append(retry_attempt)
                        underlength_retry_generations += 1
                        if not needs_underlength_retry(
                            retry_attempt["generated_text"],
                            target_line_count=retry_attempt["target_line_count"],
                        ):
                            break

                accepted = attempts[-1]
                retry_attempts = attempts[1:]
                target_line_count = accepted["target_line_count"]
                failure_tags = structural_failure_tags(accepted["generated_text"], target_line_count=target_line_count)
                if retry_attempts and "fixable_underlength" in failure_tags:
                    underlength_retry_exhausted += 1
                elif retry_attempts:
                    underlength_retry_resolved += 1
                record = {
                    "row_id": row["row_id"],
                    "prompt_key": row["prompt_key"],
                    "candidate_index": row["candidate_index"],
                    "sample_index": row["sample_index"],
                    "prompt": row["prompt"],
                    "theme": row.get("theme"),
                    "style": row.get("style"),
                    "seed": accepted["seed"],
                    "initial_seed": row["seed"],
                    "raw_generated_text": accepted["raw_generated_text"],
                    "raw_generated_text_with_special": accepted["raw_generated_text_with_special"],
                    "generated_text": accepted["generated_text"],
                    "postprocess_applied": bool(accepted["postprocess_actions"])
                    or accepted["raw_generated_text"].strip() != accepted["generated_text"].strip(),
                    "postprocess_actions": accepted["postprocess_actions"],
                    "raw_line_count": accepted["raw_line_count"],
                    "postprocessed_line_count": accepted["postprocessed_line_count"],
                    "initial_raw_line_count": attempts[0]["raw_line_count"],
                    "initial_postprocessed_line_count": attempts[0]["postprocessed_line_count"],
                    "target_line_count": target_line_count,
                    "failure_tags": failure_tags,
                    "generation_attempt_count": len(attempts),
                    "accepted_attempt_index": accepted["attempt_index"],
                    "underlength_retry_count": len(retry_attempts),
                    "underlength_retry_exhausted": "fixable_underlength" in failure_tags and bool(retry_attempts),
                    "retry_policy": {
                        "mode": "underlength_only",
                        "max_additional_attempts": args.underlength_retries,
                    },
                    "retry_attempts": retry_attempts,
                    "hit_eos": accepted["hit_eos"],
                    "hit_token_cap": accepted["hit_token_cap"],
                    "finish_reason": accepted["finish_reason"],
                    "first_eos_token_index": accepted["first_eos_token_index"],
                    "timing": accepted["timing"],
                    "settings": settings,
                }
                attach_prompt_metadata(record, row)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1
            index += len(current)
            if generated % 25 == 0 or index >= len(plan):
                elapsed = time.perf_counter() - started
                print(
                    "[sweep] "
                    + json.dumps(
                        {
                            "generated_total": generated,
                            "remaining_this_run": len(plan) - index,
                            "elapsed_minutes": round(elapsed / 60.0, 2),
                            "batch_size": batch_size,
                            "underlength_retry_generations": underlength_retry_generations,
                        }
                    )
                )

    total_seconds = time.perf_counter() - started
    final_output_audit = inspect_output_jsonl(args.output_jsonl)
    final_ids = set(final_output_audit["row_ids"])
    missing_ids = expected_ids - final_ids
    unexpected_ids = final_ids - expected_ids
    if (
        final_output_audit["malformed_rows"]
        or final_output_audit["duplicate_row_ids"]
        or missing_ids
        or unexpected_ids
    ):
        raise RuntimeError(
            "Generation output does not exactly match the planned row-id set: "
            f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}, "
            f"duplicates={len(final_output_audit['duplicate_row_ids'])}, "
            f"malformed={len(final_output_audit['malformed_rows'])}."
        )
    summary = {
        "status": "complete",
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir) if args.adapter else None,
        "adapter_enabled": args.adapter,
        "output_jsonl": str(args.output_jsonl),
        "run_manifest": str(run_manifest_path),
        "run_fingerprint": run_fingerprint,
        "requested_candidates": args.num_candidates,
        "already_present_at_start": len(done_ids),
        "generated_this_run": len(plan),
        "total_expected_rows": args.num_candidates,
        "unique_output_rows": len(final_ids),
        "load_seconds": round(load_seconds, 2),
        "generation_wall_seconds": round(total_seconds, 2),
        "generation_wall_minutes": round(total_seconds / 60.0, 2),
        "underlength_retry_triggered": underlength_retry_triggered,
        "underlength_retry_generations": underlength_retry_generations,
        "underlength_retry_resolved": underlength_retry_resolved,
        "underlength_retry_exhausted": underlength_retry_exhausted,
        "runtime": runtime,
        "settings": settings,
        "ended_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
