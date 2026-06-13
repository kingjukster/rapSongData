"""Generate lyrics locally with the saved LoRA adapter.

This loads the base model plus the adapter saved under
`model/artifacts/rap-lyrics-lora-adapter`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = Path("model/artifacts/rap-lyrics-lora-adapter")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--title", default="Untitled")
    parser.add_argument("--artist", default="Original")
    parser.add_argument("--rap-family", default="Melodic / Emo / Cloud")
    parser.add_argument("--rap-category", default="Emo Rap / Melodic Rap")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--views-log", default="0.0000")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.22)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
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
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]

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
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    torch.manual_seed(args.seed)
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    if not use_cuda:
        model.to("cpu")

    prompt = build_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt")
    if use_cuda:
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bad_words_ids=blocked_phrase_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if args.clean_output:
        text = clean_generated_lyrics(text)
    print(text)


if __name__ == "__main__":
    main()
