from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original_500, filtered_500_probe, and DPO_control_500 outputs."
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[
            "original_500=model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-384-500-simple/checkpoint-500",
            "filtered_500=model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-filtered-384-500-probe/checkpoint-500",
            "dpo_500=model/artifacts/qwen2.5-7b-rap-lora-mixed-sft-500-dpo-control-probe/adapter",
        ],
        help="Repeatable label=adapter_path arguments.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base model name.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Write a 16-line rap verse about ambition after failure. Include internal rhymes.",
            "Write a gritty storytelling verse about a late-night train ride.",
            "Write a catchy hook about loyalty and pressure.",
            "Write a battle rap verse with clever punchlines but no slurs.",
            "Write a reflective verse about success feeling lonely.",
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.78)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--output-jsonl",
        default="reports/dpo_control_eval.jsonl",
        help="Per-prompt output rows.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/dpo_control_eval.md",
        help="Comparison report.",
    )
    return parser.parse_args()


def _load_tokenizer(base_model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model_for_eval(base_model: str, adapter_path: str, quant_cfg: BitsAndBytesConfig):
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    if not adapter_path:
        return base

    try:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    except TypeError:
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
    return model


def _line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _line_endings(text: str) -> list[str]:
    lines = [ln.strip().lower().rstrip(" !?,.") for ln in text.splitlines() if ln.strip()]
    return [ln.split()[-1] if ln.split() else "" for ln in lines]


def _rhyme_density(text: str) -> float:
    endings = _line_endings(text)
    if len(endings) < 2:
        return 0.0
    matches = 0
    for prev, nxt in zip(endings, endings[1:]):
        if prev and nxt and prev[-2:] == nxt[-2:]:
            matches += 1
    return round(matches / max(len(endings) - 1, 1), 3)


def _quality_tags(text: str) -> list[str]:
    lower = text.lower()
    tags: list[str] = []
    if re.search(r"\b(haha|ha+|lol|lmao)\b", lower):
        tags.append("reject_laughter_drift")
    if re.search(r"\b(\".*\"|'.*')\b", lower):
        tags.append("reject_dialogue_drift")
    if _line_count(text) < 8:
        tags.append("reject_line_count_failure")
    if re.search(r"\b(blood|shoot|murder|kill|stab|weapon)\b", lower):
        tags.append("reject_violent_derailment")
    if not lower.strip().endswith((".", "?", "!", "...")):
        tags.append("reject_incomplete_ending")
    return tags


def _score_control(text: str) -> int:
    score = 100
    lower = text.lower()
    if "laugh" in lower or re.search(r"\b(haha|ha+|lol|lmao)\b", lower):
        score -= 12
    if re.search(r"\b(\"|')\b", lower) and re.search(r"\b(yo|man|bro|uh|yeah|ain't|gonna)\b", lower):
        score -= 10
    if re.search(r"\b(blood|shoot|murder|kill|stab|weapon)\b", lower):
        score -= 8
    if _line_count(text) < 8:
        score -= 12
    if _line_count(text) > 34:
        score -= 6
    if not lower.strip().endswith((".", "?", "!", "...")):
        score -= 8
    return max(score, 0)


def _score_creative(text: str) -> int:
    score = 50
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0
    score += min(len(lines), 20) * 2
    avg_words = sum(len(line.split()) for line in lines) / max(len(lines), 1)
    score += min(int(avg_words), 10)
    score += int(_rhyme_density(text) * 20)
    if re.search(r"\b(metro|city|rain|street|lights|window|train|pressure|loyalty|ambition|lonely|echo|midnight)\b", text.lower()):
        score += 8
    if any(word in text.lower() for word in ("i'm", " i am ", "we ", "you ", "you'll", "we'll", "they'll")):
        score += 6
    return min(score, 100)


def _generate_one(model, tokenizer, prompt: str, seed: int, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        start = time.perf_counter()
        output = model.generate(
            **input_ids,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=50,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        elapsed = time.perf_counter() - start

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    gen_text = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    gen_text = gen_text.strip()
    gen_tokens = max(output.shape[1] - input_ids["input_ids"].shape[1], 0)
    tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0.0
    tags = _quality_tags(gen_text)
    return {
        "prompt": prompt,
        "output": gen_text,
        "gen_seconds": round(elapsed, 4),
        "token_count": int(gen_tokens),
        "tokens_per_sec": round(tokens_per_sec, 3),
        "line_count": _line_count(gen_text),
        "rhyme_density": _rhyme_density(gen_text),
        "quality_tags": tags,
        "control_score": _score_control(gen_text),
        "creative_score": _score_creative(gen_text),
        "seed": seed,
    }


def _model_label_and_path(item: str):
    if "=" not in item:
        raise ValueError(f"Bad --models item '{item}'. Expected label=path.")
    label, path = item.split("=", 1)
    return label.strip(), path.strip()


def main() -> None:
    args = parse_args()
    model_entries = [_model_label_and_path(item) for item in args.models]
    model_map = {k: v for k, v in model_entries}
    tokenizer = _load_tokenizer(args.base_model)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    rng = random.Random(args.seed)
    records: list[dict] = []
    aggregate: dict[str, list[dict]] = defaultdict(list)

    for label, adapter_path in model_entries:
        load_start = time.perf_counter()
        model = _load_model_for_eval(args.base_model, adapter_path, bnb_cfg)
        load_sec = round(time.perf_counter() - load_start, 3)

        for index, prompt in enumerate(args.prompts):
            seed = args.seed + (index * 17) + len(label)
            rng.seed(seed)
            row = _generate_one(model, tokenizer, prompt, seed, args)
            row.update({
                "model_label": label,
                "model_path": model_map[label],
                "prompt_index": index,
                "seed": seed,
                "load_seconds": load_sec,
            })
            records.append(row)
            aggregate[label].append(row)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped = {}
    for label, rows in aggregate.items():
        total = len(rows)
        grouped[label] = {
            "rows": total,
            "avg_control": round(sum(row["control_score"] for row in rows) / total, 2) if total else 0,
            "avg_creative": round(sum(row["creative_score"] for row in rows) / total, 2) if total else 0,
            "avg_lines": round(sum(row["line_count"] for row in rows) / total, 2) if total else 0,
            "avg_tokens_per_sec": round(sum(row["tokens_per_sec"] for row in rows) / total, 3) if total else 0,
            "avg_rhyme_density": round(sum(row["rhyme_density"] for row in rows) / total, 3) if total else 0,
            "fail_rates": dict(Counter(tag for row in rows for tag in row["quality_tags"])),
            "avg_gen_seconds": round(sum(row["gen_seconds"] for row in rows) / total, 3) if total else 0,
        }

    output_md = Path(args.output_md)
    with output_md.open("w", encoding="utf-8") as handle:
        handle.write("# DPO Control Probe Eval\n\n")
        handle.write("## Settings\n\n")
        handle.write(f"- base_model: `{args.base_model}`\n")
        handle.write(f"- prompts: `{len(args.prompts)}`\n")
        handle.write(f"- max_new_tokens: `{args.max_new_tokens}`\n")
        handle.write(f"- temperature: `{args.temperature}`\n")
        handle.write(f"- top_p: `{args.top_p}`\n")
        handle.write(f"- repetition_penalty: `{args.repetition_penalty}`\n")
        handle.write(f"- seed: `{args.seed}`\n\n")

        handle.write("## Aggregate Scores\n\n")
        for label, summary in grouped.items():
            handle.write(f"### {label}\n\n")
            handle.write(f"- control_score: `{summary['avg_control']}`\n")
            handle.write(f"- creative_score: `{summary['avg_creative']}`\n")
            handle.write(f"- avg_lines: `{summary['avg_lines']}`\n")
            handle.write(f"- avg_tokens_per_sec: `{summary['avg_tokens_per_sec']}`\n")
            handle.write(f"- avg_rhyme_density: `{summary['avg_rhyme_density']}`\n")
            handle.write(f"- avg_gen_seconds: `{summary['avg_gen_seconds']}`\n")
            failures = ", ".join(f"{name}:{count}" for name, count in sorted(summary["fail_rates"].items()))
            handle.write(f"- fail_counts: {failures or 'none'}\n\n")

        handle.write("## Recommendations\n\n")
        handle.write("- Prefer models with higher control score without a clear drop in creative score.\n")
        handle.write("- Watch incomplete endings and dialogue/laughter drift as hard failures.\n")

        handle.write("## Rows\n\n")
        for row in records:
            tags = ", ".join(row["quality_tags"]) or "none"
            handle.write(
                f"- **{row['model_label']}** prompt {row['prompt_index']} | "
                f"control `{row['control_score']}` | creative `{row['creative_score']}` | "
                f"lines `{row['line_count']}` | tags `{tags}`\n"
            )
            handle.write(f"  - `{row['output'][:240].replace('`', '')}`\n\n")

    summary_path = output_md.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "base_model": args.base_model,
                    "prompts": args.prompts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed,
                },
                "aggregate": grouped,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote generation JSONL -> {output_jsonl}")
    print(f"Wrote comparison report -> {output_md}")
    print(f"Wrote comparison summary -> {summary_path}")


if __name__ == "__main__":
    main()
