from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import PRIVATE_RESEARCH_POLICY, command_record, iter_jsonl, utc_now, write_json
from .compliance import apply_line_controls
from .evaluation import lyric_lines, prompt_from_row, repeated_line_ratio, summarize_outputs, target_lines


SWEEP_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "native_default", "temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.05, "budget": "fixed", "control": False, "retry": False},
    {"name": "cool_fixed", "temperature": 0.70, "top_p": 0.90, "repetition_penalty": 1.03, "budget": "fixed", "control": False, "retry": False},
    {"name": "balanced_fixed", "temperature": 0.80, "top_p": 0.90, "repetition_penalty": 1.03, "budget": "fixed", "control": False, "retry": False},
    {"name": "dynamic_default", "temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.05, "budget": "line_adjusted", "control": False, "retry": False},
    {"name": "dynamic_cool", "temperature": 0.70, "top_p": 0.85, "repetition_penalty": 1.03, "budget": "line_adjusted", "control": False, "retry": False},
    {"name": "dynamic_repeat", "temperature": 0.80, "top_p": 0.90, "repetition_penalty": 1.06, "budget": "line_adjusted", "control": False, "retry": False},
    {"name": "dynamic_control", "temperature": 0.80, "top_p": 0.90, "repetition_penalty": 1.03, "budget": "line_adjusted", "control": True, "retry": False},
    {"name": "dynamic_control_retry", "temperature": 0.80, "top_p": 0.90, "repetition_penalty": 1.03, "budget": "line_adjusted", "control": True, "retry": True},
    {"name": "cool_control_retry", "temperature": 0.70, "top_p": 0.85, "repetition_penalty": 1.00, "budget": "line_adjusted", "control": True, "retry": True},
    {"name": "diverse_control_retry", "temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.06, "budget": "line_adjusted", "control": True, "retry": True},
)


def length_bucket(requested: int) -> str:
    if requested <= 8:
        return "short_1_8"
    if requested <= 16:
        return "medium_9_16"
    if requested <= 24:
        return "long_17_24"
    return "extra_long_25_32"


def stratified_indices(rows: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[length_bucket(target_lines(row))].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    names = sorted(buckets)
    per_bucket = count // len(names)
    remainder = count % len(names)
    for offset, name in enumerate(names):
        candidates = list(buckets[name])
        rng.shuffle(candidates)
        take = per_bucket + (1 if offset < remainder else 0)
        selected.extend(candidates[:take])
    return sorted(selected)


def token_budget(requested_lines: int, mode: str, fixed: int, maximum: int) -> int:
    if mode == "fixed":
        return fixed
    return min(maximum, max(64, requested_lines * 18))


class SweepGenerator:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
        )
        self.model.eval()
        self.model.config.use_cache = True

    def generate(
        self,
        prompts: list[str],
        budgets: list[int],
        *,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        batch_size: int,
        seed: int,
    ) -> tuple[list[str], list[int], list[int]]:
        torch = self.torch
        outputs = [""] * len(prompts)
        token_counts = [0] * len(prompts)
        maximum_positions = int(self.model.config.max_position_embeddings)
        effective_budgets = [
            max(
                1,
                min(
                    budget,
                    maximum_positions
                    - len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"]),
                ),
            )
            for prompt, budget in zip(prompts, budgets)
        ]
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, budget in enumerate(effective_budgets):
            grouped[budget].append(index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for budget, indices in sorted(grouped.items()):
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_prompts = [prompts[index] for index in batch_indices]
                encoded = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=384,
                ).to(self.model.device)
                with torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=budget,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_width = encoded["input_ids"].shape[1]
                for row_index, sequence in enumerate(generated):
                    continuation = sequence[prompt_width:]
                    output_index = batch_indices[row_index]
                    token_counts[output_index] = int(continuation.shape[0])
                    outputs[output_index] = self.tokenizer.decode(
                        continuation, skip_special_tokens=True
                    ).strip()
        return outputs, token_counts, effective_budgets


def pareto_names(results: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for candidate in results:
        metrics = candidate["system_metrics"]
        dominated = False
        for other in results:
            if other is candidate:
                continue
            comparison = other["system_metrics"]
            no_worse = (
                comparison["exact_line_match_rate"] >= metrics["exact_line_match_rate"]
                and comparison["distinct_3"] >= metrics["distinct_3"]
                and comparison["average_repeated_line_ratio"] <= metrics["average_repeated_line_ratio"]
            )
            strictly_better = (
                comparison["exact_line_match_rate"] > metrics["exact_line_match_rate"]
                or comparison["distinct_3"] > metrics["distinct_3"]
                or comparison["average_repeated_line_ratio"] < metrics["average_repeated_line_ratio"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            names.append(candidate["config"]["name"])
    return names


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    all_rows = list(iter_jsonl(Path(args.corpus_dir) / "test.jsonl"))[:500]
    count = min(args.samples, len(all_rows))
    selected_indices = stratified_indices(all_rows, count, args.seed)
    rows = [all_rows[index] for index in selected_indices]
    prompts = [prompt_from_row(row) for row in rows]
    targets = [target_lines(row) for row in rows]
    configs = SWEEP_CONFIGS[:2] if args.smoke else SWEEP_CONFIGS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exact_command.txt").write_text(" ".join(command_record()) + "\n", encoding="utf-8")
    write_json(
        output_dir / "selected_prompts.json",
        {"indices": selected_indices, "target_lines": targets, "prompts": prompts},
    )
    generator = SweepGenerator(args.model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs):
        config_started = time.monotonic()
        budgets = [
            token_budget(target, config["budget"], args.fixed_tokens, args.max_dynamic_tokens)
            for target in targets
        ]
        raw_outputs, token_counts, budgets = generator.generate(
            prompts,
            budgets,
            temperature=config["temperature"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            batch_size=args.batch_size,
            seed=args.seed + config_index,
        )
        retry_count = 0
        if config["retry"]:
            retry_indices = [
                index
                for index, (output, target) in enumerate(zip(raw_outputs, targets))
                if len(lyric_lines(output)) < target
            ]
            if retry_indices:
                retry_prompts = [prompts[index] for index in retry_indices]
                retry_budgets = [
                    min(args.retry_max_tokens, max(budgets[index] * 2, targets[index] * 24))
                    for index in retry_indices
                ]
                retry_outputs, retry_tokens, retry_budgets = generator.generate(
                    retry_prompts,
                    retry_budgets,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    repetition_penalty=config["repetition_penalty"],
                    batch_size=args.batch_size,
                    seed=args.seed + 10_000 + config_index,
                )
                retry_count = len(retry_indices)
                for local_index, output_index in enumerate(retry_indices):
                    if abs(len(lyric_lines(retry_outputs[local_index])) - targets[output_index]) < abs(
                        len(lyric_lines(raw_outputs[output_index])) - targets[output_index]
                    ):
                        raw_outputs[output_index] = retry_outputs[local_index]
                        token_counts[output_index] = retry_tokens[local_index]
                        budgets[output_index] = retry_budgets[local_index]
        system_outputs = [
            apply_line_controls(output, target) if config["control"] else output
            for output, target in zip(raw_outputs, targets)
        ]
        native_metrics = summarize_outputs(rows, raw_outputs)
        system_metrics = summarize_outputs(rows, system_outputs)
        elapsed = time.monotonic() - config_started
        generated_tokens = sum(token_counts)
        result = {
            "config": config,
            "native_metrics": native_metrics,
            "system_metrics": system_metrics,
            "truncation_rate": sum(
                count >= budget - 1 for count, budget in zip(token_counts, budgets)
            ) / len(rows),
            "average_native_lines": sum(len(lyric_lines(output)) for output in raw_outputs) / len(rows),
            "retry_count": retry_count,
            "wall_seconds": round(elapsed, 3),
            "generated_tokens": generated_tokens,
            "tokens_per_second": round(generated_tokens / max(elapsed, 1e-9), 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 6) if torch.cuda.is_available() else 0.0,
            "outputs": system_outputs,
        }
        results.append(result)
        write_json(output_dir / f"{config['name']}.json", result)
    pareto = pareto_names(results)
    baseline = next(result for result in results if result["config"]["name"] == "native_default")
    quality_eligible = [
        result
        for result in results
        if result["config"]["name"] in pareto
        and result["system_metrics"]["average_repeated_line_ratio"]
        <= baseline["system_metrics"]["average_repeated_line_ratio"] * 1.10
        and result["system_metrics"]["distinct_3"]
        >= baseline["system_metrics"]["distinct_3"] * 0.90
    ]
    selection_pool = quality_eligible or [
        result for result in results if result["config"]["name"] in pareto
    ]
    selected = max(
        selection_pool,
        key=lambda result: (
            result["system_metrics"]["exact_line_match_rate"],
            result["system_metrics"]["distinct_3"],
            -result["system_metrics"]["average_repeated_line_ratio"],
        ),
    )
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "model": args.model,
        "sample_count": len(rows),
        "strata": dict(sorted(defaultdict(int, {name: sum(length_bucket(target) == name for target in targets) for name in {length_bucket(target) for target in targets}}).items())),
        "results": [{key: value for key, value in result.items() if key != "outputs"} for result in results],
        "pareto_efficient_configs": pareto,
        "quality_eligible_pareto_configs": [result["config"]["name"] for result in quality_eligible],
        "selection_rule": {
            "maximum_repetition_ratio": baseline["system_metrics"]["average_repeated_line_ratio"] * 1.10,
            "minimum_distinct_3": baseline["system_metrics"]["distinct_3"] * 0.90,
            "then_rank_by": ["system_exact_line_match_rate", "distinct_3", "lower_repetition"],
        },
        "recommended_config": selected["config"]["name"],
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output_dir / "sweep_report.json", report)
    return report


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fixed-tokens", type=int, default=256)
    parser.add_argument("--max-dynamic-tokens", type=int, default=576)
    parser.add_argument("--retry-max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--smoke", action="store_true")
