from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .common import PRIVATE_RESEARCH_POLICY, command_record, read_json, utc_now, write_json
from .compliance import apply_line_controls
from .evaluation import lyric_lines
from .sweep import SweepGenerator, token_budget


ALLOWED_SECTIONS = {"verse", "hook", "chorus", "bridge", "intro", "outro"}


def structured_prompt(request: dict[str, Any]) -> str:
    title = str(request.get("title") or "Untitled").strip()
    section = str(request.get("section") or "verse").strip().lower()
    if section not in ALLOWED_SECTIONS:
        raise ValueError(f"Unsupported section: {section}")
    lines = int(request.get("target_lines") or request.get("lines") or 12)
    if lines < 1 or lines > 32:
        raise ValueError("target_lines must be between 1 and 32")
    content = str(request.get("content_policy") or "clean").lower()
    content_token = "<|content_explicit|>" if content == "explicit" else "<|content_clean|>"
    year = str(request.get("year") or "<|year_unknown|>")
    return (
        f"<|bos|><|task_generate|><|title|>{title}\n<|year|>{year}\n"
        f"<|section|>{section}\n<|target_lines|>{lines}\n{content_token}<|lyrics|>\n"
    )


def generate_controlled(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = read_json(Path(args.config))
    request_payload = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    requests = request_payload if isinstance(request_payload, list) else request_payload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Requests must be a non-empty JSON list or an object containing a requests list.")
    prompts = [structured_prompt(request) for request in requests]
    targets = [int(request.get("target_lines") or request.get("lines") or 12) for request in requests]
    decoding = config["decoding"]
    budget_config = config["token_budget"]
    retry_config = config["retry"]
    generator = SweepGenerator(config["model_release"])
    requested_budgets = [
        token_budget(
            target,
            "line_adjusted",
            int(budget_config["minimum_new_tokens"]),
            int(budget_config["configured_maximum_new_tokens"]),
        )
        for target in targets
    ]
    raw_outputs, generated_tokens, effective_budgets = generator.generate(
        prompts,
        requested_budgets,
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        repetition_penalty=float(decoding["repetition_penalty"]),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    retry_indices = [
        index
        for index, (output, target) in enumerate(zip(raw_outputs, targets))
        if len(lyric_lines(output)) < target
    ]
    retry_accepted: set[int] = set()
    if retry_indices and int(retry_config["underlength_retries"]) > 0:
        retry_prompts = [prompts[index] for index in retry_indices]
        retry_requested_budgets = [
            min(
                int(retry_config["configured_maximum_new_tokens"]),
                max(
                    effective_budgets[index] * 2,
                    targets[index] * int(retry_config["tokens_per_requested_line"]),
                ),
            )
            for index in retry_indices
        ]
        retry_outputs, retry_tokens, retry_effective_budgets = generator.generate(
            retry_prompts,
            retry_requested_budgets,
            temperature=float(decoding["temperature"]),
            top_p=float(decoding["top_p"]),
            repetition_penalty=float(decoding["repetition_penalty"]),
            batch_size=args.batch_size,
            seed=args.seed + 10_000,
        )
        for local_index, output_index in enumerate(retry_indices):
            old_error = abs(len(lyric_lines(raw_outputs[output_index])) - targets[output_index])
            new_error = abs(len(lyric_lines(retry_outputs[local_index])) - targets[output_index])
            if new_error < old_error:
                raw_outputs[output_index] = retry_outputs[local_index]
                generated_tokens[output_index] = retry_tokens[local_index]
                effective_budgets[output_index] = retry_effective_budgets[local_index]
                retry_accepted.add(output_index)
    records: list[dict[str, Any]] = []
    for index, (request, prompt, output, target) in enumerate(zip(requests, prompts, raw_outputs, targets)):
        controlled = apply_line_controls(output, target)
        records.append(
            {
                "request_index": index,
                "request": request,
                "prompt": prompt,
                "raw_output": output,
                "output": controlled,
                "native_line_count": len(lyric_lines(output)),
                "system_line_count": len(lyric_lines(controlled)),
                "target_lines": target,
                "native_exact": len(lyric_lines(output)) == target,
                "system_exact": len(lyric_lines(controlled)) == target,
                "retry_attempted": index in retry_indices,
                "retry_accepted": index in retry_accepted,
                "generated_tokens": generated_tokens[index],
                "effective_token_budget": effective_budgets[index],
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "system_name": config["system_name"],
        "generated_at": utc_now(),
        "command": command_record(),
        "config": str(args.config),
        "sample_count": len(records),
        "native_exact_rate": sum(record["native_exact"] for record in records) / len(records),
        "system_exact_rate": sum(record["system_exact"] for record in records) / len(records),
        "retry_attempted": len(retry_indices),
        "retry_accepted": len(retry_accepted),
        "wall_seconds": round(time.monotonic() - started, 3),
        "records": records,
    }
    (output_dir / "exact_command.txt").write_text(" ".join(command_record()) + "\n", encoding="utf-8")
    write_json(output_dir / "generation_report.json", report)
    return report


def add_router_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/scratch_30m_sft_v1_1_control.json"),
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
