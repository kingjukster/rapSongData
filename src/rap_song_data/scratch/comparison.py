from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    PRIVATE_RESEARCH_POLICY,
    command_record,
    hash_file,
    iter_jsonl,
    read_json,
    utc_now,
    write_json,
)
from .compliance import apply_line_controls
from .evaluation import distinct_n, lyric_lines, repeated_line_ratio
from .router import structured_prompt
from .sweep import SweepGenerator, token_budget


FAMILIES: dict[str, dict[str, Any]] = {
    "melodic": {
        "section": "hook",
        "requests": [
            ("Neon After Rain", "finding resolve while city lights return after a storm"),
            ("Last Train Glow", "choosing hope on the final train home"),
            ("Summer on the Fire Escape", "a warm night shared above a restless block"),
            ("Signal Through Static", "holding onto a relationship through distance and doubt"),
        ],
    },
    "story": {
        "section": "verse",
        "requests": [
            ("The Missing Envelope", "a courier discovers an important envelope is missing and retraces the route"),
            ("Corner Store Eclipse", "two old friends reunite during a sudden blackout at a corner store"),
            ("Borrowed Bicycle", "a teenager borrows a bicycle, gets lost, and returns with a hard-earned lesson"),
            ("Keys Under the Bleachers", "a groundskeeper finds keys after a night game and identifies their owner"),
        ],
    },
    "technical": {
        "section": "verse",
        "requests": [
            ("Clockwork Syntax", "precision, timing, and layered internal rhyme without losing meaning"),
            ("Blueprint in Motion", "engineering a better future through systems, measurement, and revision"),
            ("Cipher Telescope", "using astronomy and code imagery in a coherent display of skill"),
            ("Pressure Algorithm", "solving difficult problems calmly while the stakes keep rising"),
        ],
    },
    "clean": {
        "section": "verse",
        "requests": [
            ("Saturday Gym Lights", "a youth team practicing early and learning to trust one another"),
            ("Library Card", "discovering new worlds through a neighborhood library"),
            ("Garden on the Roof", "neighbors turning an empty rooftop into a shared garden"),
            ("First Day Route", "a new bus driver learning the route and helping passengers get home"),
        ],
    },
}

LENGTHS = (4, 8, 16, 32)
DIFFICULTIES = ("standard", "moderate", "hard", "hard")
STRUCTURAL_RISKS = ("short_stop", "normal", "long_progression", "long_progression_and_stop")
LINE_LENGTHS = ("short", "medium", "long", "mixed")


def _qwen_prompt(family: str, target: int, semantic_request: str) -> str:
    family_direction = {
        "melodic": "Use a memorable melodic cadence and a hook-like through-line without copying lines.",
        "story": "Make every line advance a clear scene, action, or consequence.",
        "technical": "Use layered internal rhyme and precise imagery while keeping the meaning coherent.",
        "clean": "Use clean, radio-safe language with no profanity or slurs.",
    }[family]
    return (
        f"Write exactly {target} lines of original rap lyrics about {semantic_request}. "
        f"{family_direction} End naturally on line {target}. "
        "Return lyrics only: no title, section label, numbering, blank-line commentary, or explanation."
    )


def prompt_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, spec in FAMILIES.items():
        for target in LENGTHS:
            for variant, (title, semantic_request) in enumerate(spec["requests"], start=1):
                prompt_id = f"{family}-{target:02d}-{variant}"
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "theme_id": prompt_id,
                        "prompt_family": family,
                        "family": family,
                        "target_lines": target,
                        "difficulty": DIFFICULTIES[variant - 1],
                        "structural_risk": STRUCTURAL_RISKS[LENGTHS.index(target)],
                        "expected_line_length": LINE_LENGTHS[variant - 1],
                        "title": title,
                        "section": spec["section"],
                        "content_policy": "clean" if family == "clean" else "explicit",
                        "semantic_request": semantic_request,
                        "prompt": _qwen_prompt(family, target, semantic_request),
                    }
                )
    return rows


def validate_prompt_matrix(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 64:
        raise ValueError(f"Expected 64 prompts, found {len(rows)}")
    ids = [str(row["prompt_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt ids are not unique")
    for family in FAMILIES:
        for target in LENGTHS:
            count = sum(row["family"] == family and int(row["target_lines"]) == target for row in rows)
            if count != 4:
                raise ValueError(f"Expected four prompts for {family}/{target}, found {count}")


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = prompt_matrix()
    validate_prompt_matrix(rows)
    matrix_path = root / "prompt_matrix.json"
    qwen_path = root / "qwen_prompts.json"
    smoke_path = root / "qwen_prompts_smoke.json"
    write_json(matrix_path, rows)
    write_json(qwen_path, rows)
    smoke_ids = {"melodic-04-1", "story-08-2", "technical-16-3", "clean-32-4"}
    write_json(smoke_path, [row for row in rows if row["prompt_id"] in smoke_ids])

    scratch_release = Path(args.scratch_release)
    router_config = Path(args.router_config)
    qwen_runner = Path(args.qwen_runner)
    release_manifest = scratch_release / "release_manifest.json"
    definitions = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "frozen_for_automated_comparison",
        "frozen_at": utc_now(),
        "seed": int(args.seed),
        "prompt_matrix": {"path": str(matrix_path), "sha256": hash_file(matrix_path)},
        "systems": {
            "scratch_native_v1": {
                "model_release": str(scratch_release),
                "release_manifest_sha256": hash_file(release_manifest),
                "prompt_protocol": "scratch-structured-v1",
                "temperature": 0.9,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "token_budget": "18_per_requested_line_context_capped",
                "line_control": False,
                "retry": False,
            },
            "scratch_router_v1_1": {
                "model_release": str(scratch_release),
                "release_manifest_sha256": hash_file(release_manifest),
                "router_config": str(router_config),
                "router_config_sha256": hash_file(router_config),
            },
            "qwen_production_base_v1": {
                "base_model": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "adapter_enabled": False,
                "runner": str(qwen_runner),
                "runner_sha256": hash_file(qwen_runner),
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "max_new_tokens": 640,
                "load_in_4bit": True,
                "disable_thinking": True,
                "block_slurs": True,
                "enforce_target_line_count": True,
                "underlength_retries": 2,
                "strict_row_seeds": True,
            },
        },
    }
    definitions_path = root / "frozen_systems.json"
    write_json(definitions_path, definitions)
    result = {
        **PRIVATE_RESEARCH_POLICY,
        "status": "ready",
        "prompt_count": len(rows),
        "candidate_count": len(rows) * 3,
        "prompt_matrix": str(matrix_path),
        "qwen_prompts": str(qwen_path),
        "qwen_smoke_prompts": str(smoke_path),
        "frozen_systems": str(definitions_path),
    }
    write_json(root / "build_summary.json", result)
    return result


def _scratch_records(
    generator: SweepGenerator,
    rows: list[dict[str, Any]],
    *,
    system: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    control: bool,
    retry: bool,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [structured_prompt(row) for row in rows]
    targets = [int(row["target_lines"]) for row in rows]
    requested = [token_budget(target, "line_adjusted", 64, 576) for target in targets]
    started = time.monotonic()
    raw, token_counts, budgets = generator.generate(
        prompts,
        requested,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        batch_size=batch_size,
        seed=seed,
    )
    retry_attempted: set[int] = set()
    retry_accepted: set[int] = set()
    if retry:
        retry_indices = [index for index, (text, target) in enumerate(zip(raw, targets)) if len(lyric_lines(text)) < target]
        retry_attempted.update(retry_indices)
        if retry_indices:
            retry_outputs, retry_tokens, retry_budgets = generator.generate(
                [prompts[index] for index in retry_indices],
                [min(768, max(budgets[index] * 2, targets[index] * 24)) for index in retry_indices],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                seed=seed + 10_000,
            )
            for local, index in enumerate(retry_indices):
                old_error = abs(len(lyric_lines(raw[index])) - targets[index])
                new_error = abs(len(lyric_lines(retry_outputs[local])) - targets[index])
                if new_error < old_error:
                    raw[index] = retry_outputs[local]
                    token_counts[index] = retry_tokens[local]
                    budgets[index] = retry_budgets[local]
                    retry_accepted.add(index)
    records: list[dict[str, Any]] = []
    for index, (row, text, target) in enumerate(zip(rows, raw, targets)):
        output = apply_line_controls(text, target) if control else text
        records.append(
            {
                "prompt_id": row["prompt_id"],
                "system": system,
                "raw_output": text,
                "output": output,
                "target_lines": target,
                "native_line_count": len(lyric_lines(text)),
                "line_count": len(lyric_lines(output)),
                "native_exact": len(lyric_lines(text)) == target,
                "exact": len(lyric_lines(output)) == target,
                "retry_attempted": index in retry_attempted,
                "retry_accepted": index in retry_accepted,
                "control_applied": control and output != text,
                "generated_tokens": token_counts[index],
                "effective_token_budget": budgets[index],
            }
        )
    elapsed = time.monotonic() - started
    return records, {
        "wall_seconds": round(elapsed, 3),
        "generated_tokens": sum(token_counts),
        "tokens_per_second": round(sum(token_counts) / max(elapsed, 1e-9), 3),
        "retry_attempted": len(retry_attempted),
        "retry_accepted": len(retry_accepted),
    }


def generate_scratch_comparison(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_json(Path(args.prompt_matrix))
    validate_prompt_matrix(rows if not args.smoke else prompt_matrix())
    if args.smoke:
        smoke_ids = {"melodic-04-1", "story-08-2", "technical-16-3", "clean-32-4"}
        rows = [row for row in rows if row["prompt_id"] in smoke_ids]
    generator = SweepGenerator(str(args.model))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    native, native_runtime = _scratch_records(
        generator, rows, system="scratch_native_v1", temperature=0.9, top_p=0.95,
        repetition_penalty=1.05, control=False, retry=False,
        batch_size=args.batch_size, seed=args.seed,
    )
    router, router_runtime = _scratch_records(
        generator, rows, system="scratch_router_v1_1", temperature=0.9, top_p=0.95,
        repetition_penalty=1.06, control=True, retry=True,
        batch_size=args.batch_size, seed=args.seed + 1,
    )
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "complete",
        "generated_at": utc_now(),
        "command": command_record(),
        "smoke": bool(args.smoke),
        "sample_count_per_system": len(rows),
        "native_runtime": native_runtime,
        "router_runtime": router_runtime,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 6) if torch.cuda.is_available() else 0.0,
        "records": native + router,
    }
    write_json(output_dir / "scratch_generations.json", report)
    (output_dir / "exact_command.txt").write_text(" ".join(command_record()) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "records"}


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [record["output"] for record in records]
    return {
        "count": len(records),
        "exact_line_rate": sum(bool(record["exact"]) for record in records) / max(1, len(records)),
        "average_repeated_line_ratio": sum(repeated_line_ratio(text) for text in outputs) / max(1, len(outputs)),
        "distinct_1": distinct_n(outputs, 1),
        "distinct_2": distinct_n(outputs, 2),
        "distinct_3": distinct_n(outputs, 3),
        "retry_rate": sum(bool(record.get("retry_attempted")) for record in records) / max(1, len(records)),
        "control_change_rate": sum(bool(record.get("control_applied")) for record in records) / max(1, len(records)),
    }


def build_blind_comparison(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = read_json(Path(args.prompt_matrix))
    validate_prompt_matrix(rows)
    by_prompt = {row["prompt_id"]: row for row in rows}
    scratch = read_json(Path(args.scratch_generations))["records"]
    all_records = list(scratch)
    for row in iter_jsonl(Path(args.qwen_generations)):
        prompt_id = str(row.get("theme_id") or "")
        if prompt_id not in by_prompt:
            matches = [item["prompt_id"] for item in rows if item["prompt"] == row.get("prompt")]
            if len(matches) != 1:
                raise ValueError(f"Could not match Qwen row {row.get('row_id')}")
            prompt_id = matches[0]
        target = int(by_prompt[prompt_id]["target_lines"])
        output = str(row["generated_text"])
        all_records.append(
            {
                "prompt_id": prompt_id,
                "system": "qwen_production_base_v1",
                "raw_output": row.get("raw_generated_text", output),
                "output": output,
                "target_lines": target,
                "line_count": len(lyric_lines(output)),
                "exact": len(lyric_lines(output)) == target,
                "retry_attempted": int(row.get("underlength_retry_count", 0)) > 0,
                "retry_accepted": int(row.get("accepted_attempt_index", 1)) > 1,
                "control_applied": bool(row.get("postprocess_applied")),
            }
        )
    systems = ("scratch_native_v1", "scratch_router_v1_1", "qwen_production_base_v1")
    record_map = {(row["prompt_id"], row["system"]): row for row in all_records}
    missing = [(pid, system) for pid in by_prompt for system in systems if (pid, system) not in record_map]
    if missing:
        raise ValueError(f"Missing {len(missing)} prompt/system outputs; first: {missing[0]}")

    rng = random.Random(args.seed)
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    labels = ("a", "b", "c")
    for prompt in rows:
        shuffled = list(systems)
        rng.shuffle(shuffled)
        packet: dict[str, Any] = {
            key: prompt[key]
            for key in ("prompt_id", "family", "target_lines", "difficulty", "structural_risk", "expected_line_length", "semantic_request")
        }
        key_row: dict[str, Any] = {"prompt_id": prompt["prompt_id"], "mapping": {}}
        for label, system in zip(labels, shuffled):
            record = record_map[(prompt["prompt_id"], system)]
            packet[f"candidate_{label}"] = record["output"]
            key_row["mapping"][label] = {
                "system": system,
                "actual_retry_attempted": bool(record.get("retry_attempted")),
                "actual_control_applied": bool(record.get("control_applied")),
            }
        packet_rows.append(packet)
        key_rows.append(key_row)

    packet_path = root / "candidate_packet.csv"
    with packet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(packet_rows[0]))
        writer.writeheader()
        writer.writerows(packet_rows)
    write_json(root / "private_identity_key.json", {**PRIVATE_RESEARCH_POLICY, "rows": key_rows})

    system_metrics: dict[str, Any] = {}
    slice_metrics: dict[str, Any] = {}
    for system in systems:
        records = [record for record in all_records if record["system"] == system]
        system_metrics[system] = _metric_summary(records)
        slice_metrics[system] = {
            "by_length": {
                str(length): _metric_summary([record for record in records if int(record["target_lines"]) == length])
                for length in LENGTHS
            },
            "by_family": {
                family: _metric_summary([record for record in records if by_prompt[record["prompt_id"]]["family"] == family])
                for family in FAMILIES
            },
        }
    uplift = {"by_length": {}, "by_family": {}}
    for dimension, values in (("by_length", map(str, LENGTHS)), ("by_family", FAMILIES)):
        for value in values:
            native = slice_metrics["scratch_native_v1"][dimension][value]
            router = slice_metrics["scratch_router_v1_1"][dimension][value]
            uplift[dimension][value] = {
                "compliance_delta": router["exact_line_rate"] - native["exact_line_rate"],
                "repetition_delta": router["average_repeated_line_ratio"] - native["average_repeated_line_ratio"],
                "distinct_3_delta": router["distinct_3"] - native["distinct_3"],
                "quality_delta": "deprecated_human_review_removed",
            }
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "automated_comparison_complete",
        "human_review_deprecated": True,
        "generated_at": utc_now(),
        "candidate_count": len(all_records),
        "system_metrics": system_metrics,
        "slice_metrics": slice_metrics,
        "router_uplift": uplift,
        "candidate_packet": str(packet_path),
        "private_identity_key": str(root / "private_identity_key.json"),
    }
    write_json(root / "automated_comparison.json", report)
    return report


def add_build_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-release", type=Path, default=Path("model/releases/scratch-30m-sft-v1"))
    parser.add_argument("--router-config", type=Path, default=Path("configs/generation/scratch_30m_sft_v1_1_control.json"))
    parser.add_argument("--qwen-runner", type=Path, default=Path("scripts/run_qwen3_generation_sweep.py"))
    parser.add_argument("--seed", type=int, default=20260713)


def add_scratch_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-matrix", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model/releases/scratch-30m-sft-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--smoke", action="store_true")


def add_blind_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-matrix", type=Path, required=True)
    parser.add_argument("--scratch-generations", type=Path, required=True)
    parser.add_argument("--qwen-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
