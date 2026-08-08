#!/usr/bin/env python3
"""Prepare deterministic family-routed prompt subsets and combine their outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_router(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    family_to_route: dict[str, str] = {}
    for route, details in config["routes"].items():
        for family in details["prompt_families"]:
            if family in family_to_route:
                raise ValueError(f"Prompt family {family} is assigned to multiple routes")
            family_to_route[family] = route
    config["family_to_route"] = family_to_route
    return config


def prepare(config_path: Path, prompts_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_router(config_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list):
        raise ValueError("Prompt file must contain a JSON list")
    grouped = {route: [] for route in config["routes"]}
    seen_prompts: set[str] = set()
    for original_index, row in enumerate(prompts, start=1):
        family = row.get("prompt_family")
        route = config["family_to_route"].get(family)
        if route is None:
            raise ValueError(f"No route for prompt family {family!r}")
        prompt = str(row.get("prompt") or "").strip()
        if not prompt or prompt in seen_prompts:
            raise ValueError("Prompts must be non-empty and unique")
        seen_prompts.add(prompt)
        routed = dict(row)
        routed["generation_prompt_index"] = original_index
        routed["router_route"] = route
        grouped[route].append(routed)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_outputs: dict[str, Any] = {}
    for route, rows in grouped.items():
        path = output_dir / f"prompts_{route}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        route_outputs[route] = {
            "path": str(path),
            "rows": len(rows),
            "adapter_dir": config["routes"][route]["adapter_dir"],
            "prompt_families": config["routes"][route]["prompt_families"],
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "router": str(config_path),
        "router_sha256": sha256_file(config_path),
        "source_prompts": str(prompts_path),
        "source_prompts_sha256": sha256_file(prompts_path),
        "source_prompt_count": len(prompts),
        "routed_prompt_count": sum(item["rows"] for item in route_outputs.values()),
        "routes": route_outputs,
    }
    (output_dir / "routing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--input values must use route=path")
    route, path = value.split("=", 1)
    return route, Path(path)


def select_route_outputs(
    config_path: Path,
    prompts_path: Path,
    route: str,
    input_jsonl: Path,
    output_jsonl: Path,
    base_seed: int,
) -> dict[str, Any]:
    config = load_router(config_path)
    if route not in config["routes"]:
        raise ValueError(f"Unknown route: {route}")
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    expected = {
        str(row["prompt"]): (index, row["prompt_family"])
        for index, row in enumerate(prompts, start=1)
        if config["family_to_route"].get(row["prompt_family"]) == route
    }
    selected: list[dict[str, Any]] = []
    for row in read_jsonl(input_jsonl):
        prompt = str(row.get("prompt"))
        if prompt not in expected:
            continue
        original_index, _ = expected[prompt]
        expected_seed = base_seed + (original_index - 1) * 100_000
        if int(row.get("seed")) != expected_seed:
            raise ValueError(f"Seed mismatch for routed replay prompt {original_index}")
        selected.append(row)
    if len(selected) != len(expected) or len({row["prompt"] for row in selected}) != len(expected):
        raise ValueError(f"Replay did not exactly cover route {route}: {len(selected)} != {len(expected)}")
    selected.sort(key=lambda row: expected[str(row["prompt"])][0])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "route": route,
        "source": str(input_jsonl),
        "source_sha256": sha256_file(input_jsonl),
        "output": str(output_jsonl),
        "output_sha256": sha256_file(output_jsonl),
        "rows": len(selected),
        "seed_preservation": True,
    }


def combine(
    config_path: Path,
    prompts_path: Path,
    inputs: list[str],
    output_jsonl: Path,
    summary_path: Path,
    base_seed: int,
) -> dict[str, Any]:
    config = load_router(config_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    expected = {str(row["prompt"]): index for index, row in enumerate(prompts, start=1)}
    by_prompt: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {}
    for value in inputs:
        route, path = parse_input(value)
        if route not in config["routes"]:
            raise ValueError(f"Unknown route: {route}")
        rows = read_jsonl(path)
        for row in rows:
            prompt = str(row.get("prompt"))
            if prompt not in expected:
                raise ValueError(f"Unexpected prompt in {path}: {prompt}")
            family = prompts[expected[prompt] - 1]["prompt_family"]
            if config["family_to_route"].get(family) != route:
                raise ValueError(f"Prompt was generated by wrong route: {prompt}")
            if prompt in by_prompt:
                raise ValueError(f"Duplicate routed output for prompt: {prompt}")
            expected_seed = base_seed + (expected[prompt] - 1) * 100_000
            if int(row.get("seed")) != expected_seed:
                raise ValueError(f"Seed mismatch for prompt {expected[prompt]}: {row.get('seed')} != {expected_seed}")
            copied = dict(row)
            copied["router"] = {
                "name": config["name"],
                "route": route,
                "adapter_dir": config["routes"][route]["adapter_dir"],
                "prompt_family": family,
            }
            by_prompt[prompt] = copied
        run_summary = path.parent / "run_summary.json"
        sources[route] = {
            "generations": str(path),
            "generations_sha256": sha256_file(path),
            "rows": len(rows),
            "run_summary": str(run_summary) if run_summary.exists() else None,
        }
        if run_summary.exists():
            data = json.loads(run_summary.read_text(encoding="utf-8"))
            sources[route]["wall_seconds"] = data.get("wall_seconds")
            sources[route]["run_metrics"] = data.get("run_metrics")
    missing = [prompt for prompt in expected if prompt not in by_prompt]
    if missing or len(by_prompt) != len(expected):
        raise ValueError(f"Routed outputs do not exactly cover prompts; missing={len(missing)}")
    ordered = [by_prompt[str(row["prompt"])] for row in prompts]
    for index, row in enumerate(ordered, start=1):
        row["index"] = index
        row["prompt_index"] = index
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "router": str(config_path),
        "router_sha256": sha256_file(config_path),
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "base_seed": base_seed,
        "rows": len(ordered),
        "exact_prompt_coverage": True,
        "seed_preservation": True,
        "sources": sources,
        "output_jsonl": str(output_jsonl),
        "output_sha256": sha256_file(output_jsonl),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--prompts", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--config", type=Path, required=True)
    combine_parser.add_argument("--prompts", type=Path, required=True)
    combine_parser.add_argument("--input", action="append", required=True)
    combine_parser.add_argument("--output-jsonl", type=Path, required=True)
    combine_parser.add_argument("--summary", type=Path, required=True)
    combine_parser.add_argument("--base-seed", type=int, default=42)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--config", type=Path, required=True)
    select_parser.add_argument("--prompts", type=Path, required=True)
    select_parser.add_argument("--route", required=True)
    select_parser.add_argument("--input-jsonl", type=Path, required=True)
    select_parser.add_argument("--output-jsonl", type=Path, required=True)
    select_parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config, args.prompts, args.output_dir)
    elif args.command == "combine":
        result = combine(args.config, args.prompts, args.input, args.output_jsonl, args.summary, args.base_seed)
    else:
        result = select_route_outputs(
            args.config, args.prompts, args.route, args.input_jsonl, args.output_jsonl, args.base_seed
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
