#!/usr/bin/env python3
"""Build seed-preserving prompt-family slices from saved generation runs."""

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


def family_specs(prompts_path: Path, family: str) -> list[dict[str, Any]]:
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prompt file must contain a JSON list")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError("Prompt entries must be objects with a string prompt field")
        prompt = item["prompt"].strip()
        if not prompt or prompt in seen:
            raise ValueError("Prompts must be non-empty and unique")
        seen.add(prompt)
        if item.get("prompt_family") == family:
            spec = dict(item)
            spec["prompt"] = prompt
            spec["generation_prompt_index"] = index
            specs.append(spec)
    if not specs:
        raise ValueError(f"No prompts found for family {family!r}")
    return specs


def collect(
    prompts_path: Path,
    family: str,
    inputs: list[Path],
    output_jsonl: Path,
    missing_prompts: Path,
    summary_path: Path,
    base_seed: int,
    require_complete: bool,
    skip_seed_mismatches: bool = False,
) -> dict[str, Any]:
    specs = family_specs(prompts_path, family)
    expected = {spec["prompt"]: spec for spec in specs}
    selected: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in inputs:
        matched = 0
        seed_mismatches = 0
        for row in read_jsonl(path):
            prompt = str(row.get("prompt") or "").strip()
            if prompt not in expected:
                continue
            prompt_index = int(expected[prompt]["generation_prompt_index"])
            expected_seed = base_seed + (prompt_index - 1) * 100_000
            if int(row.get("seed", -1)) != expected_seed:
                if skip_seed_mismatches:
                    seed_mismatches += 1
                    continue
                raise ValueError(f"Seed mismatch for prompt index {prompt_index} in {path}")
            if prompt in selected:
                prior = selected[prompt]
                if prior.get("lyrics") != row.get("lyrics"):
                    raise ValueError(f"Conflicting duplicate output for prompt index {prompt_index}")
                continue
            selected[prompt] = dict(row)
            matched += 1
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "matched_rows": matched,
            "seed_mismatch_rows_skipped": seed_mismatches,
        })

    missing = [spec for spec in specs if spec["prompt"] not in selected]
    if require_complete and missing:
        raise ValueError(f"Family slice is incomplete: {len(missing)} prompts missing")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for spec in specs:
            row = selected.get(spec["prompt"])
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing_prompts.parent.mkdir(parents=True, exist_ok=True)
    missing_prompts.write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "prompt_family": family,
        "base_seed": base_seed,
        "expected_rows": len(specs),
        "selected_rows": len(selected),
        "missing_rows": len(missing),
        "complete": not missing,
        "seed_preservation": True,
        "sources": sources,
        "output_jsonl": str(output_jsonl),
        "output_sha256": sha256_file(output_jsonl),
        "missing_prompts": str(missing_prompts),
        "missing_prompts_sha256": sha256_file(missing_prompts),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--missing-prompts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-seed-mismatches", action="store_true")
    args = parser.parse_args()
    result = collect(
        args.prompts,
        args.family,
        args.input,
        args.output_jsonl,
        args.missing_prompts,
        args.summary,
        args.base_seed,
        args.require_complete,
        args.skip_seed_mismatches,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
