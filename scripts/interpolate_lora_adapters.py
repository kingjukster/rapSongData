#!/usr/bin/env python3
"""Create an exact weighted sum of two LoRA deltas using rank concatenation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_configs(left: dict[str, Any], right: dict[str, Any]) -> None:
    ignored = {"target_modules"}
    for key in sorted(set(left) | set(right)):
        if key in ignored:
            continue
        if left.get(key) != right.get(key):
            raise ValueError(f"Adapter config mismatch for {key}: {left.get(key)!r} != {right.get(key)!r}")
    if set(left.get("target_modules") or []) != set(right.get("target_modules") or []):
        raise ValueError("Adapter target modules differ")
    if left.get("use_rslora") or right.get("use_rslora"):
        raise ValueError("RS-LoRA interpolation is not supported by this exact concatenation path")
    if left.get("rank_pattern") or right.get("rank_pattern"):
        raise ValueError("Per-module rank patterns are not supported")
    if left.get("alpha_pattern") or right.get("alpha_pattern"):
        raise ValueError("Per-module alpha patterns are not supported")


def interpolate_states(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    left_weight: float,
    right_weight: float,
    left_scale: float,
    right_scale: float,
    output_scale: float,
) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        raise ValueError("Adapter tensor keys differ")
    output: dict[str, torch.Tensor] = {}
    a_keys = sorted(key for key in left if ".lora_A." in key)
    if not a_keys or len(a_keys) * 2 != len(left):
        raise ValueError("Expected only matched LoRA A/B tensors")
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key not in left:
            raise ValueError(f"Missing B tensor for {a_key}")
        left_a, right_a = left[a_key], right[a_key]
        left_b, right_b = left[b_key], right[b_key]
        if left_a.shape != right_a.shape or left_b.shape != right_b.shape:
            raise ValueError(f"Tensor shape mismatch for {a_key}")
        if left_a.shape[0] != left_b.shape[1] or right_a.shape[0] != right_b.shape[1]:
            raise ValueError(f"Invalid LoRA factor shapes for {a_key}")
        output[a_key] = torch.cat((left_a, right_a), dim=0).contiguous()
        output[b_key] = torch.cat(
            (
                left_b * (left_weight * left_scale / output_scale),
                right_b * (right_weight * right_scale / output_scale),
            ),
            dim=1,
        ).contiguous()
    return output


def validate_one_delta(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    *,
    left_weight: float,
    right_weight: float,
    left_scale: float,
    right_scale: float,
    output_scale: float,
) -> dict[str, Any]:
    candidates = sorted(
        (key for key in left if ".lora_A." in key),
        key=lambda key: left[key].shape[1] * left[key.replace(".lora_A.", ".lora_B.")].shape[0],
    )
    a_key = candidates[0]
    b_key = a_key.replace(".lora_A.", ".lora_B.")
    expected = left_weight * left_scale * (left[b_key] @ left[a_key])
    expected.add_(right_weight * right_scale * (right[b_key] @ right[a_key]))
    observed = output_scale * (output[b_key] @ output[a_key])
    maximum_error = float((expected - observed).abs().max().item())
    if not torch.allclose(expected, observed, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Weighted-delta validation failed for {a_key}; max error {maximum_error}")
    return {"tensor_pair": a_key.rsplit(".lora_A.", 1)[0], "max_abs_error": maximum_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("--right-weight must be between 0 and 1")
    left_weight = 1.0 - args.right_weight
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    left_config_path = args.left / "adapter_config.json"
    right_config_path = args.right / "adapter_config.json"
    left_model_path = args.left / "adapter_model.safetensors"
    right_model_path = args.right / "adapter_model.safetensors"
    left_config = load_config(left_config_path)
    right_config = load_config(right_config_path)
    validate_configs(left_config, right_config)

    left_state = load_file(left_model_path, device="cpu")
    right_state = load_file(right_model_path, device="cpu")
    left_rank = int(left_config["r"])
    right_rank = int(right_config["r"])
    output_rank = left_rank + right_rank
    left_scale = float(left_config["lora_alpha"]) / left_rank
    right_scale = float(right_config["lora_alpha"]) / right_rank
    output_alpha = output_rank
    output_scale = output_alpha / output_rank
    output_state = interpolate_states(
        left_state,
        right_state,
        left_weight=left_weight,
        right_weight=args.right_weight,
        left_scale=left_scale,
        right_scale=right_scale,
        output_scale=output_scale,
    )
    validation = validate_one_delta(
        left_state,
        right_state,
        output_state,
        left_weight=left_weight,
        right_weight=args.right_weight,
        left_scale=left_scale,
        right_scale=right_scale,
        output_scale=output_scale,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    output_model_path = args.output / "adapter_model.safetensors"
    output_config_path = args.output / "adapter_config.json"
    output_config = dict(left_config)
    output_config["r"] = output_rank
    output_config["lora_alpha"] = output_alpha
    output_config["target_modules"] = sorted(set(left_config["target_modules"]))
    save_file(output_state, output_model_path, metadata={"format": "pt"})
    output_config_path.write_text(json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source = args.left / filename
        if source.exists():
            shutil.copy2(source, args.output / filename)

    exact_command = " ".join([sys.executable, *sys.argv])
    (args.output / "exact_command.txt").write_text(exact_command + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "exact_weighted_lora_delta_sum_via_rank_concatenation",
        "left": {"path": str(args.left), "weight": left_weight, "sha256": sha256_file(left_model_path)},
        "right": {"path": str(args.right), "weight": args.right_weight, "sha256": sha256_file(right_model_path)},
        "output": {"path": str(args.output), "rank": output_rank, "alpha": output_alpha},
        "scales": {"left": left_scale, "right": right_scale, "output": output_scale},
        "tensor_count": len(output_state),
        "validation": validation,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "command": exact_command,
    }
    manifest["output"]["sha256"] = sha256_file(output_model_path)
    (args.output / "interpolation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
