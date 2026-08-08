from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "training"


def load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_v41_changes_only_dataset_cache_and_output_from_v4() -> None:
    for suffix in ("smoke", "e1"):
        v4 = load(f"local_cuda_qwen3_4b_12line_section_v4_{suffix}.json")
        v41 = load(f"local_cuda_qwen3_4b_12line_section_v41_{suffix}.json")
        v4["output_dir"] = v41["output_dir"]
        v4["dataset"] = v41["dataset"]
        v4["training"]["tokenized_cache_dir"] = v41["training"]["tokenized_cache_dir"]
        assert v4 == v41


def test_v41_smoke_and_full_budgets() -> None:
    smoke = load("local_cuda_qwen3_4b_12line_section_v41_smoke.json")
    full = load("local_cuda_qwen3_4b_12line_section_v41_e1.json")
    assert smoke["training"]["max_steps"] == 10
    assert smoke["training"]["max_wall_time_minutes"] == 10
    assert full["training"]["max_steps"] == 0
    assert full["training"]["num_train_epochs"] == 1.0
    assert full["training"]["seed"] == 20260712
