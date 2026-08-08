import json
from pathlib import Path


def test_olmo_configs_match_qwen_v51_except_model_runtime_paths_and_stage_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    qwen = json.loads((root / "configs/training/local_cuda_qwen3_4b_12line_section_v51_e1.json").read_text())
    smoke = json.loads((root / "configs/training/local_cuda_olmo3_7b_12line_section_v51_smoke.json").read_text())
    full = json.loads((root / "configs/training/local_cuda_olmo3_7b_12line_section_v51_e1.json").read_text())

    assert smoke["base_model"] == full["base_model"] == "allenai/Olmo-3-7B-Instruct"
    assert smoke["model_revision"] == full["model_revision"]
    assert smoke["dataset"] == full["dataset"]
    assert full["dataset"] == qwen["dataset"]

    ignored = {
        "max_steps", "max_wall_time_minutes", "logging_steps", "timing_log_steps",
        "eval_strategy", "save_strategy", "tokenized_cache_dir",
    }
    assert {k: v for k, v in full["training"].items() if k not in ignored} == {
        k: v for k, v in qwen["training"].items() if k not in ignored
    }
    assert smoke["training"]["max_steps"] == 10
    assert full["training"]["max_steps"] == 0
