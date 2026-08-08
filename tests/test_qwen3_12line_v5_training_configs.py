import json
from pathlib import Path


def test_v5_configs_are_matched_and_conservative() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke = json.loads((root / "configs/training/local_cuda_qwen3_4b_12line_section_v5_smoke.json").read_text())
    full = json.loads((root / "configs/training/local_cuda_qwen3_4b_12line_section_v5_e1.json").read_text())
    assert smoke["base_model"] == full["base_model"] == "Qwen/Qwen3-4B"
    assert smoke["dataset"]["train_path"] == full["dataset"]["train_path"]
    assert smoke["training"]["max_steps"] == 10
    assert full["training"]["max_steps"] == 0
    assert full["training"]["num_train_epochs"] == 1.0
    for config in (smoke, full):
        training = config["training"]
        assert training["sequence_length"] == 512
        assert training["load_in_4bit"] is True
        assert training["bf16"] is True
        assert training["lora_rank"] == 8
        assert training["gradient_checkpointing"] is False
