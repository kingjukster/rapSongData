import json
from pathlib import Path

from scripts.family_route_prompts import combine, prepare, select_route_outputs
from src.rap_song_data.evaluation.fixed_generation import load_prompt_specs


def write_config(path: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "name": "test-router",
        "routes": {
            "a": {"prompt_families": ["family_a"], "adapter_dir": "adapter-a"},
            "b": {"prompt_families": ["family_b"], "adapter_dir": "adapter-b"},
        },
    }), encoding="utf-8")


def test_routing_preserves_original_generation_indices_and_seeds(tmp_path: Path) -> None:
    config = tmp_path / "router.json"
    prompts = tmp_path / "prompts.json"
    write_config(config)
    prompts.write_text(json.dumps([
        {"prompt": "first", "prompt_family": "family_a"},
        {"prompt": "second", "prompt_family": "family_b"},
        {"prompt": "third", "prompt_family": "family_a"},
    ]), encoding="utf-8")
    manifest = prepare(config, prompts, tmp_path / "prepared")
    assert manifest["routed_prompt_count"] == 3
    specs = load_prompt_specs(tmp_path / "prepared" / "prompts_a.json")
    assert [spec["generation_prompt_index"] for spec in specs] == [1, 3]

    input_paths = []
    for route, rows in {"a": [("first", 42), ("third", 200042)], "b": [("second", 100042)]}.items():
        run_dir = tmp_path / route
        run_dir.mkdir()
        path = run_dir / "generations.jsonl"
        path.write_text("".join(json.dumps({"prompt": prompt, "seed": seed}) + "\n" for prompt, seed in rows), encoding="utf-8")
        input_paths.append(f"{route}={path}")
    result = combine(config, prompts, input_paths, tmp_path / "combined.jsonl", tmp_path / "summary.json", 42)
    assert result["exact_prompt_coverage"] is True
    combined = [json.loads(line) for line in (tmp_path / "combined.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in combined] == ["first", "second", "third"]
    assert [row["router"]["route"] for row in combined] == ["a", "b", "a"]


def test_select_route_outputs_reuses_only_assigned_family(tmp_path: Path) -> None:
    config = tmp_path / "router.json"
    prompts = tmp_path / "prompts.json"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "selected.jsonl"
    write_config(config)
    prompts.write_text(json.dumps([
        {"prompt": "first", "prompt_family": "family_a"},
        {"prompt": "second", "prompt_family": "family_b"},
        {"prompt": "third", "prompt_family": "family_a"},
    ]), encoding="utf-8")
    source.write_text("".join(json.dumps(row) + "\n" for row in [
        {"prompt": "first", "seed": 42},
        {"prompt": "second", "seed": 100042},
        {"prompt": "third", "seed": 200042},
    ]), encoding="utf-8")
    result = select_route_outputs(config, prompts, "a", source, output, 42)
    assert result["rows"] == 2
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in selected] == ["first", "third"]
