import json
from pathlib import Path

import pytest

from scripts.select_prompt_family_outputs import collect


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_collect_preserves_global_indices_and_emits_missing_prompts(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    source = tmp_path / "source.jsonl"
    prompts.write_text(json.dumps([
        {"prompt": "first", "prompt_family": "other"},
        {"prompt": "second", "prompt_family": "clean"},
        {"prompt": "third", "prompt_family": "clean"},
    ]), encoding="utf-8")
    write_jsonl(source, [{"prompt": "third", "seed": 200042, "lyrics": "three"}])
    result = collect(
        prompts, "clean", [source], tmp_path / "selected.jsonl",
        tmp_path / "missing.json", tmp_path / "summary.json", 42, False,
    )
    assert result["selected_rows"] == 1
    assert result["missing_rows"] == 1
    missing = json.loads((tmp_path / "missing.json").read_text(encoding="utf-8"))
    assert missing == [{"prompt": "second", "prompt_family": "clean", "generation_prompt_index": 2}]


def test_collect_combines_sources_and_rejects_bad_seed(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    prompts.write_text(json.dumps([
        {"prompt": "one", "prompt_family": "clean"},
        {"prompt": "two", "prompt_family": "clean"},
    ]), encoding="utf-8")
    write_jsonl(first, [{"prompt": "one", "seed": 42, "lyrics": "one"}])
    write_jsonl(second, [{"prompt": "two", "seed": 100042, "lyrics": "two"}])
    result = collect(
        prompts, "clean", [first, second], tmp_path / "selected.jsonl",
        tmp_path / "missing.json", tmp_path / "summary.json", 42, True,
    )
    assert result["complete"] is True
    assert result["selected_rows"] == 2

    write_jsonl(second, [{"prompt": "two", "seed": 7, "lyrics": "two"}])
    with pytest.raises(ValueError, match="Seed mismatch"):
        collect(
            prompts, "clean", [first, second], tmp_path / "bad.jsonl",
            tmp_path / "bad-missing.json", tmp_path / "bad-summary.json", 42, True,
        )


def test_collect_can_report_and_skip_noncomparable_seed_rows(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    source = tmp_path / "source.jsonl"
    prompts.write_text(json.dumps([
        {"prompt": "one", "prompt_family": "clean"},
        {"prompt": "two", "prompt_family": "clean"},
    ]), encoding="utf-8")
    write_jsonl(source, [
        {"prompt": "one", "seed": 42, "lyrics": "one"},
        {"prompt": "two", "seed": 7, "lyrics": "wrong seed"},
    ])
    result = collect(
        prompts, "clean", [source], tmp_path / "selected.jsonl",
        tmp_path / "missing.json", tmp_path / "summary.json", 42, False, True,
    )
    assert result["selected_rows"] == 1
    assert result["missing_rows"] == 1
    assert result["sources"][0]["seed_mismatch_rows_skipped"] == 1
