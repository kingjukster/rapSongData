from __future__ import annotations

import json
import sys
from pathlib import Path

from rap_song_data.cli import _build_parser, _load_config


ROOT = Path(__file__).resolve().parents[1]


def test_standard_project_layout_is_present() -> None:
    expected = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "src" / "rap_song_data" / "__init__.py",
        ROOT / "configs" / "pipeline.yaml",
        ROOT / "notebooks" / "rap_songs_filter.ipynb",
    ]
    assert all(path.exists() for path in expected)


def test_all_versioned_json_configs_are_valid() -> None:
    config_paths = sorted((ROOT / "configs").rglob("*.json"))
    assert config_paths
    for path in config_paths:
        json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_config_section_overrides_defaults(monkeypatch) -> None:
    config_path = ROOT / "configs" / "pipeline.yaml"
    parser = _build_parser()
    args = parser.parse_args(["--config", str(config_path), "generate"])
    monkeypatch.setattr(sys, "argv", ["rap-pipeline", "--config", str(config_path), "generate"])

    loaded = _load_config(str(config_path), args)

    assert loaded.run_dir == "runs/generation"
    assert loaded.temperature == 0.85


def test_explicit_cli_value_wins_over_config(monkeypatch) -> None:
    config_path = ROOT / "configs" / "pipeline.yaml"
    parser = _build_parser()
    args = parser.parse_args(
        ["--config", str(config_path), "generate", "--run-dir", "runs/custom"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rap-pipeline", "--config", str(config_path), "generate", "--run-dir", "runs/custom"],
    )

    loaded = _load_config(str(config_path), args)

    assert loaded.run_dir == "runs/custom"
