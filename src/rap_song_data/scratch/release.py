from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .common import PRIVATE_RESEARCH_POLICY, command_record, hash_file, utc_now, write_json


LOADABLE_CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def freeze_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    destination = Path(args.release_dir) / args.name
    if destination.exists():
        raise FileExistsError(f"Versioned release already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for name in LOADABLE_CHECKPOINT_FILES:
        source_file = source / name
        if not source_file.is_file():
            if name == "generation_config.json":
                continue
            raise FileNotFoundError(f"Required checkpoint file is missing: {source_file}")
        target_file = temporary / name
        shutil.copy2(source_file, target_file)
        files.append(
            {
                "name": name,
                "bytes": target_file.stat().st_size,
                "sha256": hash_file(target_file),
            }
        )
    manifest = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "frozen",
        "release_name": args.name,
        "checkpoint_kind": args.kind,
        "created_at": utc_now(),
        "source_checkpoint": str(source),
        "command": command_record(),
        "files": files,
    }
    write_json(temporary / "release_manifest.json", manifest)
    temporary.replace(destination)
    return {**manifest, "release_path": str(destination)}


def add_release_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, default=Path("model/releases"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", choices=("base", "sft"), required=True)
