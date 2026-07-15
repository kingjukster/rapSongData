"""Download high-throughput Hugging Face webtext samples into the corpus lake.

This is for auxiliary pretraining text, not lyric-only data. It preserves raw
Parquet shards under D: and writes a manifest so later dedupe/filtering can
decide how much to mix into lyric-model pretraining.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_OUTPUT_ROOT = Path("data/corpus_lake/raw/huggingface_webtext")
DEFAULT_CACHE_DIR = Path("data/corpus_lake/cache/huggingface")

PRESETS = {
    "fineweb_sample_10bt": {
        "repo_id": "HuggingFaceFW/fineweb",
        "path_in_repo": "sample/10BT",
        "license": "odc-by",
        "text_partition": "auxiliary_webtext",
        "expected_tokens": 10_000_000_000,
        "description": "FineWeb random sample around 10B GPT-2 tokens.",
    },
    "fineweb_edu_sample_10bt": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "path_in_repo": "sample/10BT",
        "license": "odc-by",
        "text_partition": "auxiliary_educational_webtext",
        "expected_tokens": 10_000_000_000,
        "description": "FineWeb-Edu random sample around 10B GPT-2 tokens.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def parquet_probe(path: Path) -> dict[str, Any]:
    metadata = pq.read_metadata(path)
    columns = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
    return {
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": columns,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-files", type=int, default=0, help="Smoke-test cap; 0 downloads all files.")
    parser.add_argument("--resume", action="store_true", help="Reuse already downloaded files when present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    repo_id = preset["repo_id"]
    path_in_repo = preset["path_in_repo"]
    safe_repo = repo_id.replace("/", "__")
    output_dir = args.output_root / safe_repo / args.snapshot_id
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    command_path = output_dir / "command.json"
    summary_path = output_dir / "download_summary.json"

    start = time.time()
    command = {
        "generated_at_utc": utc_now(),
        "args": jsonable_args(args),
        "preset": preset,
        "output_dir": str(output_dir),
    }
    write_json(command_path, command)

    api = HfApi()
    tree = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo, recursive=True))
    files = [item for item in tree if item.path.endswith(".parquet")]
    files = sorted(files, key=lambda item: item.path)
    if args.max_files:
        files = files[: args.max_files]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "preset": args.preset,
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "license": preset["license"],
        "text_partition": preset["text_partition"],
        "expected_tokens": preset["expected_tokens"],
        "description": preset["description"],
        "output_dir": str(output_dir),
        "files": [],
    }
    write_json(manifest_path, manifest)

    downloaded_bytes = 0
    total_rows = 0
    for index, item in enumerate(files, start=1):
        local_name = Path(item.path).name
        final_path = data_dir / local_name
        print(f"[hf-webtext] file_started {index}/{len(files)} repo_path={item.path}", flush=True)
        if final_path.exists() and args.resume and final_path.stat().st_size == (item.size or final_path.stat().st_size):
            downloaded_path = final_path
            reused = True
        else:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=item.path,
                cache_dir=str(args.cache_dir),
                local_dir=str(data_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != final_path.resolve() and downloaded_path.exists():
                # hf_hub_download(local_dir=...) usually returns the final path.
                final_path = downloaded_path
            reused = False
        stat = final_path.stat()
        probe = parquet_probe(final_path)
        downloaded_bytes += stat.st_size
        total_rows += int(probe["rows"])
        file_record = {
            "repo_path": item.path,
            "local_path": str(final_path),
            "expected_size_bytes": item.size,
            "size_bytes": stat.st_size,
            "reused": reused,
            **probe,
            "completed_at_utc": utc_now(),
        }
        manifest["files"].append(file_record)
        manifest["generated_at_utc"] = utc_now()
        manifest["downloaded_bytes"] = downloaded_bytes
        manifest["total_rows"] = total_rows
        write_json(manifest_path, manifest)
        print(
            f"[hf-webtext] file_completed {index}/{len(files)} bytes={stat.st_size} rows={probe['rows']} "
            f"elapsed={time.time() - start:.1f}s",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "preset": args.preset,
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "license": preset["license"],
        "text_partition": preset["text_partition"],
        "expected_tokens": preset["expected_tokens"],
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "file_count": len(manifest["files"]),
        "downloaded_bytes": downloaded_bytes,
        "total_rows": total_rows,
        "wall_seconds": round(time.time() - start, 3),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
