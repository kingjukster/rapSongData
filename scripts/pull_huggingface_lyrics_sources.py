from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_SNAPSHOT_ID = "20260715_hf_public_snapshot"


def source_id_for(repo_id: str) -> str:
    return "hf_private_lyrics_" + repo_id.lower().replace("/", "__").replace("-", "_").replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull public Hugging Face lyric dataset files into D: raw snapshots.")
    parser.add_argument("--repo", action="append", required=True, help="HF dataset repo id; can be repeated.")
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--raw-root", type=Path, default=Path("data/corpus_lake/raw/huggingface_lyrics"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/corpus_lake/cache/huggingface_hub"))
    parser.add_argument("--allow", action="append", default=[], help="Optional filename substring allow-list; can repeat.")
    parser.add_argument("--skip-model-artifacts", action="store_true", default=True)
    args = parser.parse_args()

    api = HfApi()
    started_all = time.time()
    for repo_id in args.repo:
        started = time.time()
        info = api.dataset_info(repo_id, files_metadata=True)
        slug = repo_id.replace("/", "__")
        out_dir = args.raw_root / slug / args.snapshot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        for sibling in info.siblings or []:
            name = sibling.rfilename
            if name.startswith(".git"):
                continue
            if args.allow and not any(token in name for token in args.allow):
                skipped.append({"file": name, "reason": "not_allowed"})
                continue
            lower = name.lower()
            if args.skip_model_artifacts and (
                name.startswith("Models/")
                or lower.endswith((".pth", ".npy", ".npz"))
                or "trained_model" in lower
                or "embeddings" in lower
            ):
                skipped.append({"file": name, "reason": "model_or_embedding_artifact"})
                continue
            try:
                cached = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=name,
                    cache_dir=args.cache_dir,
                )
                cached_path = Path(cached)
                dest = out_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.stat().st_size != cached_path.stat().st_size:
                    shutil.copy2(cached_path, dest)
                downloaded.append(
                    {
                        "path": str(dest.as_posix()),
                        "bytes": dest.stat().st_size,
                        "rfilename": name,
                    }
                )
                print(json.dumps({"event": "file_downloaded", "repo": repo_id, "file": name, "bytes": dest.stat().st_size}))
            except Exception as exc:  # pragma: no cover - operational script
                errors.append({"file": name, "error": repr(exc)})
                print(json.dumps({"event": "file_error", "repo": repo_id, "file": name, "error": repr(exc)}))
        manifest = {
            "source_id": source_id_for(repo_id),
            "huggingface_dataset": repo_id,
            "snapshot_id": args.snapshot_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "private": info.private,
            "gated": info.gated,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags,
            "license_tags": [tag for tag in (info.tags or []) if str(tag).startswith("license:")],
            "rights_note": "Private/personal-use only unless underlying lyric rights are later verified.",
            "downloaded_files": downloaded,
            "skipped_files": skipped,
            "errors": errors,
            "wall_seconds": round(time.time() - started, 3),
        }
        (out_dir / "hf_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "dataset_completed",
                    "repo": repo_id,
                    "downloaded_files": len(downloaded),
                    "skipped_files": len(skipped),
                    "errors": len(errors),
                    "wall_seconds": manifest["wall_seconds"],
                }
            )
        )
    print(json.dumps({"event": "all_completed", "wall_seconds": round(time.time() - started_all, 3)}))


if __name__ == "__main__":
    main()
