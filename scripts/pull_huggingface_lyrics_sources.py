from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


DEFAULT_SNAPSHOT_ID = "20260715_hf_public_snapshot"


def source_id_for(repo_id: str) -> str:
    return "hf_private_lyrics_" + repo_id.lower().replace("/", "__").replace("-", "_").replace(".", "_")


def direct_download(repo_id: str, filename: str, dest: Path) -> None:
    """Stream a public dataset file directly to dest with simple resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    headers: dict[str, str] = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    existing = partial.stat().st_size if partial.exists() else 0
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if existing and response.status_code == 416:
            partial.replace(dest)
            return
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
            headers.pop("Range", None)
            response.close()
            with requests.get(url, headers=headers, stream=True, timeout=60) as retry:
                retry.raise_for_status()
                with partial.open(mode) as handle:
                    for chunk in retry.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        else:
            response.raise_for_status()
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    partial.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull public Hugging Face lyric dataset files into D: raw snapshots.")
    parser.add_argument("--repo", action="append", required=True, help="HF dataset repo id; can be repeated.")
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--raw-root", type=Path, default=Path("data/corpus_lake/raw/huggingface_lyrics"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/corpus_lake/cache/huggingface_hub"))
    parser.add_argument("--allow", action="append", default=[], help="Optional filename substring allow-list; can repeat.")
    parser.add_argument("--skip-model-artifacts", action="store_true", default=True)
    parser.add_argument(
        "--direct-download",
        action="store_true",
        help="Stream files directly to the raw snapshot instead of first populating the HF cache.",
    )
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
                dest = out_dir / name
                if args.direct_download:
                    expected_size = getattr(sibling, "size", None)
                    if not dest.exists() or (expected_size and dest.stat().st_size != expected_size):
                        direct_download(repo_id, name, dest)
                else:
                    cached = hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=name,
                        cache_dir=args.cache_dir,
                    )
                    cached_path = Path(cached)
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
