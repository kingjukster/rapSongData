from __future__ import annotations

import argparse
import json
from pathlib import Path


def slugify_ref(kaggle_ref: str) -> str:
    return kaggle_ref.replace("/", "__").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print ranked private-use Kaggle lyric dataset candidates and download targets."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/datasets/kaggle_lyrics_candidate_registry.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/corpus_lake/raw/kaggle_private_lyrics"),
    )
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    candidates = sorted(registry["candidates"], key=lambda row: int(row["rank"]))[: args.limit]
    print(f"registry: {args.registry}")
    print(f"intended_partition: {registry['intended_partition']}")
    print(f"default_training_eligibility: {registry['default_training_eligibility']}")
    print()
    for candidate in candidates:
        ref = candidate["kaggle_ref"]
        target = args.raw_root / slugify_ref(ref)
        print(f"{candidate['rank']}. {candidate['name']}")
        print(f"   ref: {ref}")
        print(f"   priority: {candidate['private_use_priority']}")
        print(f"   observed_license: {candidate['observed_kaggle_license']}")
        print(f"   clean_core_eligibility: {candidate['clean_core_eligibility']}")
        print(f"   target_dir: {target}")
        print(f"   command: kaggle datasets download -d {ref} -p {target} --unzip")
        print(f"   next_action: {candidate['next_action']}")
        print()


if __name__ == "__main__":
    main()
