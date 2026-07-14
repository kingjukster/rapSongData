from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


PRIVATE_RESEARCH_POLICY = {
    "private_research_only": True,
    "source_license": "unknown",
    "public_release_blocked": True,
}

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|title|>",
    "<|year|>",
    "<|year_unknown|>",
    "<|lyrics|>",
    "<|task_generate|>",
    "<|section|>",
    "<|target_lines|>",
    "<|content_clean|>",
    "<|content_explicit|>",
    "<|verse|>",
    "<|hook|>",
    "<|chorus|>",
    "<|bridge|>",
    "<|intro|>",
    "<|outro|>",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_text(value: str, *, digest_size: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=digest_size).hexdigest()


def hash_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, mode: str = "w") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def command_record() -> list[str]:
    return [sys.executable, *sys.argv]


def path_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])
