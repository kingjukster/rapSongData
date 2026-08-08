"""Filter extreme shape outliers from an already materialized private lyric lake profile.

This preserves the input profile and writes a new profile containing filtered
train/validation/test JSONL plus an audit manifest and outlier quarantine file.
It does not inspect or copy raw upstream sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def text_field(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("lyrics", "text", "body", "content", "song_text", "lyric_text"):
        value = row.get(key)
        if isinstance(value, str):
            return value, key
    return "", ""


def line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def outlier_reasons(text: str, *, max_chars: int, max_lines: int) -> list[str]:
    reasons: list[str] = []
    if len(text) > max_chars:
        reasons.append("over_max_chars")
    if line_count(text) > max_lines:
        reasons.append("over_max_lines")
    return reasons


def output_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-name", default="scratch-private-lyric-lake-v2_1_outlier_filtered")
    parser.add_argument("--max-chars", type=int, default=50_000)
    parser.add_argument("--max-lines", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    input_manifest_path = input_dir / "corpus_manifest.json"
    if not input_manifest_path.exists():
        raise FileNotFoundError(f"missing input corpus_manifest.json: {input_manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "corpus_manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"), flush=True)
        return 0

    if args.force:
        for pattern in ("*.jsonl", "*.partial", "corpus_manifest.json", "filter_manifest.json"):
            for path in output_dir.glob(pattern):
                path.unlink()
        tokenized = output_dir / "tokenized"
        tokenizer = output_dir / "tokenizer"
        for path in (tokenized, tokenizer):
            if path.exists():
                shutil.rmtree(path)

    started = time.monotonic()
    started_at = utc_now()
    input_manifest = read_json(input_manifest_path)
    counters: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    max_seen = {"chars": 0, "lines": 0}

    quarantine_path = output_dir / "outlier_quarantine.jsonl.partial"
    with quarantine_path.open("w", encoding="utf-8", newline="\n") as quarantine:
        for split in ("train", "validation", "test"):
            input_path = input_dir / f"{split}.jsonl"
            output_partial = output_dir / f"{split}.jsonl.partial"
            if not input_path.exists():
                raise FileNotFoundError(f"missing input split: {input_path}")
            with input_path.open("r", encoding="utf-8") as inp, output_partial.open(
                "w", encoding="utf-8", newline="\n"
            ) as out:
                for line in inp:
                    counters["input_records"] += 1
                    counters[f"input_{split}"] += 1
                    row = json.loads(line)
                    text, field = text_field(row)
                    chars = len(text)
                    lines = line_count(text)
                    max_seen["chars"] = max(max_seen["chars"], chars)
                    max_seen["lines"] = max(max_seen["lines"], lines)
                    reasons = outlier_reasons(text, max_chars=args.max_chars, max_lines=args.max_lines)
                    if reasons:
                        counters["rejected_outlier"] += 1
                        counters[f"rejected_{split}"] += 1
                        for reason in reasons:
                            by_reason[reason] += 1
                        write_jsonl(
                            quarantine,
                            {
                                "split": split,
                                "record_id": row.get("record_id"),
                                "source_id": row.get("source_id"),
                                "source_family": row.get("source_family"),
                                "text_field": field,
                                "chars": chars,
                                "lines": lines,
                                "reasons": reasons,
                                "normalized_hash": row.get("normalized_hash"),
                                "removal_key": row.get("removal_key"),
                            },
                        )
                        continue
                    row["profile"] = args.profile_name
                    write_jsonl(out, row)
                    counters["retained"] += 1
                    counters[f"split_{split}"] += 1
                    by_source[str(row.get("source_id") or "unknown")] += 1
                    if row.get("content_flags"):
                        counters["explicit_flagged"] += 1
                    if counters["input_records"] % args.progress_every == 0:
                        print(
                            "[filter] progress "
                            f"input={counters['input_records']} retained={counters['retained']} "
                            f"rejected_outlier={counters['rejected_outlier']}",
                            flush=True,
                        )
            output_partial.replace(output_dir / f"{split}.jsonl")

    quarantine_path.replace(output_dir / "outlier_quarantine.jsonl")
    outputs = {
        split: output_info(output_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    outputs["outlier_quarantine"] = output_info(output_dir / "outlier_quarantine.jsonl")

    manifest = {
        "private_research_only": True,
        "source_license": "unknown",
        "public_release_blocked": True,
        "schema_version": 1,
        "status": "complete",
        "operation": "filter-private-lyric-lake-outliers",
        "profile": args.profile_name,
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": sys.argv,
        "input_profile": {
            "path": str(input_dir),
            "corpus_manifest": str(input_manifest_path),
            "profile": input_manifest.get("profile"),
            "counts": input_manifest.get("counts"),
        },
        "settings": {
            "max_chars": args.max_chars,
            "max_lines": args.max_lines,
            "filter": "reject records where lyrics/text exceeds max chars or max nonblank lines",
            "raw_preserved": True,
        },
        "counts": dict(counters),
        "records_by_source": dict(sorted(by_source.items())),
        "rejected_by_reason": dict(sorted(by_reason.items())),
        "max_seen": max_seen,
        "outputs": outputs,
        "acceptance": {
            "minimum_retained_songs": 1,
            "retained_songs_passed": counters["retained"] >= 1,
            "token_gate_pending": True,
        },
        "note": "v2.1 profile preserves the v2 source profile and quarantines only extreme shape outliers before tokenizer training.",
    }
    write_json(manifest_path, manifest)
    write_json(output_dir / "filter_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
