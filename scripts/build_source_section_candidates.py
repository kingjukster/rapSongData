"""Extract source-level rap verse/hook candidates from the raw song corpus.

This intentionally works from the raw song table shape instead of existing SFT
mixtures. The parquet mirror of the 9GB CSV is preferred for wall time, but the
section output keeps raw song ids/titles/artists so the pipeline is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HEADER_RE = re.compile(r"^\s*\[(?P<label>[^\]]{1,120})\]\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
ARTIFACT_RE = re.compile(
    r"(?:download the full version|itunes|genius\.com|you might also like|embed|"
    r"https?://|www\.|lyrics taken from|contributor|transcriber)",
    re.IGNORECASE,
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line


def section_kind(label: str) -> str:
    lowered = label.lower()
    if "verse" in lowered:
        return "verse"
    if "hook" in lowered or "chorus" in lowered or "refrain" in lowered:
        return "hook"
    if "bridge" in lowered or "pre-chorus" in lowered or "pre chorus" in lowered:
        return "bridge"
    if "intro" in lowered:
        return "intro"
    if "outro" in lowered:
        return "outro"
    return "other"


def split_sections(lyrics: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_label = "untagged"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        lines = [line for line in current_lines if line]
        if lines:
            sections.append({"label": current_label, "kind": section_kind(current_label), "lines": lines})
        current_lines = []

    for raw_line in str(lyrics or "").splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        match = HEADER_RE.match(line)
        if match:
            flush()
            current_label = match.group("label").strip()
            continue
        current_lines.append(line)
    flush()
    return sections


def iter_parquet_rows(path: Path, *, batch_size: int, columns: list[str]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    selected = [column for column in columns if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield from batch.to_pylist()


def row_is_eligible(row: dict[str, Any]) -> bool:
    tag = str(row.get("tag") or "").lower()
    language = str(row.get("language") or row.get("language_ft") or row.get("language_cld3") or "").lower()
    lyrics = str(row.get("lyrics") or "")
    if tag and tag != "rap":
        return False
    if language and language != "en":
        return False
    if len(lyrics) < 120 or ARTIFACT_RE.search(lyrics):
        return False
    return True


def section_shape(lines: list[str]) -> dict[str, Any]:
    line_word_counts = [len(words(line)) for line in lines]
    text = "\n".join(lines)
    return {
        "line_count": len(lines),
        "word_count": len(words(text)),
        "max_line_words": max(line_word_counts) if line_word_counts else 0,
        "avg_line_words": round(sum(line_word_counts) / len(line_word_counts), 2) if line_word_counts else 0,
        "has_artifact": bool(ARTIFACT_RE.search(text)),
    }


def candidate_ok(kind: str, shape: dict[str, Any], args: argparse.Namespace) -> bool:
    if shape["has_artifact"]:
        return False
    if shape["max_line_words"] > args.max_line_words:
        return False
    if kind == "verse":
        return args.min_verse_lines <= shape["line_count"] <= args.max_verse_lines and shape["word_count"] >= 45
    if kind == "hook":
        return args.min_hook_lines <= shape["line_count"] <= args.max_hook_lines and shape["word_count"] >= 12
    return args.include_other and 4 <= shape["line_count"] <= 16 and shape["word_count"] >= 20


def cmd_extract(args: argparse.Namespace) -> None:
    columns = [
        "title",
        "tag",
        "artist",
        "artist_clean",
        "year",
        "views",
        "features",
        "id",
        "language_cld3",
        "language_ft",
        "language",
        "lyrics",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    counts: Counter[str] = Counter()
    written = 0
    scanned_rows = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(iter_parquet_rows(args.input, batch_size=args.batch_size, columns=columns)):
            scanned_rows += 1
            if args.scan_limit is not None and scanned_rows > args.scan_limit:
                break
            if not row_is_eligible(row):
                counts["row_rejected"] += 1
                continue
            song_id = str(row.get("id") or row_index)
            for section_index, section in enumerate(split_sections(str(row.get("lyrics") or ""))):
                kind = section["kind"]
                if kind not in {"verse", "hook"} and not args.include_other:
                    counts[f"skip_kind_{kind}"] += 1
                    continue
                shape = section_shape(section["lines"])
                if not candidate_ok(kind, shape, args):
                    counts[f"skip_shape_{kind}"] += 1
                    continue
                text = "\n".join(section["lines"]).strip()
                digest = hashlib.sha1(text.lower().encode("utf-8", errors="ignore")).hexdigest()
                if digest in seen_hashes:
                    counts["duplicate_section"] += 1
                    continue
                seen_hashes.add(digest)
                record = {
                    "record_id": f"song:{song_id}:section:{section_index}:sha1:{digest[:12]}",
                    "song_id": song_id,
                    "section_index": section_index,
                    "section_label": section["label"],
                    "section_kind": kind,
                    "title": str(row.get("title") or ""),
                    "artist": str(row.get("artist_clean") or row.get("artist") or ""),
                    "year": row.get("year"),
                    "views": row.get("views"),
                    "shape": shape,
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                counts[f"written_{kind}"] += 1
                if args.limit is not None and written >= args.limit:
                    break
            if args.limit is not None and written >= args.limit:
                break

    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "scanned_rows": scanned_rows,
        "written": written,
        "counts": dict(counts),
        "settings": settings,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    extract.add_argument("--output", type=Path, default=Path("data/source_sections/raw_rap_source_section_candidates.jsonl"))
    extract.add_argument("--summary-output", type=Path, default=Path("data/source_sections/raw_rap_source_section_candidates_summary.json"))
    extract.add_argument("--batch-size", type=int, default=8192)
    extract.add_argument("--scan-limit", type=int, default=None)
    extract.add_argument("--limit", type=int, default=100000)
    extract.add_argument("--min-verse-lines", type=int, default=8)
    extract.add_argument("--max-verse-lines", type=int, default=28)
    extract.add_argument("--min-hook-lines", type=int, default=4)
    extract.add_argument("--max-hook-lines", type=int, default=12)
    extract.add_argument("--max-line-words", type=int, default=32)
    extract.add_argument("--include-other", action=argparse.BooleanOptionalAction, default=False)
    extract.set_defaults(func=cmd_extract)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
