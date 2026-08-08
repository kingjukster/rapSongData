"""Build a ranked full-corpus song manifest for OpenAI full-song splitting."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SECTION_HEADER_RE = re.compile(r"^\s*\[(?:verse|hook|chorus|bridge|intro|outro|pre-chorus|post-chorus)[^\]]*\]\s*$", re.I)
JUNK_RE = re.compile(r"(embed|you might also like|lyrics taken from|genius\.com|http://|https://)", re.I)


def iter_parquet_rows(path: Path, *, batch_size: int, columns: list[str]) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    selected = [column for column in columns if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield from batch.to_pylist()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def song_key(row: dict[str, Any], lyrics: str) -> str:
    return str(row.get("id") or hashlib.sha1(lyrics.encode("utf-8", errors="ignore")).hexdigest()[:16])


def score_candidate(*, views: int, word_count: int, line_count: int, header_count: int, avg_line_words: float, title: str, artist: str) -> float:
    score = 0.0
    score += min(math.log10(max(views, 0) + 1.0), 7.0) * 10.0
    score += min(word_count / 500.0, 4.0) * 8.0
    score += min(line_count / 40.0, 3.0) * 8.0
    score += min(header_count, 8) * 3.0
    if 5.0 <= avg_line_words <= 14.0:
        score += 12.0
    elif 3.0 <= avg_line_words <= 20.0:
        score += 5.0
    if title.strip():
        score += 2.0
    if artist.strip():
        score += 2.0
    return round(score, 3)


def candidate_or_reject(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    lyrics = str(row.get("lyrics") or "")
    title = str(row.get("title") or "")
    artist = str(row.get("artist") or row.get("artist_clean") or "")
    tag = str(row.get("tag") or "").lower()
    language = str(row.get("language") or row.get("language_ft") or row.get("language_cld3") or "").lower()
    if tag and tag != "rap":
        return None, "non_rap_tag"
    if language and language != "en":
        return None, "non_english"
    if JUNK_RE.search(lyrics[:2000]):
        return None, "obvious_scrape_junk"
    line_values = [line.strip() for line in lyrics.splitlines() if line.strip()]
    word_count = len(words(lyrics))
    if word_count < 80:
        return None, "too_few_words"
    if len(line_values) < 8:
        return None, "too_few_lines"
    text_hash = hashlib.sha1(re.sub(r"\s+", " ", lyrics).strip().lower().encode("utf-8", errors="ignore")).hexdigest()
    header_count = sum(1 for line in line_values if SECTION_HEADER_RE.match(line))
    avg_line_words = word_count / max(1, len(line_values))
    views = as_int(row.get("views"))
    key = song_key(row, lyrics)
    record = {
        "song_key": key,
        "source_id": row.get("id"),
        "title": title,
        "artist": artist,
        "year": as_int(row.get("year"), default=0) or None,
        "views": views,
        "tag": row.get("tag"),
        "language": row.get("language") or row.get("language_ft") or row.get("language_cld3"),
        "word_count": word_count,
        "line_count": len(line_values),
        "header_count": header_count,
        "avg_line_words": round(avg_line_words, 2),
        "char_count": len(lyrics),
        "text_hash": text_hash,
        "rank_score": score_candidate(
            views=views,
            word_count=word_count,
            line_count=len(line_values),
            header_count=header_count,
            avg_line_words=avg_line_words,
            title=title,
            artist=artist,
        ),
    }
    return record, "accepted"


def cmd_build(args: argparse.Namespace) -> None:
    columns = ["title", "tag", "artist", "artist_clean", "year", "views", "id", "language_cld3", "language_ft", "language", "lyrics"]
    rejection_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    seen_keys: set[str] = set()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    all_count = 0
    accepted_count = 0
    duplicate_count = 0

    for index, row in enumerate(iter_parquet_rows(args.input, batch_size=args.read_batch_size, columns=columns), start=1):
        all_count += 1
        if args.scan_limit is not None and all_count > args.scan_limit:
            break
        candidate, reason = candidate_or_reject(row)
        if candidate is None:
            rejection_counts[reason] += 1
            continue
        if candidate["song_key"] in seen_keys or candidate["text_hash"] in seen_hashes:
            duplicate_count += 1
            rejection_counts["duplicate"] += 1
            continue
        seen_keys.add(str(candidate["song_key"]))
        seen_hashes.add(str(candidate["text_hash"]))
        accepted_count += 1
        item = (float(candidate["rank_score"]), index, candidate)
        if args.max_manifest_rows is not None and args.max_manifest_rows > 0:
            if len(heap) < args.max_manifest_rows:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        else:
            heap.append(item)
        if args.progress_every and all_count % args.progress_every == 0:
            print(json.dumps({"scanned": all_count, "accepted": accepted_count, "kept_for_manifest": len(heap)}, ensure_ascii=False))

    ranked = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    for rank, row in enumerate(ranked, start=1):
        row["manifest_rank"] = rank

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output_manifest": str(args.output_manifest),
        "scanned_rows": all_count,
        "accepted_unique_candidates": accepted_count,
        "manifest_rows": len(ranked),
        "duplicate_count": duplicate_count,
        "rejection_counts": dict(rejection_counts),
        "score_range": {
            "best": ranked[0]["rank_score"] if ranked else None,
            "worst": ranked[-1]["rank_score"] if ranked else None,
        },
        "top_examples": ranked[:10],
        "criteria": {
            "max_manifest_rows": args.max_manifest_rows,
            "scan_limit": args.scan_limit,
            "read_batch_size": args.read_batch_size,
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    parser.add_argument("--output-manifest", type=Path, default=Path("data/manifests/full_song_openai_ranked_manifest_top250k.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/manifests/full_song_openai_ranked_manifest_top250k_summary.json"))
    parser.add_argument("--max-manifest-rows", type=int, default=250000, help="Keep only the top N candidates by score. Use 0 for every accepted candidate.")
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cmd_build(args)


if __name__ == "__main__":
    main()
