from __future__ import annotations

import argparse
import csv
import gzip
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .common import command_record, hash_file, hash_text, path_manifest, utc_now, write_json, write_jsonl


GUTENBERG_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
GUTENBERG_ROBOT_POLICY_URL = "https://www.gutenberg.org/policy/robot_access.html"
GUTENBERG_OFFLINE_CATALOGS_URL = "https://www.gutenberg.org/ebooks/offline_catalogs.html"
SOURCE_ID = "project_gutenberg_songbooks"
KEYWORDS = (
    "song",
    "songs",
    "songbook",
    "ballad",
    "ballads",
    "hymn",
    "hymns",
    "libretto",
    "libretti",
    "opera",
    "poem",
    "poems",
    "poetry",
    "verse",
    "drama",
    "plays",
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


def _urlretrieve(url: str, path: Path, *, timeout: int = 60) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "rapSongData corpus pilot"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        temporary.write_bytes(response.read())
    temporary.replace(path)


def ensure_catalog(raw_dir: Path, *, force: bool = False) -> Path:
    catalog = raw_dir / "catalog" / "pg_catalog.csv.gz"
    if not catalog.exists() or force:
        _urlretrieve(GUTENBERG_CATALOG_URL, catalog)
    return catalog


def iter_catalog_rows(catalog: Path) -> Iterable[dict[str, str]]:
    with gzip.open(catalog, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def candidate_score(row: dict[str, str]) -> int:
    haystack = " ".join(
        str(row.get(field) or "").lower()
        for field in ("Title", "Subjects", "Bookshelves", "LoCC")
    )
    return sum(3 if keyword in str(row.get("Title", "")).lower() else 1 for keyword in KEYWORDS if keyword in haystack)


def select_gutenberg_candidates(catalog: Path, *, limit: int, min_score: int = 1) -> list[dict[str, str]]:
    candidates: list[tuple[int, int, dict[str, str]]] = []
    for row in iter_catalog_rows(catalog):
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        text_id = str(row.get("Text#") or "").strip()
        if not text_id.isdigit():
            continue
        score = candidate_score(row)
        if score >= min_score:
            candidates.append((score, int(text_id), row))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in candidates[:limit]]


def text_urls(text_id: str) -> list[str]:
    return [
        f"https://www.gutenberg.org/files/{text_id}/{text_id}-0.txt",
        f"https://www.gutenberg.org/files/{text_id}/{text_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{text_id}/pg{text_id}.txt",
    ]


def fetch_text(text_id: str, raw_text_dir: Path, *, delay_seconds: float) -> tuple[Path | None, str | None]:
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    for url in text_urls(text_id):
        target = raw_text_dir / f"{text_id}.txt"
        if target.exists():
            return target, url
        try:
            _urlretrieve(url, target)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            return target, url
        except urllib.error.HTTPError:
            continue
        except urllib.error.URLError:
            continue
    return None, None


def strip_gutenberg_wrapper(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = END_RE.search(text)
    if end:
        text = text[: end.start()]
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def approx_tokens(text: str) -> int:
    return int(round(len(WORD_RE.findall(text)) * 1.3))


def normalized_record(row: dict[str, str], text: str, raw_path: Path, source_url: str, retrieved_at: str) -> dict[str, Any]:
    text_id = str(row["Text#"])
    clean = strip_gutenberg_wrapper(text)
    return {
        "record_id": hash_text(f"{SOURCE_ID}:{text_id}", digest_size=16),
        "source_id": SOURCE_ID,
        "source_item_id": text_id,
        "retrieved_at": retrieved_at,
        "source_url": source_url,
        "source_url_hash": hash_text(source_url, digest_size=16),
        "rights_tier": "release_candidate",
        "rights_status": "item_review_required",
        "license_id": "project_gutenberg_terms_us_public_domain_or_permissioned",
        "license_evidence": GUTENBERG_OFFLINE_CATALOGS_URL,
        "license_evidence_scope": "item_required",
        "title": row.get("Title"),
        "authors": row.get("Authors"),
        "issued": row.get("Issued"),
        "language": row.get("Language"),
        "subjects": row.get("Subjects"),
        "bookshelves": row.get("Bookshelves"),
        "locc": row.get("LoCC"),
        "text_sha256": hash_file(raw_path),
        "normalized_text_sha256": hash_text(clean, digest_size=32),
        "dedupe_cluster_id": None,
        "removal_key": f"{SOURCE_ID}:{text_id}",
        "split_group_id": None,
        "split_group_status": "pending_deduplication",
        "word_count": len(WORD_RE.findall(clean)),
        "approx_tokens": approx_tokens(clean),
        "text": clean,
    }


def acquire_gutenberg(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    raw_root = Path(args.raw_dir)
    normalized_root = Path(args.normalized_dir)
    output_dir = Path(args.output_dir) / SOURCE_ID
    snapshot_id = args.snapshot_id or started_at.replace(":", "").replace("+", "Z")
    raw_dir = raw_root / SOURCE_ID / snapshot_id
    raw_text_dir = raw_dir / "texts"
    normalized_dir = normalized_root / SOURCE_ID / snapshot_id
    catalog = ensure_catalog(raw_dir, force=args.force_catalog)
    candidates = select_gutenberg_candidates(catalog, limit=args.candidate_limit, min_score=args.min_score)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    retrieved_at = utc_now()
    token_total = 0

    for row in candidates:
        if len(records) >= args.max_records:
            break
        if token_total >= args.raw_target_tokens:
            break
        text_id = str(row["Text#"])
        raw_path, source_url = fetch_text(text_id, raw_text_dir, delay_seconds=args.delay_seconds)
        if raw_path is None or source_url is None:
            rejected.append({"source_item_id": text_id, "title": row.get("Title"), "reason": "text_download_failed"})
            continue
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
        record = normalized_record(row, text, raw_path, source_url, retrieved_at)
        if record["word_count"] < args.min_words:
            rejected.append({"source_item_id": text_id, "title": row.get("Title"), "reason": "too_short"})
            continue
        records.append(record)
        token_total += int(record["approx_tokens"])

    records_path = normalized_dir / "records.jsonl"
    rejected_path = normalized_dir / "rejected.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(rejected_path, rejected)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "started_at": started_at,
        "command": command_record(),
        "operation": "acquire-gutenberg",
        "source_id": SOURCE_ID,
        "status": "pilot_complete",
        "snapshot_id": snapshot_id,
        "raw_dir": str(raw_dir),
        "normalized_dir": str(normalized_dir),
        "official_sources": {
            "catalog_url": GUTENBERG_CATALOG_URL,
            "robot_policy_url": GUTENBERG_ROBOT_POLICY_URL,
            "offline_catalogs_url": GUTENBERG_OFFLINE_CATALOGS_URL,
        },
        "selection": {
            "keywords": list(KEYWORDS),
            "candidate_limit": args.candidate_limit,
            "max_records": args.max_records,
            "min_score": args.min_score,
            "min_words": args.min_words,
            "raw_target_tokens": args.raw_target_tokens,
        },
        "rights": {
            "training_eligibility": "conditional",
            "admission_status": "not_admitted",
            "note": "Rows are release candidates only after item-level rights review; no profile admission is performed here.",
        },
        "counts": {
            "candidates_considered": len(candidates),
            "accepted_records": len(records),
            "rejected_records": len(rejected),
            "approx_tokens": token_total,
        },
        "outputs": {
            "catalog": path_manifest(catalog),
            "records": path_manifest(records_path),
            "rejected": path_manifest(rejected_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def add_gutenberg_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, default=Path("data/corpus_lake/raw"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=25)
    parser.add_argument("--raw-target-tokens", type=int, default=250_000)
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--force-catalog", action="store_true")
