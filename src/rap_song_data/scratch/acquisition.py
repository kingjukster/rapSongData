from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .common import command_record, hash_file, hash_text, iter_jsonl, path_manifest, utc_now, write_json, write_jsonl


GUTENBERG_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
GUTENBERG_ROBOT_POLICY_URL = "https://www.gutenberg.org/policy/robot_access.html"
GUTENBERG_OFFLINE_CATALOGS_URL = "https://www.gutenberg.org/ebooks/offline_catalogs.html"
SOURCE_ID = "project_gutenberg_songbooks"
OPEN_HYMNAL_SOURCE_ID = "open_hymnal"
OPEN_HYMNAL_ABC_URL = "http://openhymnal.org/OpenHymnal2014.06.abc"
OPEN_HYMNAL_COPYING_URL = "http://openhymnal.org/copying.html"
WIKISOURCE_SOURCE_ID = "wikisource_lyrics"
WIKISOURCE_API_URL = "https://en.wikisource.org/w/api.php"
WIKISOURCE_COPYRIGHT_POLICY_URL = "https://en.wikisource.org/wiki/Wikisource:Copyright_policy"
WIKISOURCE_DEFAULT_CATEGORIES = ("Song lyrics", "Ballads", "Hymns", "Songs")
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
ABC_FIELD_RE = re.compile(r"^([A-Z]):\s*(.*)$")
WIKISOURCE_TEMPLATE_RE = re.compile(r"\{\{([^{}\n|]+)(?:\|[^{}\n]*)?\}\}", re.I)
WIKISOURCE_POEM_RE = re.compile(r"<poem\b[^>]*>(.*?)</poem>", re.I | re.S)


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


def seen_gutenberg_ids(normalized_root: Path, *, source_id: str = SOURCE_ID) -> set[str]:
    root = normalized_root / source_id
    if not root.exists():
        return set()
    seen: set[str] = set()
    for records_path in root.glob("*/records.jsonl"):
        for row in iter_jsonl(records_path):
            source_item_id = str(row.get("source_item_id") or "").strip()
            if source_item_id:
                seen.add(source_item_id)
    return seen


def select_gutenberg_candidates(
    catalog: Path,
    *,
    limit: int,
    min_score: int = 1,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    excluded = exclude_ids or set()
    candidates: list[tuple[int, int, dict[str, str]]] = []
    for row in iter_catalog_rows(catalog):
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        text_id = str(row.get("Text#") or "").strip()
        if not text_id.isdigit():
            continue
        if text_id in excluded:
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


def gutenberg_header(text: str) -> str:
    start = START_RE.search(text)
    header = text[: start.start()] if start else text[:5000]
    return header.replace("\r\n", "\n").replace("\r", "\n").strip()


def review_gutenberg_record(record: dict[str, Any], raw_text: str) -> dict[str, Any]:
    header = gutenberg_header(raw_text)
    header_lower = header.lower()
    issues: list[str] = []
    if (
        "this ebook is for the use of anyone anywhere" not in header_lower
        or "almost no restrictions whatsoever" not in header_lower
    ):
        issues.append("missing_standard_gutenberg_us_unrestricted_notice")
    restricted_markers = [
        "copyrighted ebook",
        "permission of the copyright holder",
        "restricted by copyright law",
        "copyright (c)",
    ]
    for marker in restricted_markers:
        if marker in header_lower:
            issues.append(f"restricted_marker:{marker}")
    if record.get("language") != "en":
        issues.append("not_english")
    if int(record.get("word_count") or 0) < 200:
        issues.append("too_short")
    decision = "approved_release_candidate" if not issues else "quarantine"
    return {
        "source_id": SOURCE_ID,
        "source_item_id": record["source_item_id"],
        "record_id": record["record_id"],
        "title": record.get("title"),
        "authors": record.get("authors"),
        "source_url": record.get("source_url"),
        "source_url_hash": record.get("source_url_hash"),
        "text_sha256": record.get("text_sha256"),
        "normalized_text_sha256": record.get("normalized_text_sha256"),
        "word_count": record.get("word_count"),
        "approx_tokens": record.get("approx_tokens"),
        "rights_decision": decision,
        "rights_tier": "release_candidate" if decision == "approved_release_candidate" else "quarantine",
        "rights_status": "reviewed_us_unrestricted_header" if decision == "approved_release_candidate" else "needs_manual_review",
        "license_id": "project_gutenberg_us_unrestricted_notice",
        "license_evidence": "https://www.gutenberg.org/policy/license.html",
        "license_evidence_scope": "item_header",
        "quality_decision": "accepted" if decision == "approved_release_candidate" else "quarantine",
        "issues": issues,
    }


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
    exclude_ids = seen_gutenberg_ids(normalized_root) if args.skip_seen else set()
    candidates = select_gutenberg_candidates(
        catalog,
        limit=args.candidate_limit,
        min_score=args.min_score,
        exclude_ids=exclude_ids,
    )
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
            "skip_seen": args.skip_seen,
            "excluded_seen_records": len(exclude_ids),
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


def _latest_snapshot(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise ValueError(f"No snapshots found under {root}")
    return sorted(candidates)[-1]


def review_gutenberg(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    normalized_root = Path(args.normalized_dir) / SOURCE_ID
    raw_root = Path(args.raw_dir) / SOURCE_ID
    snapshot_dir = normalized_root / args.snapshot_id if args.snapshot_id else _latest_snapshot(normalized_root)
    snapshot_id = snapshot_dir.name
    raw_snapshot_dir = raw_root / snapshot_id
    records_path = snapshot_dir / "records.jsonl"
    if not records_path.exists():
        raise ValueError(f"Missing Gutenberg records file: {records_path}")
    review_dir = Path(args.output_dir) / SOURCE_ID / "reviews" / snapshot_id
    reviews: list[dict[str, Any]] = []
    approved_tokens = 0
    quarantined_tokens = 0
    for record in iter_jsonl(records_path):
        raw_path = raw_snapshot_dir / "texts" / f"{record['source_item_id']}.txt"
        if not raw_path.exists():
            review = {
                "source_id": SOURCE_ID,
                "source_item_id": record["source_item_id"],
                "record_id": record["record_id"],
                "rights_decision": "quarantine",
                "rights_status": "missing_raw_text",
                "quality_decision": "quarantine",
                "issues": ["missing_raw_text"],
            }
        else:
            review = review_gutenberg_record(
                record,
                raw_path.read_text(encoding="utf-8", errors="ignore"),
            )
        if review["rights_decision"] == "approved_release_candidate":
            approved_tokens += int(review.get("approx_tokens") or 0)
        else:
            quarantined_tokens += int(review.get("approx_tokens") or 0)
        reviews.append(review)

    review_path = review_dir / "item_reviews.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    write_jsonl(review_path, reviews)
    approved = [row for row in reviews if row["rights_decision"] == "approved_release_candidate"]
    quarantined = [row for row in reviews if row["rights_decision"] != "approved_release_candidate"]
    approved_ids = {row["source_item_id"] for row in approved}
    approved_records = [row for row in iter_jsonl(records_path) if row["source_item_id"] in approved_ids]
    write_jsonl(approved_records_path, approved_records)
    admission_allowed = (
        len(quarantined) == 0 and len(approved) > 0
    ) or (args.allow_partial_admission and len(approved) > 0)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": "review-gutenberg",
        "source_id": SOURCE_ID,
        "snapshot_id": snapshot_id,
        "review_policy": {
            "jurisdiction": "US",
            "basis": "Project Gutenberg item header contains the standard unrestricted US notice and no restricted marker before the START marker.",
            "license_policy_url": "https://www.gutenberg.org/policy/license.html",
            "permission_policy_url": "https://www.gutenberg.org/policy/permission.html",
            "manual_review_required_for_quarantine": True,
        },
        "profile_eligibility": ["scratch-core-open-v1", "scratch-research-nc-v1", "scratch-private-extended-v1"],
        "admission_allowed": admission_allowed,
        "admission_scope": "all_reviewed_records" if len(quarantined) == 0 else "approved_records_only",
        "counts": {
            "reviewed_records": len(reviews),
            "approved_records": len(approved),
            "quarantined_records": len(quarantined),
            "approved_approx_tokens": approved_tokens,
            "quarantined_approx_tokens": quarantined_tokens,
        },
        "inputs": {
            "records": path_manifest(records_path),
            "raw_snapshot_dir": str(raw_snapshot_dir),
        },
        "outputs": {
            "item_reviews": path_manifest(review_path),
            "approved_records": path_manifest(approved_records_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(review_dir / "review_manifest.json", manifest)
    write_json(Path(args.output_dir) / SOURCE_ID / "review_manifest.json", manifest)
    return manifest


def ensure_open_hymnal_abc(raw_dir: Path, *, force: bool = False) -> Path:
    abc_path = raw_dir / "OpenHymnal2014.06.abc"
    if not abc_path.exists() or force:
        _urlretrieve(OPEN_HYMNAL_ABC_URL, abc_path)
    return abc_path


def iter_abc_tunes(text: str) -> Iterable[dict[str, Any]]:
    current: list[str] = []
    prelude: list[str] = []
    in_tune = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw_line.startswith("X:"):
            if current:
                yield parse_abc_tune(current)
            current = [*prelude[-24:], raw_line]
            prelude = []
            in_tune = True
            continue
        if in_tune:
            current.append(raw_line)
        else:
            prelude.append(raw_line)
    if current:
        yield parse_abc_tune(current)


def clean_abc_lyric_line(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^\d+\.\s*~?", "", text)
    text = text.replace("~", " ")
    text = text.replace("\\-", "-")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" _", " ").replace("_", "")
    text = text.replace("*", "").strip(" |")
    return text.strip()


def parse_abc_tune(lines: list[str]) -> dict[str, Any]:
    fields: dict[str, list[str]] = {}
    oh_fields: dict[str, list[str]] = {}
    lyrics: list[str] = []
    comments: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%OH"):
            key, _, value = stripped[1:].partition(" ")
            oh_fields.setdefault(key.strip(), []).append(value.strip())
            comments.append(stripped)
            continue
        if stripped.startswith("%"):
            comments.append(stripped)
            continue
        if stripped.startswith("w:"):
            lyric = clean_abc_lyric_line(stripped[2:])
            if lyric:
                lyrics.append(lyric)
            continue
        match = ABC_FIELD_RE.match(stripped)
        if match:
            fields.setdefault(match.group(1), []).append(match.group(2).strip())
    raw = "\n".join(lines).strip()
    title = fields.get("T", [""])[0]
    text = "\n".join(lyrics).strip()
    source_item_id = fields.get("X", [hash_text(raw, digest_size=8)])[0].strip()
    return {
        "source_item_id": source_item_id,
        "title": title,
        "composer_lines": fields.get("C", []),
        "source_lines": fields.get("S", []),
        "oh_fields": oh_fields,
        "lyrics": text,
        "raw_abc": raw,
        "raw_abc_sha256": hash_text(raw, digest_size=32),
        "copyright_lines": [
            line for line in fields.get("C", []) if "copyright" in line.lower()
        ],
    }


def normalized_open_hymnal_record(tune: dict[str, Any], *, retrieved_at: str, abc_manifest: dict[str, Any]) -> dict[str, Any]:
    text = str(tune.get("lyrics") or "").strip()
    source_item_id = str(tune.get("source_item_id") or "")
    title = str(tune.get("title") or f"Open Hymnal {source_item_id}").strip()
    source_url = f"{OPEN_HYMNAL_ABC_URL}#X:{source_item_id}"
    return {
        "record_id": hash_text(f"{OPEN_HYMNAL_SOURCE_ID}:{source_item_id}", digest_size=16),
        "source_id": OPEN_HYMNAL_SOURCE_ID,
        "source_item_id": source_item_id,
        "retrieved_at": retrieved_at,
        "source_url": source_url,
        "source_url_hash": hash_text(source_url, digest_size=16),
        "rights_tier": "release_candidate",
        "rights_status": "item_review_required",
        "license_id": "open_hymnal_item_specific",
        "license_evidence": OPEN_HYMNAL_COPYING_URL,
        "license_evidence_scope": "abc_item_required",
        "title": title,
        "authors": "; ".join(tune.get("composer_lines") or []),
        "issued": "2014.06",
        "language": "en",
        "subjects": "; ".join(tune.get("oh_fields", {}).get("OHTOPICS", [])),
        "bookshelves": "Open Hymnal",
        "locc": None,
        "text_sha256": tune.get("raw_abc_sha256") or abc_manifest["sha256"],
        "normalized_text_sha256": hash_text(text, digest_size=32),
        "source_archive_sha256": abc_manifest["sha256"],
        "dedupe_cluster_id": None,
        "removal_key": f"{OPEN_HYMNAL_SOURCE_ID}:{source_item_id}",
        "split_group_id": None,
        "split_group_status": "pending_deduplication",
        "word_count": len(WORD_RE.findall(text)),
        "approx_tokens": approx_tokens(text),
        "copyright_lines": tune.get("copyright_lines") or [],
        "text": text,
    }


def review_open_hymnal_record(record: dict[str, Any]) -> dict[str, Any]:
    evidence = "\n".join(str(line) for line in record.get("copyright_lines") or [])
    lower = evidence.lower()
    issues: list[str] = []
    if "copyright: public domain" not in lower:
        issues.append("missing_public_domain_item_statement")
    restricted_markers = [
        "all rights reserved",
        "used by permission",
        "provided it is not altered",
        "may be freely reproduced",
        "creative commons",
    ]
    for marker in restricted_markers:
        if marker in lower:
            issues.append(f"restricted_or_non_pd_marker:{marker}")
    if int(record.get("word_count") or 0) < 20:
        issues.append("too_short")
    decision = "approved_release_candidate" if not issues else "quarantine"
    return {
        "source_id": OPEN_HYMNAL_SOURCE_ID,
        "source_item_id": record["source_item_id"],
        "record_id": record["record_id"],
        "title": record.get("title"),
        "authors": record.get("authors"),
        "source_url": record.get("source_url"),
        "source_url_hash": record.get("source_url_hash"),
        "text_sha256": record.get("text_sha256"),
        "normalized_text_sha256": record.get("normalized_text_sha256"),
        "word_count": record.get("word_count"),
        "approx_tokens": record.get("approx_tokens"),
        "rights_decision": decision,
        "rights_tier": "release_candidate" if decision == "approved_release_candidate" else "quarantine",
        "rights_status": "reviewed_public_domain_abc_item" if decision == "approved_release_candidate" else "needs_manual_review",
        "license_id": "open_hymnal_public_domain_item",
        "license_evidence": OPEN_HYMNAL_COPYING_URL,
        "license_evidence_scope": "abc_item_header",
        "quality_decision": "accepted" if decision == "approved_release_candidate" else "quarantine",
        "issues": issues,
    }


def acquire_open_hymnal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    raw_root = Path(args.raw_dir)
    normalized_root = Path(args.normalized_dir)
    output_dir = Path(args.output_dir) / OPEN_HYMNAL_SOURCE_ID
    snapshot_id = args.snapshot_id or started_at.replace(":", "").replace("+", "Z")
    raw_dir = raw_root / OPEN_HYMNAL_SOURCE_ID / snapshot_id
    normalized_dir = normalized_root / OPEN_HYMNAL_SOURCE_ID / snapshot_id
    abc_path = ensure_open_hymnal_abc(raw_dir, force=args.force_download)
    abc_manifest = path_manifest(abc_path)
    abc_text = abc_path.read_text(encoding="utf-8", errors="ignore")
    retrieved_at = utc_now()
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    token_total = 0
    tunes_seen = 0
    for tune in iter_abc_tunes(abc_text):
        tunes_seen += 1
        if len(records) >= args.max_records or token_total >= args.raw_target_tokens:
            break
        record = normalized_open_hymnal_record(tune, retrieved_at=retrieved_at, abc_manifest=abc_manifest)
        if record["word_count"] < args.min_words:
            rejected.append(
                {
                    "source_item_id": record["source_item_id"],
                    "title": record["title"],
                    "reason": "too_short",
                }
            )
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
        "operation": "acquire-open-hymnal",
        "source_id": OPEN_HYMNAL_SOURCE_ID,
        "status": "pilot_complete",
        "snapshot_id": snapshot_id,
        "raw_dir": str(raw_dir),
        "normalized_dir": str(normalized_dir),
        "official_sources": {
            "abc_url": OPEN_HYMNAL_ABC_URL,
            "copying_url": OPEN_HYMNAL_COPYING_URL,
        },
        "selection": {
            "max_records": args.max_records,
            "min_words": args.min_words,
            "raw_target_tokens": args.raw_target_tokens,
        },
        "rights": {
            "training_eligibility": "conditional",
            "admission_status": "not_admitted",
            "note": "Rows are release candidates only after ABC item-level public-domain review.",
        },
        "counts": {
            "tunes_considered": tunes_seen,
            "accepted_records": len(records),
            "rejected_records": len(rejected),
            "approx_tokens": token_total,
        },
        "outputs": {
            "abc": abc_manifest,
            "records": path_manifest(records_path),
            "rejected": path_manifest(rejected_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def review_open_hymnal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    normalized_root = Path(args.normalized_dir) / OPEN_HYMNAL_SOURCE_ID
    snapshot_dir = normalized_root / args.snapshot_id if args.snapshot_id else _latest_snapshot(normalized_root)
    snapshot_id = snapshot_dir.name
    records_path = snapshot_dir / "records.jsonl"
    if not records_path.exists():
        raise ValueError(f"Missing Open Hymnal records file: {records_path}")
    review_dir = Path(args.output_dir) / OPEN_HYMNAL_SOURCE_ID / "reviews" / snapshot_id
    reviews: list[dict[str, Any]] = []
    approved_tokens = 0
    quarantined_tokens = 0
    for record in iter_jsonl(records_path):
        review = review_open_hymnal_record(record)
        if review["rights_decision"] == "approved_release_candidate":
            approved_tokens += int(review.get("approx_tokens") or 0)
        else:
            quarantined_tokens += int(review.get("approx_tokens") or 0)
        reviews.append(review)
    review_path = review_dir / "item_reviews.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    write_jsonl(review_path, reviews)
    approved = [row for row in reviews if row["rights_decision"] == "approved_release_candidate"]
    quarantined = [row for row in reviews if row["rights_decision"] != "approved_release_candidate"]
    approved_ids = {row["source_item_id"] for row in approved}
    approved_records = [row for row in iter_jsonl(records_path) if row["source_item_id"] in approved_ids]
    write_jsonl(approved_records_path, approved_records)
    admission_allowed = (
        len(quarantined) == 0 and len(approved) > 0
    ) or (args.allow_partial_admission and len(approved) > 0)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": "review-open-hymnal",
        "source_id": OPEN_HYMNAL_SOURCE_ID,
        "snapshot_id": snapshot_id,
        "review_policy": {
            "jurisdiction": "US",
            "basis": "ABC item metadata contains a public-domain copyright line and no restricted marker.",
            "copying_policy_url": OPEN_HYMNAL_COPYING_URL,
            "manual_review_required_for_quarantine": True,
        },
        "profile_eligibility": ["scratch-core-open-v1", "scratch-research-nc-v1", "scratch-private-extended-v1"],
        "admission_allowed": admission_allowed,
        "admission_scope": "all_reviewed_records" if len(quarantined) == 0 else "approved_records_only",
        "counts": {
            "reviewed_records": len(reviews),
            "approved_records": len(approved),
            "quarantined_records": len(quarantined),
            "approved_approx_tokens": approved_tokens,
            "quarantined_approx_tokens": quarantined_tokens,
        },
        "inputs": {"records": path_manifest(records_path)},
        "outputs": {
            "item_reviews": path_manifest(review_path),
            "approved_records": path_manifest(approved_records_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(review_dir / "review_manifest.json", manifest)
    write_json(Path(args.output_dir) / OPEN_HYMNAL_SOURCE_ID / "review_manifest.json", manifest)
    return manifest


def wikisource_api(params: dict[str, Any], *, timeout: int = 60, retries: int = 3) -> dict[str, Any]:
    query = {
        **params,
        "format": "json",
        "formatversion": "2",
    }
    url = f"{WIKISOURCE_API_URL}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rapSongData/0.1 private corpus pilot; respectful bounded research"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt >= retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = min(float(retry_after or 5), 30.0)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def seen_wikisource_titles(normalized_root: Path, *, source_id: str = WIKISOURCE_SOURCE_ID) -> set[str]:
    root = normalized_root / source_id
    if not root.exists():
        return set()
    seen: set[str] = set()
    for path in root.glob("*/records.jsonl"):
        for row in iter_jsonl(path):
            title = str(row.get("source_item_id") or "")
            if title:
                seen.add(title)
    return seen


def select_wikisource_candidates(
    *,
    categories: Iterable[str],
    limit: int,
    delay_seconds: float,
    exclude_titles: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_titles = set(exclude_titles or set())
    for category in categories:
        cmcontinue: str | None = None
        while len(selected) < limit:
            params: dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmnamespace": 0,
                "cmlimit": min(50, max(1, limit - len(selected))),
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            payload = wikisource_api(params)
            for page in payload.get("query", {}).get("categorymembers", []):
                title = str(page.get("title") or "")
                if title and title not in seen_titles:
                    selected.append({**page, "category": category})
                    seen_titles.add(title)
                    if len(selected) >= limit:
                        break
            cmcontinue = payload.get("continue", {}).get("cmcontinue")
            if len(selected) >= limit or not cmcontinue:
                break
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        if len(selected) >= limit:
            break
    return selected


def fetch_wikisource_page(title: str) -> dict[str, Any]:
    payload = wikisource_api(
        {
            "action": "query",
            "prop": "revisions|info|categories",
            "titles": title,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "inprop": "url",
            "cllimit": 500,
        }
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise ValueError(f"Missing Wikisource page: {title}")
    return pages[0]


def wikisource_templates(wikitext: str) -> list[str]:
    return [match.group(1).strip() for match in WIKISOURCE_TEMPLATE_RE.finditer(wikitext)]


def clean_wikisource_wikitext(wikitext: str) -> str:
    poem_blocks = WIKISOURCE_POEM_RE.findall(wikitext)
    text = "\n\n".join(poem_blocks) if poem_blocks else wikitext
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^/]*/>", "", text, flags=re.I)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    for _ in range(8):
        updated = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"</?[^>]+>", "", text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        stripped = re.sub(r"^;+\s*", "", stripped)
        stripped = re.sub(r"^={2,}\s*(.*?)\s*={2,}$", r"\1", stripped)
        if stripped.startswith("|") or stripped.startswith("__"):
            continue
        cleaned_lines.append(stripped)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_wikisource_record(
    page: dict[str, Any],
    *,
    category: str,
    retrieved_at: str,
) -> dict[str, Any]:
    revision = page.get("revisions", [{}])[0]
    slot = revision.get("slots", {}).get("main", {})
    wikitext = str(slot.get("content") or "")
    text = clean_wikisource_wikitext(wikitext)
    title = str(page.get("title") or "")
    source_url = str(page.get("fullurl") or f"https://en.wikisource.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}")
    categories = [str(row.get("title") or "") for row in page.get("categories", [])]
    templates = wikisource_templates(wikitext)
    return {
        "record_id": hash_text(f"{WIKISOURCE_SOURCE_ID}:{title}", digest_size=16),
        "source_id": WIKISOURCE_SOURCE_ID,
        "source_item_id": title,
        "retrieved_at": retrieved_at,
        "source_url": source_url,
        "source_url_hash": hash_text(source_url, digest_size=16),
        "rights_tier": "release_candidate",
        "rights_status": "item_review_required",
        "license_id": "wikisource_item_specific",
        "license_evidence": WIKISOURCE_COPYRIGHT_POLICY_URL,
        "license_evidence_scope": "page_categories_and_templates",
        "title": title,
        "authors": None,
        "issued": None,
        "language": str(page.get("pagelanguage") or "en"),
        "subjects": category,
        "bookshelves": "Wikisource",
        "locc": None,
        "pageid": page.get("pageid"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "categories": categories,
        "templates": templates,
        "text_sha256": hash_text(wikitext, digest_size=32),
        "normalized_text_sha256": hash_text(text, digest_size=32),
        "dedupe_cluster_id": None,
        "removal_key": f"{WIKISOURCE_SOURCE_ID}:{title}",
        "split_group_id": None,
        "split_group_status": "pending_deduplication",
        "word_count": len(WORD_RE.findall(text)),
        "approx_tokens": approx_tokens(text),
        "text": text,
    }


def review_wikisource_record(record: dict[str, Any]) -> dict[str, Any]:
    evidence_values = [*record.get("categories", []), *record.get("templates", [])]
    evidence = "\n".join(str(value) for value in evidence_values)
    lower = evidence.lower()
    issues: list[str] = []
    public_domain_markers = ("pd-", "pd/", "pd-old", "public domain", "pd-us", "pd/us")
    if not any(marker in lower for marker in public_domain_markers):
        issues.append("missing_public_domain_page_marker")
    restricted_markers = ("copyrighted", "fair use", "non-free", "permission", "no derivatives", "non-commercial")
    for marker in restricted_markers:
        if marker in lower:
            issues.append(f"restricted_marker:{marker}")
    if int(record.get("word_count") or 0) < 20:
        issues.append("too_short")
    decision = "approved_release_candidate" if not issues else "quarantine"
    return {
        "source_id": WIKISOURCE_SOURCE_ID,
        "source_item_id": record["source_item_id"],
        "record_id": record["record_id"],
        "title": record.get("title"),
        "source_url": record.get("source_url"),
        "source_url_hash": record.get("source_url_hash"),
        "pageid": record.get("pageid"),
        "revision_id": record.get("revision_id"),
        "revision_timestamp": record.get("revision_timestamp"),
        "text_sha256": record.get("text_sha256"),
        "normalized_text_sha256": record.get("normalized_text_sha256"),
        "word_count": record.get("word_count"),
        "approx_tokens": record.get("approx_tokens"),
        "rights_decision": decision,
        "rights_tier": "release_candidate" if decision == "approved_release_candidate" else "quarantine",
        "rights_status": "reviewed_public_domain_wikisource_page" if decision == "approved_release_candidate" else "needs_manual_review",
        "license_id": "wikisource_public_domain_item",
        "license_evidence": WIKISOURCE_COPYRIGHT_POLICY_URL,
        "license_evidence_scope": "page_categories_and_templates",
        "quality_decision": "accepted" if decision == "approved_release_candidate" else "quarantine",
        "issues": issues,
        "evidence_markers": evidence_values,
    }


def acquire_wikisource(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    normalized_root = Path(args.normalized_dir)
    output_dir = Path(args.output_dir) / WIKISOURCE_SOURCE_ID
    snapshot_id = args.snapshot_id or started_at.replace(":", "").replace("+", "Z")
    normalized_dir = normalized_root / WIKISOURCE_SOURCE_ID / snapshot_id
    exclude_titles = seen_wikisource_titles(normalized_root) if args.skip_seen else set()
    candidates = select_wikisource_candidates(
        categories=args.categories,
        limit=args.candidate_limit,
        delay_seconds=args.delay_seconds,
        exclude_titles=exclude_titles,
    )
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    token_total = 0
    retrieved_at = utc_now()
    for candidate in candidates:
        if len(records) >= args.max_records or token_total >= args.raw_target_tokens:
            break
        title = str(candidate.get("title") or "")
        try:
            page = fetch_wikisource_page(title)
            record = normalized_wikisource_record(page, category=str(candidate.get("category") or ""), retrieved_at=retrieved_at)
        except Exception as error:  # pragma: no cover - network edge logging
            rejected.append({"source_item_id": title, "reason": f"fetch_or_parse_error:{type(error).__name__}"})
            continue
        if record["word_count"] < args.min_words:
            rejected.append({"source_item_id": title, "reason": "too_short", "word_count": record["word_count"]})
            continue
        records.append(record)
        token_total += int(record["approx_tokens"])
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)
    records_path = normalized_dir / "records.jsonl"
    rejected_path = normalized_dir / "rejected.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(rejected_path, rejected)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "started_at": started_at,
        "command": command_record(),
        "operation": "acquire-wikisource",
        "source_id": WIKISOURCE_SOURCE_ID,
        "status": "pilot_complete",
        "snapshot_id": snapshot_id,
        "normalized_dir": str(normalized_dir),
        "official_sources": {
            "api_url": WIKISOURCE_API_URL,
            "copyright_policy_url": WIKISOURCE_COPYRIGHT_POLICY_URL,
        },
        "selection": {
            "categories": list(args.categories),
            "candidate_limit": args.candidate_limit,
            "max_records": args.max_records,
            "min_words": args.min_words,
            "raw_target_tokens": args.raw_target_tokens,
            "skip_seen": args.skip_seen,
        },
        "rights": {
            "training_eligibility": "conditional",
            "admission_status": "not_admitted",
            "note": "Rows are release candidates only after page-level public-domain category/template review.",
        },
        "counts": {
            "candidates_considered": len(candidates),
            "accepted_records": len(records),
            "rejected_records": len(rejected),
            "approx_tokens": token_total,
        },
        "outputs": {
            "records": path_manifest(records_path),
            "rejected": path_manifest(rejected_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def review_wikisource(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    normalized_root = Path(args.normalized_dir) / WIKISOURCE_SOURCE_ID
    snapshot_dir = normalized_root / args.snapshot_id if args.snapshot_id else _latest_snapshot(normalized_root)
    snapshot_id = snapshot_dir.name
    records_path = snapshot_dir / "records.jsonl"
    if not records_path.exists():
        raise ValueError(f"Missing Wikisource records file: {records_path}")
    review_dir = Path(args.output_dir) / WIKISOURCE_SOURCE_ID / "reviews" / snapshot_id
    reviews = [review_wikisource_record(record) for record in iter_jsonl(records_path)]
    approved = [row for row in reviews if row["rights_decision"] == "approved_release_candidate"]
    quarantined = [row for row in reviews if row["rights_decision"] != "approved_release_candidate"]
    approved_ids = {row["source_item_id"] for row in approved}
    approved_records = [row for row in iter_jsonl(records_path) if row["source_item_id"] in approved_ids]
    review_path = review_dir / "item_reviews.jsonl"
    approved_records_path = review_dir / "approved_records.jsonl"
    write_jsonl(review_path, reviews)
    write_jsonl(approved_records_path, approved_records)
    admission_allowed = (
        len(quarantined) == 0 and len(approved) > 0
    ) or (args.allow_partial_admission and len(approved) > 0)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "command": command_record(),
        "operation": "review-wikisource",
        "source_id": WIKISOURCE_SOURCE_ID,
        "snapshot_id": snapshot_id,
        "review_policy": {
            "jurisdiction": "US",
            "basis": "Wikisource page categories/templates contain public-domain markers and no restricted markers.",
            "copyright_policy_url": WIKISOURCE_COPYRIGHT_POLICY_URL,
            "manual_review_required_for_quarantine": True,
        },
        "profile_eligibility": ["scratch-core-open-v1", "scratch-research-nc-v1", "scratch-private-extended-v1"],
        "admission_allowed": admission_allowed,
        "admission_scope": "all_reviewed_records" if len(quarantined) == 0 else "approved_records_only",
        "counts": {
            "reviewed_records": len(reviews),
            "approved_records": len(approved),
            "quarantined_records": len(quarantined),
            "approved_approx_tokens": sum(int(row.get("approx_tokens") or 0) for row in approved),
            "quarantined_approx_tokens": sum(int(row.get("approx_tokens") or 0) for row in quarantined),
        },
        "inputs": {"records": path_manifest(records_path)},
        "outputs": {
            "item_reviews": path_manifest(review_path),
            "approved_records": path_manifest(approved_records_path),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(review_dir / "review_manifest.json", manifest)
    write_json(Path(args.output_dir) / WIKISOURCE_SOURCE_ID / "review_manifest.json", manifest)
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
    parser.add_argument("--skip-seen", action="store_true")


def add_gutenberg_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, default=Path("data/corpus_lake/raw"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--allow-partial-admission", action="store_true")


def add_open_hymnal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, default=Path("data/corpus_lake/raw"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--raw-target-tokens", type=int, default=250_000)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--force-download", action="store_true")


def add_open_hymnal_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--allow-partial-admission", action="store_true")


def add_wikisource_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--categories", nargs="+", default=list(WIKISOURCE_DEFAULT_CATEGORIES))
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=25)
    parser.add_argument("--raw-target-tokens", type=int, default=100_000)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--skip-seen", action="store_true")


def add_wikisource_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--normalized-dir", type=Path, default=Path("data/corpus_lake/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/catalog/v2/sources"))
    parser.add_argument("--snapshot-id")
    parser.add_argument("--allow-partial-admission", action="store_true")
