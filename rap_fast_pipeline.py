#!/usr/bin/env python3
"""rap_fast_pipeline.py

Speed-first rap data + training pipeline for local CUDA workflows.
Implements:
  - deterministic curation + labeling scaffold
  - strict canonical schema checks
  - generation/mutation/preference dataset builders
  - non-destructive run logging + immutable run summaries
  - optional training and generation execution helpers
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import textwrap
import time
import hashlib
import logging
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - CUDA stack may be unavailable during some runs
    torch = None

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    detect = None
    _HAS_LANGDETECT = False

try:
    from pydantic import BaseModel, Field, ValidationError, conlist, confloat
    _HAS_PYDANTIC = True
except Exception:
    BaseModel = None
    _HAS_PYDANTIC = False

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

LOGGER = logging.getLogger("rap_fast_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CORE_SCHEMA_VERSION = "v1.0"
REQUIRED_CORE_FIELDS = [
    "song_id",
    "section_id",
    "bar_id",
    "split",
    "source_license",
    "duplicate_cluster_id",
    "quality_score",
    "artist_clean",
    "year",
    "era",
    "rap_family",
    "rap_category",
    "section_type",
    "section_index",
    "bar_index",
    "bar_text",
    "clean_bar_text",
    "word_count",
    "syllable_count",
    "end_word",
    "end_rhyme_key",
    "rhyme_group",
    "rhyme_scheme_window",
    "repetition_score",
    "theme_tags",
    "emotion_tags",
    "energy_level",
    "density_level",
    "narrative_mode",
    "line_role",
    "concreteness_score",
    "cliche_score",
    "toxicity_or_sensitive_flags",
    "generation_prompt",
    "target_completion",
]

SAFE_LICENSES = {
    "original_user_written",
    "licensed_or_public_domain",
    "synthetic_transformed",
}

DEFAULT_BANNED_WORDS = {
    "embed",
    "subscribe",
    "click",
    "ad",
    "lyrics",
    "translation",
    "visit",
    "copyright",
}

SECTION_MARKER_RE = re.compile(r"^\s*[\[\(]?\s*(intro|verse|chorus|pre[- ]?chorus|bridge|outro|adlib|spoken|hook)\s*[\]\):]?\s*$", re.I)
SECTION_TYPES = {"intro", "verse", "chorus", "pre_chorus", "bridge", "outro", "adlib", "spoken", "hook", "unknown"}


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return safe[:80] or "item"


def _file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=12)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_junk_line(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return True
    if len(s) <= 2:
        return True
    if any(tok in s for tok in ("[", "]", "<", ">", "http://", "https://", "www.")):
        return True
    if s.startswith(("verse", "chorus", "intro", "outro", "bridge", "adlib", "hook")) and s.endswith(":"):
        return True
    if s.count("|") > 6:
        return True
    return False


def _normalize_bar_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"\[[^\]]*]", "", t)
    t = re.sub(r"\([^\)]*\)", "", t)
    t = re.sub(r"\{[^\}]*\}", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
def _simple_language_confidence(text: str) -> float:
    if not text:
        return 0.0
    if _HAS_LANGDETECT:
        try:
            return 1.0 if detect(text) == "en" else 0.2
        except Exception:
            pass
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(1, len(text))


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _syllable_count(text: str) -> int:
    words = _word_tokens(text)
    vowels = set("aeiouy")
    total = 0
    for w in words:
        if not w:
            continue
        groups = 0
        prev = False
        for ch in w:
            is_v = ch in vowels
            if is_v and not prev:
                groups += 1
            prev = is_v
        total += max(1, groups)
    return total


def _end_word(text: str) -> str:
    words = _word_tokens(text)
    return words[-1] if words else ""


def _rhyme_key(word: str) -> str:
    if not word:
        return ""
    token = re.sub(r"[^a-z]", "", word.lower())
    if len(token) <= 3:
        return token
    return token[-3:]


def _to_lines(raw_text: str) -> List[str]:
    return [ln.strip() for ln in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def _section_key(label: str) -> str:
    key = label.lower().strip().replace(" ", "_").replace("-", "_")
    return key if key in SECTION_TYPES else "unknown"


def _load_records(path: Path) -> List[dict]:
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                rows.append(json.loads(ln))
        return rows
    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        return list(data)
    if ext in {".parquet", ".pqt"}:
        return pd.read_parquet(path).to_dict("records")
    if ext in {".csv", ".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
        return df.to_dict("records")
    raise ValueError(f"Unsupported input format: {ext}")


def _quality_quality_from_signal(text: str, syllables: int, words: int, repetition: float, lang_score: float, tox_flags: List[str]) -> float:
    score = 1.0
    if words < 3:
        score -= 0.5
    if not text.strip():
        score -= 0.8
    if syllables > 40:
        score -= 0.08
    if repetition > 0.8:
        score -= 0.25
    if lang_score < 0.55:
        score -= 0.35
    score -= 0.2 if tox_flags else 0.0
    return max(0.0, min(1.0, score))


def _subjective_defaults(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    if "money" in lowered or "pain" in lowered or "hurt" in lowered:
        emotions = ["woeful", "introspective"]
    elif "happy" in lowered or "light" in lowered:
        emotions = ["uplifted", "confident"]
    else:
        emotions = ["neutral"]

    if any(tok in lowered for tok in ("night", "late", "clock", "drive", "shift")):
        themes = ["night", "shift", "work", "grit"]
    elif any(tok in lowered for tok in ("love", "heart", "girl", "girl", "her")):
        themes = ["love", "relationship"]
    else:
        themes = ["grind", "ambition"]
    return {
        "theme_tags": themes[:5],
        "emotion_tags": emotions[:5],
        "energy_level": "medium",
        "density_level": "medium",
        "narrative_mode": "first_person" if "i" in lowered else "mixed",
        "line_role": "image",
        "concreteness_score": 0.55,
        "cliche_score": 0.25,
        "filler_score": 0.2,
        "label_confidence": 0.35,
    }


def _parse_sections(record: dict) -> List[dict]:
    if isinstance(record.get("sections"), list) and record["sections"]:
        result = []
        for i, sec in enumerate(record["sections"]):
            section_type = _section_key(str(sec.get("section_type", sec.get("type", "verse"))))
            raw_lines = sec.get("bars", sec.get("text", ""))
            lines = _to_lines(raw_lines) if isinstance(raw_lines, str) else [str(x) for x in raw_lines if str(x).strip()]
            lines = [_normalize_bar_text(x) for x in lines if not _is_junk_line(x)]
            if lines:
                result.append({"section_type": section_type, "section_index": int(sec.get("section_index", i)), "bars": lines})
        if result:
            return result

    lyrics = (
        record.get("lyrics")
        or record.get("lyrics_cleaned")
        or record.get("lyrics_clean")
        or record.get("text")
        or record.get("lyric")
        or record.get("lyric_lines")
        or ""
    )
    if not isinstance(lyrics, str) or not lyrics.strip():
        return []

    sections: List[dict] = []
    current_type = "verse"
    current_lines: List[str] = []
    current_idx = 0

    for raw_line in _to_lines(lyrics):
        m = SECTION_MARKER_RE.match(raw_line)
        if m:
            if current_lines:
                sections.append({
                    "section_type": current_type,
                    "section_index": current_idx,
                    "bars": current_lines,
                })
            matched = m.group(1).replace("-", "_").replace(" ", "_").lower()
            current_type = _section_key("pre_chorus" if matched == "pre_chorus" else matched)
            current_idx += 1
            current_lines = []
            continue
        cleaned = _normalize_bar_text(raw_line)
        if not _is_junk_line(cleaned):
            current_lines.append(cleaned)

    if current_lines:
        sections.append({
            "section_type": current_type,
            "section_index": current_idx,
            "bars": current_lines,
        })
    return sections


def _assign_rhyme_groups(section_rows: List[dict], window: int = 4) -> None:
    rhyme_window = deque(maxlen=window)
    group_map = {}
    next_group = ord("A")

    for row in section_rows:
        key = row["end_rhyme_key"]
        found = None
        for prior_key, group in reversed(rhyme_window):
            if prior_key == key:
                found = group
                break
        if found is None:
            found = chr(next_group)
            next_group = min(ord("Z"), next_group + 1)
        row["rhyme_group"] = found
        row["rhyme_scheme_window"] = window
        rhyme_window.append((key, found))


def _repetition_score(current_tokens: List[str], history: List[List[str]]) -> float:
    if not history:
        return 0.0
    best = 0.0
    for h in history[-4:]:
        inter = len(set(current_tokens) & set(h))
        union = len(set(current_tokens) | set(h))
        score = inter / max(1, union)
        if score > best:
            best = score
    return float(best)


def _toxicity_flags(text: str) -> List[str]:
    lower = text.lower()
    flags = []
    if any(bad in lower for bad in DEFAULT_BANNED_WORDS):
        flags.append("low_confidence_or_artifact_terms")
    if any(tok in lower for tok in ("explicit", "nsfw", "xxx")):
        flags.append("sensitive")
    return flags


def _build_bar_records(raw_records: List[dict], allow_non_english: bool = False, min_word_count: int = 2) -> List[dict]:
    rows: List[dict] = []
    for song in raw_records:
        song_id = str(song.get("song_id") or song.get("id") or _slug(song.get("title", "")) or _slug(song.get("artist", "")))
        artist_clean = str(song.get("artist_clean") or song.get("artist", "unknown"))
        year = int(song.get("year", 0)) if str(song.get("year", "0")).isdigit() else 0
        era = str(song.get("era", "unknown"))
        rap_family = str(song.get("rap_family", "unknown"))
        rap_category = str(song.get("rap_category", "unknown"))
        split = str(song.get("split", "train"))
        source_license = str(song.get("source_license", "unknown"))
        if split not in {"train", "dev", "test"}:
            split = "train"

        song_sections = _parse_sections(song)
        if not song_sections:
            continue

        for section in song_sections:
            section_type = section["section_type"]
            section_index = int(section["section_index"])
            section_lines = section["bars"]
            if not section_lines:
                continue

            section_id = f"{song_id}_{section_type}_{section_index}"
            bar_ids = []
            history_tokens: List[List[str]] = []
            for bar_idx, raw_bar in enumerate(section_lines):
                clean = _normalize_bar_text(raw_bar)
                if not clean:
                    continue
                if len(_word_tokens(clean)) < min_word_count:
                    continue
                lang_score = _simple_language_confidence(clean)
                if not allow_non_english and lang_score < 0.7:
                    continue

                words = _word_tokens(clean)
                rep = _repetition_score(words, history_tokens)
                history_tokens.append(words)
                end_w = _end_word(clean)
                syllables = _syllable_count(clean)
                word_count = len(words)
                end_key = _rhyme_key(end_w)
                quality = _quality_quality_from_signal(clean, syllables, word_count, rep, lang_score, _toxicity_flags(clean))
                bar_id = f"{section_id}_{bar_idx:03d}"
                row = {
                    "song_id": song_id,
                    "section_id": section_id,
                    "bar_id": bar_id,
                    "split": split,
                    "source_license": source_license,
                    "duplicate_cluster_id": "",
                    "quality_score": float(quality),
                    "artist_clean": artist_clean,
                    "year": year,
                    "era": era,
                    "rap_family": rap_family,
                    "rap_category": rap_category,
                    "section_type": section_type,
                    "section_index": section_index,
                    "bar_index": bar_idx,
                    "bar_text": raw_bar,
                    "clean_bar_text": clean,
                    "word_count": word_count,
                    "syllable_count": syllables,
                    "end_word": end_w,
                    "end_rhyme_key": end_key,
                    "rhyme_group": "",
                    "rhyme_scheme_window": 0,
                    "repetition_score": rep,
                    "theme_tags": [],
                    "emotion_tags": [],
                    "energy_level": "unknown",
                    "density_level": "unknown",
                    "narrative_mode": "mixed",
                    "line_role": "image",
                    "concreteness_score": 0.5,
                    "cliche_score": 0.5,
                    "toxicity_or_sensitive_flags": _toxicity_flags(clean),
                    "generation_prompt": "",
                    "target_completion": clean,
                    "schema_version": CORE_SCHEMA_VERSION,
                }
                rows.append(row)
                bar_ids.append(bar_id)

            section_bars = [r for r in rows if r["section_id"] == section_id]
            _assign_rhyme_groups(section_bars)
    return rows


def _assign_duplicate_clusters(rows: List[dict], near_threshold: float = 0.95) -> None:
    clusters: Dict[str, str] = {}
    rep_rows: Dict[str, str] = {}
    buckets: Dict[str, List[str]] = defaultdict(list)
    cluster_no = 0

    if near_threshold >= 1.0:
        for r in rows:
            text = r["clean_bar_text"]
            exact = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            if exact in clusters:
                r["duplicate_cluster_id"] = clusters[exact]
            else:
                cluster_id = f"dup_{cluster_no:06d}"
                cluster_no += 1
                clusters[exact] = cluster_id
                rep_rows[cluster_id] = text
                r["duplicate_cluster_id"] = cluster_id
            buckets[str(len(text) // 40)].append(r["duplicate_cluster_id"])
        return

    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return SequenceMatcher_ratio(a, b)

    for r in rows:
        text = r["clean_bar_text"]
        exact = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if exact in clusters:
            r["duplicate_cluster_id"] = clusters[exact]
            continue

        bucket = str(len(text) // 40)
        cand = buckets[bucket]
        match = None
        for rep_key in cand:
            rep_text = rep_rows[rep_key]
            if similarity(text, rep_text) >= near_threshold:
                match = rep_key
                break
        if match:
            r["duplicate_cluster_id"] = match
            continue

        cluster_id = f"dup_{cluster_no:06d}"
        cluster_no += 1
        r["duplicate_cluster_id"] = cluster_id
        clusters[exact] = cluster_id
        rep_rows[cluster_id] = text
        buckets[bucket].append(cluster_id)


def SequenceMatcher_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _enrich_subjective_labels(rows: List[dict], args: argparse.Namespace) -> Dict[str, int]:
    if args.label_provider == "none":
        for r in rows:
            defaults = _subjective_defaults(r["clean_bar_text"])
            r.update({
                "theme_tags": defaults["theme_tags"],
                "emotion_tags": defaults["emotion_tags"],
                "energy_level": defaults["energy_level"],
                "density_level": defaults["density_level"],
                "narrative_mode": defaults["narrative_mode"],
                "line_role": defaults["line_role"],
                "concreteness_score": defaults["concreteness_score"],
                "cliche_score": defaults["cliche_score"],
            })
            r["label_confidence"] = defaults["label_confidence"]
        return {"manual_audit_count": len(rows)}

    reviewed = 0
    audit = 0

    if args.label_provider == "openai":
        try:
            import openai
        except Exception as exc:
            raise RuntimeError("OpenAI provider selected but openai package is unavailable.") from exc
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        model = args.label_model
        system_prompt = "You are a strict JSON labeler for rap lyrics. Return JSON only."
        schema_msg = textwrap.dedent(
            """
            Return JSON with:
            theme_tags, emotion_tags, energy_level, density_level, narrative_mode, line_role, concreteness_score, cliche_score, filler_score, confidence
            """
        ).strip()

        for i, row in enumerate(rows):
            try:
                text = row["clean_bar_text"]
                user_msg = (
                    f"{schema_msg}\n\n"
                    f"Allowed energy_level: low|medium|high.\n"
                    f"Allowed density_level: sparse|medium|dense.\n"
                    f"Narrative_mode: first_person|second_person|third_person|abstract|mixed.\n"
                    f"Lyrics:\n{text}"
                )
                out = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = out.choices[0].message.content
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ValueError("Invalid JSON payload")
                defaults = _subjective_defaults(text)
                defaults["theme_tags"] = list(payload.get("theme_tags", defaults["theme_tags"]))[:5]
                defaults["emotion_tags"] = list(payload.get("emotion_tags", defaults["emotion_tags"]))[:5]
                defaults["energy_level"] = str(payload.get("energy_level", defaults["energy_level"]))
                defaults["density_level"] = str(payload.get("density_level", defaults["density_level"]))
                defaults["narrative_mode"] = str(payload.get("narrative_mode", defaults["narrative_mode"]))
                defaults["line_role"] = str(payload.get("line_role", defaults["line_role"]))
                defaults["concreteness_score"] = float(payload.get("concreteness_score", defaults["concreteness_score"]))
                defaults["cliche_score"] = float(payload.get("cliche_score", defaults["cliche_score"]))
                defaults["label_confidence"] = float(payload.get("confidence", 0.8))
                row.update({
                    "theme_tags": defaults["theme_tags"],
                    "emotion_tags": defaults["emotion_tags"],
                    "energy_level": defaults["energy_level"] if defaults["energy_level"] in {"low", "medium", "high"} else "medium",
                    "density_level": defaults["density_level"] if defaults["density_level"] in {"sparse", "medium", "dense"} else "medium",
                    "narrative_mode": defaults["narrative_mode"] if defaults["narrative_mode"] in {"first_person", "second_person", "third_person", "abstract", "mixed"} else "mixed",
                    "line_role": defaults["line_role"] or "image",
                    "concreteness_score": defaults["concreteness_score"],
                    "cliche_score": defaults["cliche_score"],
                    "label_confidence": defaults["label_confidence"],
                })
            except Exception:
                reviewed += 1
                r = _subjective_defaults(row["clean_bar_text"])
                row.update({
                    "theme_tags": r["theme_tags"],
                    "emotion_tags": r["emotion_tags"],
                    "energy_level": r["energy_level"],
                    "density_level": r["density_level"],
                    "narrative_mode": r["narrative_mode"],
                    "line_role": r["line_role"],
                    "concreteness_score": r["concreteness_score"],
                    "cliche_score": r["cliche_score"],
                    "label_confidence": r["label_confidence"],
                })
                audit += 1
            reviewed += 1
    else:
        for row in rows:
            d = _subjective_defaults(row["clean_bar_text"])
            row.update({
                "theme_tags": d["theme_tags"],
                "emotion_tags": d["emotion_tags"],
                "energy_level": d["energy_level"],
                "density_level": d["density_level"],
                "narrative_mode": d["narrative_mode"],
                "line_role": d["line_role"],
                "concreteness_score": d["concreteness_score"],
                "cliche_score": d["cliche_score"],
                "label_confidence": d["label_confidence"],
            })
    return {"labeled": reviewed, "manual_audit_count": audit}


def _default_generation_prompt(r: dict) -> str:
    return (
        f"Write a rap {r['section_type']} segment.\n"
        f"Artist vibe: {r['artist_clean']}.\n"
        f"Year/era: {r['year']}/{r['era']}.\n"
        f"Style: {r['rap_family']} / {r['rap_category']}.\n"
        f"Theme tags: {', '.join(r.get('theme_tags', []))}.\n"
        f"Emotion: {', '.join(r.get('emotion_tags', []))}.\n"
        f"Energy: {r['energy_level']}. Rhyme density: {r['density_level']}.\n"
        f"Syllable target: around {r['syllable_count']} per bar.\n"
    ).strip()


def _generate_sections_prompt(rows: List[dict], section_id: str) -> str:
    first = rows[0]
    return (
        f"Generate a controlled continuation for this section:\n"
        f"section_type={first['section_type']} section_id={section_id}\n"
        f"themes={','.join(first.get('theme_tags', []))}\n"
        f"emotion={','.join(first.get('emotion_tags', []))}\n"
        f"rhyme_target={first.get('rhyme_group')}\n"
    ).strip()


def _json_friendly(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_friendly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_friendly(value.tolist())
        except Exception:
            pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)


def _first_from_collection(value: Any, fallback: str = "neutral") -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value or fallback
    if isinstance(value, dict):
        return fallback
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple, set, tuple)):
        if len(value) == 0:
            return fallback
        return next(iter(value))
    return value if value else fallback


def _build_generation_dataset(rows: List[dict], out_path: Path, args: argparse.Namespace) -> int:
    out: List[dict] = []
    for r in rows:
        if r["source_license"] not in SAFE_LICENSES and not args.include_risk:
            continue
        if float(r["quality_score"]) < args.min_quality_score:
            continue
        prompt = _default_generation_prompt(r)
        r["generation_prompt"] = prompt
        out.append({
            "messages": [
                {"role": "system", "content": "Generate original rap lyrics. Do not copy existing songs."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": r["target_completion"]},
            ],
            "metadata": {
                "song_id": r["song_id"],
                "section_id": r["section_id"],
                "bar_id": r["bar_id"],
                "section_type": r["section_type"],
                "bar_index": r["bar_index"],
                "split": r["split"],
                "source_license": r["source_license"],
                "rap_family": r["rap_family"],
                "rap_category": r["rap_category"],
                "quality_score": r["quality_score"],
                "themes": r.get("theme_tags", []),
                "emotions": r.get("emotion_tags", []),
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(_json_friendly(row), ensure_ascii=False))
            f.write("\n")
    return len(out)


def _build_mutation_dataset(rows: List[dict], out_path: Path) -> int:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["song_id"], r["section_id"])].append(r)

    out = []
    for (song_id, section_id), bars in grouped.items():
        bars = sorted(bars, key=lambda x: x["bar_index"])
        for i in range(len(bars) - 2):
            input_bars = bars[i:i+2]
            remaining = len(bars) - (i + 2)
            if remaining < 2:
                continue
            target_count = min(6, max(2, remaining))
            output_bars = bars[i + 2 : i + 2 + target_count]
            if len(output_bars) < 2:
                continue
            first = input_bars[0]
            out.append({
                "input_bars": [b["clean_bar_text"] for b in input_bars],
                "controls": {
                    "mutation_plan": ["specificity_boost", "rhyme_scheme_extend"],
                    "output_bars": len(output_bars),
                    "preserve_meaning": True,
                    "target_emotion": _first_from_collection(first.get("emotion_tags"), "neutral"),
                    "section_type": first["section_type"],
                },
                "output_bars": [b["clean_bar_text"] for b in output_bars],
                "metadata": {
                    "song_id": song_id,
                    "section_id": section_id,
                    "source_license": first["source_license"],
                    "split": first["split"],
                },
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(_json_friendly(row), ensure_ascii=False))
            f.write("\n")
    return len(out)


def _make_bad_reject_from_candidate(candidate: str) -> str:
    words = candidate.split()
    if len(words) <= 3:
        return candidate
    return " ".join(words[: max(3, int(len(words) * 0.65))]) + " ... [filler]"


def _build_preference_dataset(rows: List[dict], out_path: Path, max_pairs: int = 20000) -> int:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["song_id"], r["section_id"])].append(r)

    out = []
    for bars in grouped.values():
        if len(bars) < 2:
            continue
        bars = sorted(bars, key=lambda x: x["quality_score"])
        if len(bars) == 2:
            low, high = bars[0], bars[-1]
        else:
            low, high = bars[0], bars[-1]
        prompt = _generate_sections_prompt(bars, bars[0]["section_id"])
        preferred = high["clean_bar_text"]
        rejected = low["clean_bar_text"] if low["quality_score"] < 0.5 else _make_bad_reject_from_candidate(preferred)
        out.append({
            "prompt": prompt,
            "chosen": preferred,
            "rejected": rejected,
            "metadata": {
                "song_id": bars[0]["song_id"],
                "section_id": bars[0]["section_id"],
                "source_license": bars[0]["source_license"],
                "split": bars[0]["split"],
                "quality_chosen": float(high["quality_score"]),
                "quality_rejected": float(low["quality_score"]),
            },
        })
        if len(out) >= max_pairs:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(_json_friendly(row), ensure_ascii=False))
            f.write("\n")
    return len(out)


def _validate_row(row: dict) -> List[str]:
    errors = []
    for col in REQUIRED_CORE_FIELDS:
        if col not in row:
            errors.append(f"missing:{col}")
    if row.get("schema_version", CORE_SCHEMA_VERSION) != CORE_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    return errors


def _validate_rows(rows: List[dict]) -> int:
    bad = 0
    for row in rows:
        errs = _validate_row(row)
        if errs:
            bad += 1
            if bad <= 25:
                LOGGER.warning("Row validation failed: %s (%s)", row.get("bar_id", "unknown"), ";".join(errs))
    return bad


@contextlib.contextmanager
def _run_log(ctx: Dict[str, Any], run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    run = {
        "start": _now_iso(),
        "command": " ".join(ctx.get("command", [])),
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda if torch else None,
            "pytorch": torch.__version__ if torch else None,
        },
        "metrics": {},
    }
    if torch:
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
    try:
        yield run
        run["status"] = "success"
    except Exception as exc:
        run["status"] = "error"
        run["error"] = str(exc)
        raise
    finally:
        end = time.time()
        run["end"] = _now_iso()
        run["wall_seconds"] = round(end - start, 4)
        if torch and torch.cuda.is_available() and hasattr(torch.cuda, "max_memory_allocated"):
            run["metrics"]["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
        run_path = run_root / "run_summary.json"
        md = _format_run_summary_markdown(run)
        (run_root / "run_summary.md").write_text(md, encoding="utf-8")
        with run_path.open("w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        if not run_root.joinpath("run_metrics.jsonl").exists():
            with run_root.joinpath("run_metrics.jsonl").open("w", encoding="utf-8") as f:
                f.write("")


def _format_run_summary_markdown(run: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        # Run summary

        - command: `{run.get('command', '')}`
        - status: `{run.get('status', 'unknown')}`
        - start: `{run.get('start', '')}`
        - end: `{run.get('end', '')}`
        - wall_seconds: `{run.get('wall_seconds', 0)}`
        - env: {run.get('env', {})}
        - metrics: {run.get('metrics', {})}
        """
    ).strip() + "\n"


def _append_analytics_csv(run: Dict[str, Any], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run_metrics_timeseries.csv"
    row = {
        "timestamp": run.get("start"),
        "command": run.get("command"),
        "status": run.get("status"),
        "wall_seconds": run.get("wall_seconds"),
        "gpu": run.get("env", {}).get("gpu"),
        "cuda": run.get("env", {}).get("cuda"),
        "pytorch": run.get("env", {}).get("pytorch"),
        "peak_vram_gb": run.get("metrics", {}).get("peak_vram_gb"),
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join("" if v is None else str(v) for v in row.values()) + "\n")


def cmd_curate(args: argparse.Namespace):
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache_file)
    arg_dict = {}
    for key, value in vars(args).items():
        if key == "func" or callable(value):
            continue
        if isinstance(value, Path):
            value = str(value)
        arg_dict[key] = value
    metadata = {
        "input_hash": _file_hash(input_path),
        "input_rows": 0,
        "kept_rows": 0,
        "bad_rows": 0,
        "start": _now_iso(),
        "args": arg_dict,
    }

    cache = {}
    if cache_path.exists() and not args.force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("input_hash") == metadata["input_hash"] and Path(cache.get("output")).exists():
            LOGGER.info("Cache hit, reusing %s", cache["output"])
            return cache["output"]

    raw = _load_records(input_path)
    if args.smoke:
        raw = raw[:500]
    metadata["input_rows"] = len(raw)
    rows = _build_bar_records(raw, allow_non_english=args.allow_non_english, min_word_count=args.min_word_count)
    rows = [r for r in rows if r["quality_score"] >= args.min_quality_score]
    metadata["kept_rows"] = len(rows)

    if args.smoke:
        _assign_duplicate_clusters(rows, near_threshold=1.0)
    else:
        _assign_duplicate_clusters(rows)
    _ = _enrich_subjective_labels(rows, args)
    _assign_generation_prompts(rows)
    bad = _validate_rows(rows)
    metadata["bad_rows"] = bad
    metadata["post_filter_rows"] = len([r for r in rows if r["quality_score"] >= args.min_quality_score])
    metadata["duplicates"] = len({r["duplicate_cluster_id"] for r in rows})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    metadata["output"] = str(output_path)
    metadata["end"] = _now_iso()
    _append_metadata(cache_path, metadata)

    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %s (%s rows)", output_path, len(rows))
    return str(output_path)


def _assign_generation_prompts(rows: List[dict]) -> None:
    for r in rows:
        r["generation_prompt"] = _default_generation_prompt(r)


def _append_metadata(cache_path: Path, metadata: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_build_datasets(args: argparse.Namespace):
    source = Path(args.input)
    df = pd.read_parquet(source)
    rows = df.to_dict("records")
    if len(rows) == 0:
        raise ValueError("No rows available for dataset builders.")
    if not args.include_risk:
        rows = [r for r in rows if r.get("source_license", "").lower() in SAFE_LICENSES]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    generation_count = _build_generation_dataset(rows, Path(args.generation_out), args)
    mutation_count = _build_mutation_dataset(rows, Path(args.mutation_out))
    preference_count = _build_preference_dataset(rows, Path(args.preference_out))

    summary = {
        "generation_records": generation_count,
        "mutation_records": mutation_count,
        "preference_records": preference_count,
    }
    (run_dir / "dataset_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Built datasets: gen=%s mutation=%s preference=%s", generation_count, mutation_count, preference_count)


def cmd_train(args: argparse.Namespace):
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    command = [os.sys.executable, str(Path(args.train_script))]
    command += ["--train_file", str(args.train_file)]
    command += ["--model_name_or_path", args.model]
    command += ["--output_dir", str(run_root / "model_output")]
    command += ["--num_train_epochs", str(args.num_train_epochs)]
    command += ["--max_steps", str(args.max_steps)]
    command += ["--learning_rate", str(args.learning_rate)]
    command += ["--per_device_train_batch_size", str(args.per_device_train_batch_size)]
    command += ["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)]
    command += ["--gradient_checkpointing", "true" if args.gradient_checkpointing else "false"]
    command += ["--max_seq_length", str(args.sequence_length)]
    if args.use_8bit_adam:
        command += ["--optim", "paged_adamw_8bit"]
    with _run_log({"command": command, "train": True}, run_root / "run"):
        start = time.time()
        p = subprocess.run(command, text=True, capture_output=True)
        elapsed = round(time.time() - start, 4)
        if p.returncode != 0:
            raise RuntimeError(f"Training command failed ({p.returncode}): {p.stderr[:1200]}")
        (run_root / "run" / "train_stdout.txt").write_text(p.stdout, encoding="utf-8")
        (run_root / "run" / "train_stderr.txt").write_text(p.stderr, encoding="utf-8")
        stats = {"wall_seconds": elapsed, "seq_len": args.sequence_length, "ga": args.gradient_accumulation_steps}
        (run_root / "run" / "run_summary.json").read_text(encoding="utf-8")
        LOGGER.info("Training complete in %ss", elapsed)


def cmd_generate(args: argparse.Namespace):
    if torch is None:
        raise RuntimeError("PyTorch not available for generation.")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    prompt = args.prompt
    if args.prompt_file:
        pfile = Path(args.prompt_file)
        if pfile.exists():
            prompt = pfile.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"prompt_file not found: {args.prompt_file}")

    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    command = ["generate", args.prompt[:40].replace("\n", " ")]

    with _run_log({"command": command, "generate": True}, run_root):
        if torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if args.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        with torch.inference_mode():
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            wall = time.time() - start
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        tps = len(outputs[0]) / max(1e-8, wall)
        result = {
            "prompt": prompt,
            "generated_text": text,
            "wall_seconds": round(wall, 4),
            "tokens_per_second": round(tps, 4),
            "generated_token_count": int(outputs.shape[-1] - inputs["input_ids"].shape[-1]),
            "model": args.model,
            "adapter_path": args.adapter_path or "",
            "settings": vars(args),
        }
        (run_root / "generation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "generation_result.txt").write_text(text, encoding="utf-8")


def cmd_audit(args: argparse.Namespace):
    input_path = Path(args.input)
    out_path = Path(args.out)
    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("No rows available for audit.")

    k = int(args.samples_per_bucket)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Label and dataset audit",
        f"Generated: {_now_iso()}",
        f"Input: `{input_path}`",
        f"Rows: {len(df)}",
        "",
    ]

    def format_bar(row):
        return (
            f"- `{row.get('bar_id', 'n/a')}` | section={row.get('section_type', 'n/a')} | "
            f"song={row.get('song_id', 'n/a')}\n"
            f"  - quality={row.get('quality_score')} cliche={row.get('cliche_score')} repetition={row.get('repetition_score')}\n"
            f"  - themes={row.get('theme_tags')} emotions={row.get('emotion_tags')}\n"
            f"  - text=`{row.get('clean_bar_text', '')}`\n"
        )

    def add_bucket(title: str, source_df: pd.DataFrame):
        lines.append(f"## {title}")
        if source_df is None or len(source_df) == 0:
            lines.append("- no rows")
            lines.append("")
            return
        for _, row in source_df.head(k).iterrows():
            lines.append(format_bar(row))
        lines.append("")

    if "quality_score" in df.columns:
        add_bucket("lowest quality_score", df.sort_values("quality_score", ascending=True))
    else:
        lines.append("## lowest quality_score")
        lines.append("- column missing")
        lines.append("")

    if "cliche_score" in df.columns:
        add_bucket("highest cliche_score", df.sort_values("cliche_score", ascending=False))
    else:
        lines.append("## highest cliche_score")
        lines.append("- column missing")
        lines.append("")

    if "repetition_score" in df.columns:
        add_bucket("highest repetition_score", df.sort_values("repetition_score", ascending=False))
    else:
        lines.append("## highest repetition_score")
        lines.append("- column missing")
        lines.append("")

    lines.append("## section_type = unknown")
    if "section_type" in df.columns:
        unknown = df[df["section_type"].fillna("unknown") == "unknown"]
        if unknown.empty:
            lines.append("- none")
            lines.append("")
        else:
            for _, row in unknown.head(k).iterrows():
                lines.append(format_bar(row))
            lines.append("")
    else:
        lines.append("- column missing")
        lines.append("")

    lines.append("## duplicate clusters with many members")
    if "duplicate_cluster_id" in df.columns:
        dup = df.copy()
        dup["cluster_size"] = dup["duplicate_cluster_id"].map(dup["duplicate_cluster_id"].value_counts())
        dup = dup.sort_values("cluster_size", ascending=False).head(k)
        if dup.empty:
            lines.append("- no duplicate clusters found")
        else:
            for _, row in dup.iterrows():
                lines.append(f"- `{row['duplicate_cluster_id']}` size={int(row['cluster_size'])} example_bar={row['bar_id']}")
    else:
        lines.append("- column missing")
    lines.append("")

    def audit_jsonl(path: str, title: str):
        lines.append(f"## {title}")
        p = Path(path)
        if not p.exists():
            lines.append("- missing file")
            lines.append("")
            return
        with p.open("r", encoding="utf-8") as f:
            cnt = 0
            for raw in f:
                if cnt >= k:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                lines.append(f"- {json.dumps(obj, ensure_ascii=False)[:500]}")
                cnt += 1
        if cnt == 0:
            lines.append("- no rows found")
        lines.append("")

    audit_jsonl("data/sft/rap_generation_sft.jsonl", "Generation SFT examples")
    audit_jsonl("data/sft/rap_mutation_sft.jsonl", "Mutation SFT examples")
    audit_jsonl("data/preferences/rap_quality_pairs.jsonl", "Preference pairs")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote audit report: %s", out_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rap_fast_pipeline")
    parser.add_argument("--config", type=str, default=None, help="Optional yaml config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_curate = sub.add_parser("curate", help="Curate raw lyrics into canonical Parquet.")
    p_curate.add_argument("--input", required=True)
    p_curate.add_argument("--output", default="data/processed/rap_sections_labeled.parquet")
    p_curate.add_argument("--cache_file", default=".cache/rap_fast_pipeline_cache.json")
    p_curate.add_argument("--min-quality-score", type=float, default=0.35)
    p_curate.add_argument("--min-word-count", type=int, default=2)
    p_curate.add_argument("--allow-non-english", action="store_true")
    p_curate.add_argument("--force", action="store_true")
    p_curate.add_argument("--smoke", action="store_true", help="Quick smoke mode: first 500 raw songs only.")
    p_curate.add_argument("--label-provider", choices=["none", "openai", "heuristic"], default="none")
    p_curate.add_argument("--label-model", default="gpt-4.1-mini")
    p_curate.set_defaults(func=cmd_curate)

    p_dataset = sub.add_parser("build-datasets", help="Build generation/mutation/preference datasets")
    p_dataset.add_argument("--input", required=True)
    p_dataset.add_argument("--run-dir", default="data/run_logs")
    p_dataset.add_argument("--generation-out", default="data/sft/rap_generation_sft.jsonl")
    p_dataset.add_argument("--mutation-out", default="data/sft/rap_mutation_sft.jsonl")
    p_dataset.add_argument("--preference-out", default="data/preferences/rap_quality_pairs.jsonl")
    p_dataset.add_argument("--include-risk", action="store_true")
    p_dataset.add_argument("--min-quality-score", type=float, default=0.35)
    p_dataset.set_defaults(func=cmd_build_datasets)

    p_audit = sub.add_parser("audit", help="Build a compact audit markdown report from curated parquet and generated JSONL files.")
    p_audit.add_argument("--input", required=True)
    p_audit.add_argument("--out", default="data/reports/label_audit.md")
    p_audit.add_argument("--samples-per-bucket", type=int, default=20)
    p_audit.set_defaults(func=cmd_audit)

    p_train = sub.add_parser("train", help="Run speed-first SFT training using your current trainer entrypoint.")
    p_train.add_argument("--train-script", default="train_local_cuda.py")
    p_train.add_argument("--train-file", required=True)
    p_train.add_argument("--validation-path", default="")
    p_train.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_train.add_argument("--run-dir", default="data/run_logs")
    p_train.add_argument("--num-train-epochs", type=int, default=1)
    p_train.add_argument("--max-steps", type=int, default=1000)
    p_train.add_argument("--learning-rate", type=float, default=2e-4)
    p_train.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p_train.add_argument("--sequence-length", type=int, default=768)
    p_train.add_argument("--use-8bit-adam", action="store_true")
    p_train.add_argument("--gradient-checkpointing", action="store_true")
    p_train.add_argument("--smoke", action="store_true", help="Run short smoke mode with 200 steps at 512 sequence length.")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Run controlled generation with timing + VRAM logging.")
    p_gen.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_gen.add_argument("--prompt", default="", help="Prompt text for generation.")
    p_gen.add_argument("--prompt-file", default="", help="Optional prompt file path.")
    p_gen.add_argument("--adapter-path", default="")
    p_gen.add_argument("--run-dir", default="data/run_logs/generation")
    p_gen.add_argument("--max-new-tokens", type=int, default=256)
    p_gen.add_argument("--temperature", type=float, default=0.85)
    p_gen.add_argument("--top-p", type=float, default=0.95)
    p_gen.add_argument("--do-sample", action="store_true")
    p_gen.add_argument("--bf16", action="store_true")
    p_gen.set_defaults(func=cmd_generate)
    return parser


def _load_config(path: Optional[str], args: argparse.Namespace) -> argparse.Namespace:
    if not path:
        return args
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; cannot read config.")
    cfg_path = Path(path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for k, v in cfg.items():
        if not hasattr(args, k):
            continue
        if getattr(args, k) in (None, False, 0, "", 0.0):
            setattr(args, k, v)
    return args


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args = _load_config(args.config, args)
    args.min_word_count = getattr(args, "min_word_count", 2)
    args.min_quality_score = getattr(args, "min_quality_score", 0.35)
    if not hasattr(args, "func"):
        raise SystemExit("No subcommand selected")
    args.func(args)


@contextlib.contextmanager
def _run_log(ctx: Dict[str, Any], run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    run = {
        "start": _now_iso(),
        "command": " ".join(ctx.get("command", [])),
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda if torch else None,
            "pytorch": torch.__version__ if torch else None,
        },
        "metrics": {},
        "meta": {"stage": "training" if ctx.get("train") else "generation" if ctx.get("generate") else "other"},
    }
    if torch and torch.cuda.is_available():
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    try:
        yield run
        run["status"] = "success"
    except Exception as exc:  # pragma: no cover - runtime passthrough
        run["status"] = "error"
        run["error"] = str(exc)
        raise
    finally:
        end = time.time()
        run["end"] = _now_iso()
        run["wall_seconds"] = round(end - start, 4)
        if torch and torch.cuda.is_available() and hasattr(torch.cuda, "max_memory_allocated"):
            run["metrics"]["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4)
        run["metrics"].setdefault("comparability_note", ctx.get("command", []))
        run_path = run_root / "run_summary.json"
        md = _format_run_summary_markdown(run)
        (run_root / "run_summary.md").write_text(md, encoding="utf-8")
        with run_path.open("w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        _append_analytics_csv(run, run_root)


def cmd_train(args: argparse.Namespace):
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    seq_len = 512 if args.smoke else args.sequence_length
    max_steps = 200 if args.smoke else args.max_steps
    ga_steps = 4 if args.smoke else args.gradient_accumulation_steps
    stage = "smoke" if args.smoke else "train"
    run_root = run_root / stage
    run_root.mkdir(parents=True, exist_ok=True)

    train_script = Path(args.train_script)
    if train_script.name.lower().startswith("train_local_cuda"):
        command = [os.sys.executable, str(train_script)]
        command += ["--train-path", str(args.train_file)]
        command += ["--base-model", args.model]
        command += ["--output-dir", str(run_root / "model_output")]
        command += ["--max-steps", str(max_steps)]
        command += ["--sequence-length", str(seq_len)]
        command += ["--validation-path", args.validation_path if args.validation_path else str(args.train_file)]
    else:
        command = [os.sys.executable, str(train_script)]
        command += ["--train_file", str(args.train_file)]
        command += ["--model_name_or_path", args.model]
        command += ["--output_dir", str(run_root / "model_output")]
        command += ["--num_train_epochs", str(args.num_train_epochs)]
        command += ["--max_steps", str(max_steps)]
        command += ["--learning-rate", str(args.learning_rate)]
        command += ["--per_device_train_batch_size", str(args.per_device_train_batch_size)]
        command += ["--gradient_accumulation_steps", str(ga_steps)]
        command += ["--gradient_checkpointing", "true" if args.gradient_checkpointing else "false"]
        command += ["--max_seq_length", str(seq_len)]
        command += ["--comparability_note", "smoke" if args.smoke else "default"]
        if args.use_8bit_adam:
            command += ["--optim", "paged_adamw_8bit"]

    with _run_log({"command": [str(x) for x in command], "train": True}, run_root / "run"):
        start = time.time()
        proc = subprocess.run(command, text=True, capture_output=True)
        elapsed = round(time.time() - start, 4)
        (run_root / "run" / "train_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (run_root / "run" / "train_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Training command failed ({proc.returncode}): {proc.stderr[:1200]}")
        run_summary = {
            "wall_seconds": elapsed,
            "seq_len": seq_len,
            "gradient_accumulation_steps": ga_steps,
            "steps": max_steps,
        }
        (run_root / "run" / "run_metrics.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Training complete in %ss", elapsed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rap_fast_pipeline")
    parser.add_argument("--config", type=str, default=None, help="Optional yaml config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_curate = sub.add_parser("curate", help="Curate raw lyrics into canonical Parquet.")
    p_curate.add_argument("--input", required=True)
    p_curate.add_argument("--output", default="data/processed/rap_sections_labeled.parquet")
    p_curate.add_argument("--cache_file", default=".cache/rap_fast_pipeline_cache.json")
    p_curate.add_argument("--min-quality-score", type=float, default=0.35)
    p_curate.add_argument("--min-word-count", type=int, default=2)
    p_curate.add_argument("--allow-non-english", action="store_true")
    p_curate.add_argument("--force", action="store_true")
    p_curate.add_argument("--smoke", action="store_true", help="Quick smoke mode: first 500 raw songs only.")
    p_curate.add_argument("--label-provider", choices=["none", "openai", "heuristic"], default="none")
    p_curate.add_argument("--label-model", default="gpt-4.1-mini")
    p_curate.set_defaults(func=cmd_curate)

    p_dataset = sub.add_parser("build-datasets", help="Build generation/mutation/preference datasets")
    p_dataset.add_argument("--input", required=True)
    p_dataset.add_argument("--run-dir", default="data/run_logs")
    p_dataset.add_argument("--generation-out", default="data/sft/rap_generation_sft.jsonl")
    p_dataset.add_argument("--mutation-out", default="data/sft/rap_mutation_sft.jsonl")
    p_dataset.add_argument("--preference-out", default="data/preferences/rap_quality_pairs.jsonl")
    p_dataset.add_argument("--include-risk", action="store_true")
    p_dataset.add_argument("--min-quality-score", type=float, default=0.35)
    p_dataset.set_defaults(func=cmd_build_datasets)

    p_train = sub.add_parser("train", help="Run speed-first SFT training using your current trainer entrypoint.")
    p_train.add_argument("--train-script", default="train_local_cuda.py")
    p_train.add_argument("--train-file", required=True)
    p_train.add_argument("--validation-path", default="")
    p_train.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_train.add_argument("--run-dir", default="data/run_logs")
    p_train.add_argument("--num-train-epochs", type=int, default=1)
    p_train.add_argument("--max-steps", type=int, default=1000)
    p_train.add_argument("--learning-rate", type=float, default=2e-4)
    p_train.add_argument("--per-device-train-batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p_train.add_argument("--sequence-length", type=int, default=768)
    p_train.add_argument("--use-8bit-adam", action="store_true")
    p_train.add_argument("--gradient-checkpointing", action="store_true")
    p_train.add_argument("--smoke", action="store_true", help="Run short smoke mode with 200 steps at 512 sequence length.")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Run controlled generation with timing + VRAM logging.")
    p_gen.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_gen.add_argument("--prompt", default="", help="Prompt text for generation.")
    p_gen.add_argument("--prompt-file", default="", help="Optional prompt file path.")
    p_gen.add_argument("--adapter-path", default="")
    p_gen.add_argument("--run-dir", default="data/run_logs/generation")
    p_gen.add_argument("--max-new-tokens", type=int, default=256)
    p_gen.add_argument("--temperature", type=float, default=0.85)
    p_gen.add_argument("--top-p", type=float, default=0.95)
    p_gen.add_argument("--do-sample", action="store_true")
    p_gen.add_argument("--bf16", action="store_true")
    p_gen.set_defaults(func=cmd_generate)
    p_audit = sub.add_parser("audit", help="Build a compact audit markdown report from curated parquet and generated JSONL files.")
    p_audit.add_argument("--input", required=True)
    p_audit.add_argument("--out", default="data/reports/label_audit.md")
    p_audit.add_argument("--samples-per-bucket", type=int, default=20)
    p_audit.set_defaults(func=cmd_audit)
    return parser


if __name__ == "__main__":
    main()
