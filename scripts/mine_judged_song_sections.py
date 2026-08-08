#!/usr/bin/env python3
"""Mine, rank, and deduplicate 12-line excerpts from the 1k judged songs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
HEADER_RE = re.compile(r"^\s*\[([^\]]{1,120})\]\s*$")
ARTIFACT_RE = re.compile(
    r"(?:genius\.com|you might also like|\bembed\b|https?://|www\.|lyrics taken from|"
    r"contributor|transcriber|translation|romanization|tracklist|album art)", re.I,
)
UNFINISHED_RE = re.compile(
    r"(?:[,;:\-(]|\b(?:and|but|because|cause|when|if|the|a|an|to|of|for|with))\s*$", re.I
)
PROFANITY_SLUR_RE = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch(?:es)?|nigg(?:a|er)s?|fagg?ot\w*|cunt\w*|"
    r"dick|pussy|motherfucker\w*|hoe(?:s)?|whore(?:s)?|slut(?:s)?)\b", re.I,
)
SEXUAL_RE = re.compile(r"\b(?:sex|sexual|orgasm|porn|naked|nude|blowjob|handjob|rape|molest\w*)\b", re.I)
DRUG_RE = re.compile(
    r"\b(?:cocaine|heroin|fentanyl|meth|crack|percocet|xanax|lean|codeine|"
    r"drug deal\w*|sell(?:ing)? dope|smok(?:e|ing) weed)\b", re.I,
)
GRAPHIC_VIOLENCE_RE = re.compile(
    r"\b(?:blood splatter|brains? (?:on|out)|decapitat\w*|dismember\w*|"
    r"stab(?:bed|bing)? .* (?:neck|chest)|shoot .* (?:head|face))\b", re.I,
)
GENERIC_RE = re.compile(
    r"\b(?:never give up|follow your dreams|reach for the stars|rise above|"
    r"believe in yourself|against all odds|keep pushing|make it to the top)\b", re.I,
)
CONCRETE_TERMS = {
    "alley", "asphalt", "basement", "blood", "brick", "bus", "car", "cash", "chain",
    "city", "clock", "concrete", "corner", "door", "glass", "gun", "hallway", "hand",
    "kitchen", "light", "mirror", "night", "rain", "road", "room", "shoe", "smoke",
    "street", "subway", "table", "train", "window", "winter",
}
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "you", "for", "but", "not",
    "are", "was", "were", "have", "has", "had", "they", "them", "their", "what", "when",
    "where", "who", "how", "all", "just", "like", "into", "out", "got", "get", "been",
}
HARD_PARENT_FLAGS = {
    "scrape_artifact", "non_lyric", "broken_structure", "incoherent", "generic_filler",
    "repetition_collapse", "artist_or_metadata_leak",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def normalized_line(line: str) -> str:
    return " ".join(words(line))


def section_kind(label: str) -> str:
    value = label.lower()
    if "verse" in value:
        return "verse"
    if any(token in value for token in ("hook", "chorus", "refrain")):
        return "hook"
    if "bridge" in value:
        return "bridge"
    if any(token in value for token in ("speaker", "interlude", "skit")):
        return "speaker"
    return "untagged"


def split_boundary_sections(lyrics: str) -> list[dict[str, Any]]:
    """Split on explicit headers and blank stanzas, preserving source line numbers."""
    sections: list[dict[str, Any]] = []
    label = "untagged"
    current: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append({"label": label, "kind": section_kind(label), "lines": current})
        current = []

    for source_line, raw in enumerate(str(lyrics or "").splitlines(), 1):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            flush()
            continue
        header = HEADER_RE.match(line)
        if header:
            flush()
            label = header.group(1).strip()
            continue
        current.append((source_line, line))
    flush()
    return sections


def parent_gate_failures(row: dict[str, Any], family: str) -> int:
    fields = ["overall", "technical_rhyme", "flow_cadence", "coherence", "thematic_depth", "ending_strength"]
    failures = sum(int(row.get(field) or 0) < 4 for field in fields)
    failures += int(int(row.get("family_compliance") or 0) != 5)
    if family == "clean":
        failures += int(int(row.get("cleanliness") or 0) != 5)
        failures += int(int(row.get("genericness") or 6) > 2)
    return failures


def parent_eligible(row: dict[str, Any], family: str) -> bool:
    flags = set(row.get("critical_failure_flags") or []) - {"none"}
    if flags & HARD_PARENT_FLAGS:
        return False
    if family == "technical":
        return bool(row.get("strict_pass")) or parent_gate_failures(row, family) <= 1 or int(row.get("technical_rhyme") or 0) >= 4
    if family == "story":
        return (
            not bool(row.get("truncated"))
            and int(row.get("overall") or 0) >= 3
            and int(row.get("coherence") or 0) >= 4
            and int(row.get("thematic_depth") or 0) >= 4
            and int(row.get("imagery") or 0) >= 3
            and int(row.get("ending_strength") or 0) >= 4
            and int(row.get("genericness") or 6) <= 3
        )
    return (
        "unsafe_for_clean" not in flags
        and int(row.get("cleanliness") or 0) == 5
        and int(row.get("genericness") or 6) <= 3
        and int(row.get("coherence") or 0) >= 2
    )


def vowel_syllables(token: str) -> int:
    clean = re.sub(r"[^a-z]", "", token.lower())
    if not clean:
        return 0
    groups = re.findall(r"[aeiouy]+", clean)
    count = len(groups)
    if clean.endswith("e") and not clean.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def rhyme_tail(token: str, chars: int = 3) -> str:
    clean = re.sub(r"[^a-z0-9]", "", token.lower())
    return clean[-chars:] if len(clean) > chars else clean


def local_metrics(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    tokens = words(text)
    endings = [rhyme_tail(words(line)[-1]) for line in lines if words(line)]
    repeated_endings = sum(count for count in Counter(endings).values() if count > 1)
    chains = sum(endings[index] == endings[index - 1] for index in range(1, len(endings)))
    internal = 0
    multis = 0
    for line in lines:
        line_tokens = [token for token in words(line) if len(token) > 3]
        tails = [rhyme_tail(token) for token in line_tokens]
        internal += int(any(count >= 2 for count in Counter(tails).values()))
        multis += sum(1 for token in line_tokens if vowel_syllables(token) >= 2 and tails.count(rhyme_tail(token)) >= 2)
    normalized = [normalized_line(line) for line in lines]
    repeated_lines = len(normalized) - len(set(normalized))
    syllables = [sum(vowel_syllables(token) for token in words(line)) for line in lines]
    avg_syllables = sum(syllables) / len(syllables)
    syllable_variance = math.sqrt(sum((value - avg_syllables) ** 2 for value in syllables) / len(syllables))
    unique = {token for token in tokens if len(token) > 2}
    content = [token for token in tokens if len(token) > 3 and token not in STOPWORDS]
    topic_reuse = sum(count for count in Counter(content).values() if count >= 2) / max(1, len(content))
    final = lines[-1]
    return {
        "word_count": len(tokens),
        "end_rhyme_density": round(repeated_endings / 12, 4),
        "internal_rhyme_density": round(internal / 12, 4),
        "multisyllabic_rhyme_count": multis,
        "rhyme_chain_continuity": round(chains / 11, 4),
        "lexical_diversity": round(len(unique) / max(1, len(tokens)), 4),
        "concrete_imagery_count": sum(token in CONCRETE_TERMS for token in tokens),
        "topic_continuity": round(topic_reuse, 4),
        "repeated_line_ratio": round(repeated_lines / 12, 4),
        "syllable_variance": round(syllable_variance, 3),
        "final_complete": not bool(UNFINISHED_RE.search(final)) and len(words(final)) >= 4,
        "profanity_slur_count": len(PROFANITY_SLUR_RE.findall(text)),
        "sexual_content_count": len(SEXUAL_RE.findall(text)),
        "drug_content_count": len(DRUG_RE.findall(text)),
        "graphic_violence_count": len(GRAPHIC_VIOLENCE_RE.findall(text)),
        "generic_phrase_count": len(GENERIC_RE.findall(text)),
        "artifact_count": len(ARTIFACT_RE.findall(text)),
    }


def local_score(metrics: dict[str, Any], parent: dict[str, Any], family: str) -> float:
    parent_signal = sum(float(parent.get(field) or 0) for field in ("overall", "coherence", "thematic_depth", "ending_strength")) / 20
    repetition_penalty = metrics["repeated_line_ratio"] * 3
    fragment_penalty = 0 if metrics["final_complete"] else 0.8
    if family == "technical":
        score = (
            metrics["end_rhyme_density"] * 2.2 + metrics["internal_rhyme_density"] * 2.4
            + min(metrics["multisyllabic_rhyme_count"] / 8, 1) * 1.6
            + metrics["rhyme_chain_continuity"] * 1.1 + metrics["lexical_diversity"]
            + min(metrics["concrete_imagery_count"] / 4, 1) * 0.6 + metrics["topic_continuity"]
            + parent_signal * 0.7 - repetition_penalty - fragment_penalty
            - max(0, metrics["syllable_variance"] - 5) * 0.1
        )
    elif family == "clean":
        score = (
            metrics["lexical_diversity"] * 1.4 + min(metrics["concrete_imagery_count"] / 4, 1) * 1.2
            + metrics["topic_continuity"] * 1.6 + metrics["internal_rhyme_density"] * 0.5
            + parent_signal * 1.0 - repetition_penalty - fragment_penalty
            - metrics["generic_phrase_count"] * 1.5
        )
    else:
        story_parent_signal = sum(
            float(parent.get(field) or 0)
            for field in ("overall", "coherence", "thematic_depth", "imagery", "ending_strength")
        ) / 25
        score = (
            metrics["lexical_diversity"] * 1.5
            + min(metrics["concrete_imagery_count"] / 5, 1) * 1.7
            + metrics["topic_continuity"] * 1.8
            + metrics["internal_rhyme_density"] * 0.35
            + story_parent_signal * 1.5
            - repetition_penalty - fragment_penalty
            - metrics["generic_phrase_count"] * 1.8
            - max(0, metrics["syllable_variance"] - 6) * 0.05
        )
    return round(score, 5)


def clean_safe(metrics: dict[str, Any]) -> bool:
    return all(metrics[field] == 0 for field in (
        "profanity_slur_count", "sexual_content_count", "drug_content_count", "graphic_violence_count", "artifact_count",
    ))


def minhash_signature(text: str, permutations: int = 24) -> tuple[int, ...]:
    tokens = words(text)
    shingles = {" ".join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}
    if not shingles:
        shingles = {""}
    signature = []
    for seed in range(permutations):
        signature.append(min(int.from_bytes(hashlib.sha1(f"{seed}:{value}".encode()).digest()[:8], "big") for value in shingles))
    return tuple(signature)


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left)


def mine_windows(parent: dict[str, Any], lyrics: str, family: str, stride: int) -> Iterable[dict[str, Any]]:
    for section_index, section in enumerate(split_boundary_sections(lyrics)):
        if section["kind"] in {"hook", "bridge", "speaker"}:
            continue
        section_lines = section["lines"]
        if len(section_lines) < 12:
            continue
        for start in range(0, len(section_lines) - 11, stride):
            window = section_lines[start:start + 12]
            lines = [text for _, text in window]
            metrics = local_metrics(lines)
            if metrics["artifact_count"] or metrics["repeated_line_ratio"] > 0.16 or not metrics["final_complete"]:
                continue
            if family == "clean" and not clean_safe(metrics):
                continue
            text = "\n".join(lines)
            digest = hashlib.sha1("\n".join(normalized_line(line) for line in lines).encode()).hexdigest()
            yield {
                "section_id": f"song:{parent['song_key']}:{family}:{window[0][0]}-{window[-1][0]}:{digest[:12]}",
                "song_key": str(parent["song_key"]), "family": family,
                "source_start_line": window[0][0], "source_end_line": window[-1][0],
                "section_label": section["label"], "section_kind": section["kind"], "line_count": 12,
                "text_hash": digest, "text": text, "local_metrics": metrics,
                "local_score": local_score(metrics, parent, family),
                "parent_judgment": {key: parent.get(key) for key in (
                    "overall", "technical_rhyme", "flow_cadence", "coherence", "thematic_depth",
                    "imagery", "ending_strength", "family_compliance", "cleanliness", "genericness",
                    "critical_failure_flags", "strict_pass",
                )},
            }


def select_deduped(candidates: list[dict[str, Any]], limit: int, per_song_cap: int, threshold: float) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    signatures: list[tuple[int, ...]] = []
    line_sets: list[set[str]] = []
    song_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in sorted(candidates, key=lambda item: (-item["local_score"], item["section_id"])):
        if len(selected) >= limit:
            break
        if song_counts[row["song_key"]] >= per_song_cap:
            counts["per_song_cap"] += 1
            continue
        lines = {normalized_line(line) for line in row["text"].splitlines()}
        if any(len(lines & existing) >= 10 for existing in line_sets):
            counts["line_overlap_duplicate"] += 1
            continue
        signature = minhash_signature(row["text"])
        if any(minhash_similarity(signature, existing) >= threshold for existing in signatures):
            counts["minhash_duplicate"] += 1
            continue
        selected.append(row)
        signatures.append(signature)
        line_sets.append(lines)
        song_counts[row["song_key"]] += 1
    return selected, counts


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=Path("data/reviews/song_quality_judge_gpt55_calibration_1k.jsonl"))
    parser.add_argument("--songs", type=Path, default=Path("data/reviews/song_triage_candidate_pool_v1.parquet"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/section_mining/section_mining_calibration_raw_3k.jsonl"))
    parser.add_argument("--selected-output", type=Path, default=Path("data/section_mining/section_mining_calibration_selected_800.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/section_mining/section_mining_calibration_summary.json"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--raw-per-family", type=int, default=1500)
    parser.add_argument("--selected-per-family", type=int, default=400)
    parser.add_argument("--per-song-cap", type=int, default=6, help="Pre-judge diversity cap; accepted examples are capped at 2 later.")
    parser.add_argument("--minhash-threshold", type=float, default=0.84)
    parser.add_argument(
        "--families", nargs="+", choices=("technical", "clean", "story"),
        default=("technical", "clean"),
        help="Families to mine. Defaults preserve the original technical/clean calibration.",
    )
    args = parser.parse_args()
    started = time.time()
    judgments = [json.loads(line) for line in args.judgments.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {str(row["song_key"]): row for row in judgments}
    lyrics_by_key = {
        str(row["song_key"]): str(row["lyrics"])
        for row in pq.read_table(args.songs, columns=["song_key", "lyrics"]).to_pylist()
        if str(row.get("song_key") or "") in by_key
    }
    all_raw: list[dict[str, Any]] = []
    selected_all: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    family_summary: dict[str, Any] = {}
    globally_selected_hashes: set[str] = set()
    for family in args.families:
        parents = [
            row for row in judgments
            if (family in {"clean", "story"} or row.get("judge_family") == family) and parent_eligible(row, family)
        ]
        candidates = [window for parent in parents for window in mine_windows(parent, lyrics_by_key.get(str(parent["song_key"]), ""), family, args.stride)]
        candidates.sort(key=lambda item: (-item["local_score"], item["section_id"]))
        raw = candidates[: args.raw_per_family]
        selected, dedupe = select_deduped(raw, args.selected_per_family, args.per_song_cap, args.minhash_threshold)
        # Do not duplicate the same lyric window across families.
        cross_unique = []
        for row in selected:
            if row["text_hash"] in globally_selected_hashes:
                counts["cross_family_exact_duplicate"] += 1
                continue
            globally_selected_hashes.add(row["text_hash"])
            cross_unique.append(row)
        all_raw.extend(raw)
        selected_all.extend(cross_unique)
        family_summary[family] = {
            "eligible_parents": len(parents), "generated_windows": len(candidates), "raw_retained": len(raw),
            "selected": len(cross_unique), "dedupe_counts": dict(dedupe),
            "unique_source_songs": len({row["song_key"] for row in cross_unique}),
            "parent_judge_families": dict(Counter(row.get("judge_family", "missing") for row in parents)),
            "local_safety_rejections_are_pre_generation": family == "clean",
        }
    write_jsonl(args.raw_output, all_raw)
    write_jsonl(args.selected_output, selected_all)
    ended = time.time()
    summary = {
        "schema_version": 1,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        "wall_time_seconds": round(ended - started, 3), "command": " ".join([sys.executable, *sys.argv]),
        "inputs": {"judgments": str(args.judgments), "songs": str(args.songs)},
        "outputs": {"raw": str(args.raw_output), "selected": str(args.selected_output)},
        "config": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "families": family_summary, "selected_total": len(selected_all), "global_counts": dict(counts),
        "selected_unique_hashes": len({row["text_hash"] for row in selected_all}),
        "post_judge_accepted_per_song_cap": 2,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
