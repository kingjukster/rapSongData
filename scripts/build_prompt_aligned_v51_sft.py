#!/usr/bin/env python3
"""Build v5.1 by aligning generic source-section prompts to target keywords."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_section_mined_v4_sft import assistant_text, read_jsonl, sha256_file, text_sha256, write_jsonl
except ImportError:
    from build_section_mined_v4_sft import assistant_text, read_jsonl, sha256_file, text_sha256, write_jsonl


WORD_RE = re.compile(r"[a-z][a-z'-]{2,}", re.I)
USER_RE = re.compile(r"(<\|im_start\|>user\n).*?(<\|im_end\|>)", re.S)
STOPWORDS = {
    "about", "after", "again", "against", "ain't", "also", "always", "another", "around", "back",
    "because", "been", "before", "being", "between", "both", "could", "didn't", "doesn't", "doing",
    "don't", "down", "each", "even", "every", "from", "getting", "give", "going", "gone", "gotta",
    "have", "having", "here", "into", "isn't", "it's", "just", "keep", "know", "like", "little",
    "make", "maybe", "more", "much", "never", "nothing", "only", "other", "over", "really", "right",
    "said", "same", "should", "some", "still", "take", "than", "that", "that's", "their", "them",
    "then", "there", "there's", "these", "they", "they're", "thing", "think", "this", "those", "through",
    "time", "under", "until", "very", "want", "wasn't", "we're", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "won't", "would", "yeah", "your", "you're",
    "dream", "dreams", "grind", "heart", "hustle", "life", "mind", "pain", "rise", "shine", "strong",
    "world", "feel", "feeling", "real", "tell", "told", "look", "looking", "come", "coming",
}
PROMPT_TEMPLATES = {
    "story": (
        "Write exactly 12 original story-driven rap lines centered on {descriptor}. "
        "Make every line advance the same concrete scene, keep the phrasing natural, and resolve the ending. Lyrics only."
    ),
    "technical": (
        "Write exactly 12 original technical rap lines centered on {descriptor}. "
        "Use controlled internal and multisyllabic rhyme without losing semantic continuity or ending strength. Lyrics only."
    ),
    "clean": (
        "Write exactly 12 clean, radio-safe original rap lines centered on {descriptor}. "
        "Keep the language specific, naturally rhythmic, and expressive, with competent rhyme and a complete payoff. Lyrics only."
    ),
}


def topic_keywords(text: str, document_frequency: Counter[str], document_count: int, limit: int = 5) -> list[str]:
    tokens = [token.lower().strip("'-") for token in WORD_RE.findall(text)]
    counts = Counter(token for token in tokens if len(token) >= 4 and token not in STOPWORDS)
    first_line = set(WORD_RE.findall(text.splitlines()[0].lower())) if text.splitlines() else set()
    scored = []
    for token, count in counts.items():
        idf = math.log((document_count + 1) / (document_frequency[token] + 1)) + 1
        position_bonus = 0.35 if token in first_line else 0
        scored.append((count * idf + position_bonus, token))
    return [token for _, token in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def aligned_prompt(family: str, keywords: list[str]) -> str:
    descriptor = ", ".join(keywords)
    return PROMPT_TEMPLATES[family].format(descriptor=descriptor)


def note_descriptor(note: str, max_words: int = 30) -> str:
    normalized = re.sub(r"\s+", " ", str(note or "")).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0].strip().rstrip(".!?")
    words = first_sentence.split()
    return " ".join(words[:max_words]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/qwen3_4b_12line_section_mined_v51_prompt_aligned"))
    parser.add_argument("--technical-clean-reviews", type=Path, default=Path("data/reviews/section_quality_judge_gpt55_combined_accepted_capped.jsonl"))
    parser.add_argument("--story-reviews", type=Path, default=Path("data/reviews/story_section_quality_judge_gpt55_v5_combined_accepted_capped.jsonl"))
    args = parser.parse_args()
    started = time.time()
    original = {split: read_jsonl(args.input_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    source_rows = [row for rows in original.values() for row in rows if row["metadata"]["prompt_family"] in PROMPT_TEMPLATES]
    documents = [set(token.lower().strip("'-") for token in WORD_RE.findall(assistant_text(row["training_text"]))) for row in source_rows]
    document_frequency: Counter[str] = Counter(token for document in documents for token in document if token not in STOPWORDS)
    review_rows = read_jsonl(args.technical_clean_reviews) + read_jsonl(args.story_reviews)
    notes_by_section = {str(row["section_id"]): str(row.get("notes") or "") for row in review_rows}

    outputs: dict[str, list[dict[str, Any]]] = {}
    keyword_counts: Counter[str] = Counter()
    changed_by_family: Counter[str] = Counter()
    for split, rows in original.items():
        converted = []
        for row in rows:
            row = json.loads(json.dumps(row))
            family = row["metadata"]["prompt_family"]
            if family in PROMPT_TEMPLATES:
                target = assistant_text(row["training_text"])
                section_id = str(row["metadata"].get("section_id") or "")
                descriptor = note_descriptor(notes_by_section.get(section_id, ""))
                keywords = topic_keywords(target, document_frequency, len(documents))
                if len(descriptor.split()) < 4:
                    descriptor = ", ".join(keywords)
                if len(descriptor.split()) < 3:
                    raise RuntimeError(f"Insufficient topic descriptor for {row['id']}: {descriptor!r}")
                prompt = PROMPT_TEMPLATES[family].format(descriptor=descriptor)
                row["training_text"], count = USER_RE.subn(r"\1" + prompt + r"\2", row["training_text"], count=1)
                if count != 1:
                    raise RuntimeError(f"Could not replace user prompt for {row['id']}")
                row["id"] = row["id"].replace("section-v5-", "section-v51-")
                row["metadata"]["source"] = "qwen3_4b_12line_section_mined_v51_prompt_aligned"
                row["metadata"]["prompt_key"] = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                row["metadata"]["derived_topic_keywords"] = keywords
                row["metadata"]["derived_topic_summary"] = descriptor
                row["metadata"]["prompt_alignment_policy"] = "existing_independent_judge_summary_v1"
                changed_by_family[family] += 1
                keyword_counts.update(keywords)
            else:
                row["metadata"]["v51_role"] = "unchanged_melodic_control"
            converted.append(row)
        outputs[split] = converted

    for split in outputs:
        before = [text_sha256(assistant_text(row["training_text"])) for row in original[split]]
        after = [text_sha256(assistant_text(row["training_text"])) for row in outputs[split]]
        if before != after:
            raise RuntimeError(f"Assistant targets changed in {split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in outputs.items():
        paths[split] = args.output_dir / f"{split}.jsonl"
        write_jsonl(paths[split], rows)
    preference_source = args.input_dir / "preference_pairs.jsonl"
    paths["preferences"] = args.output_dir / "preference_pairs.jsonl"
    shutil.copyfile(preference_source, paths["preferences"])
    manifest = {
        "name": "qwen3_4b_12line_section_mined_v51_prompt_aligned", "status": "training_ready",
        "dataset_version": "v5.1", "major_variable": "prompt_target_topic_alignment",
        "unchanged": ["assistant_targets", "row_order", "splits", "family_balance", "seed", "source_examples"],
        "prompt_alignment_policy": "existing_independent_judge_summary_v1",
        "source_manifest": {"path": str(args.input_dir / "manifest.json"), "sha256": sha256_file(args.input_dir / "manifest.json")},
        "command": " ".join([sys.executable, *sys.argv]), "wall_time_seconds": round(time.time() - started, 3),
        "counts": {"rows": sum(map(len, outputs.values())), "prompt_aligned": sum(changed_by_family.values()), "unchanged_melodic": 33},
        "aligned_by_family": dict(sorted(changed_by_family.items())), "most_common_keywords": keyword_counts.most_common(20),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest["output_sha256"] = {key: sha256_file(path) for key, path in paths.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
