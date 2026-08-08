"""Rank generated lyric candidates from a generation JSONL sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SLUR_RE = re.compile(r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b", re.I)
QUESTION_LINE_RE = re.compile(r"\?")
ABSURD_DRIFT_RE = re.compile(
    r"\b(?:unicorn|elephant|horse with saddle|ubers|shower|your momma|your sister|what kind of girl)\b",
    re.I,
)
DIALOGUE_DRIFT_RE = re.compile(
    r"\b(?:you know what\?|listen here|let me tell ya|but we'll talk|what\?|right\?)\b",
    re.I,
)
SEVERE_VIOLENCE_RE = re.compile(
    r"\b(?:kill anybody|burn your house|brains? smacked|gun out|whole room would be dead|get your ass killed|"
    r"want me dead|throwing bodies|dead body|gonna kill|going to kill|shoot(?:ing)?|stab(?:bing)?)\b",
    re.I,
)
ARTIST_LEAK_RE = re.compile(
    r"\b(?:diddy|big daddy kane|dr\.?\s*seuss|god bless america|new york times|forever 21)\b",
    re.I,
)
FRAGMENT_END_RE = re.compile(r"\b(?:don|didn|ain|gon|wanna|gonna|gotta|tryna|coulda|woulda|shoulda|th)\s*$", re.I)
BAD_END_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "but",
    "for",
    "from",
    "give",
    "if",
    "in",
    "like",
    "of",
    "or",
    "so",
    "that",
    "the",
    "then",
    "to",
    "when",
    "while",
    "with",
    "without",
    "your",
    "be",
    "been",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "tell",
    "want",
    "was",
    "were",
    "will",
    "would",
}
BAD_END_PREFIXES = {
    "at least until",
    "cause once",
    "don't get",
    "we'll see",
    "so when",
    "then come",
    "cause if",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--top-per-prompt", type=int, default=5)
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--max-line-words", type=int, default=34)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def text_for(record: dict[str, object]) -> str:
    return str(record.get("generated_text") or record.get("text") or record.get("raw_text") or "")


def prompt_requests_clean(prompt: str) -> bool:
    prompt_l = prompt.lower()
    return any(token in prompt_l for token in ["no slur", "no-slur", "clean", "radio-safe", "radio safe"])


def repeated_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines if line.strip()]
    if not normalized:
        return 0.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(normalized)


def analysis_for(record: dict[str, object], text: str) -> dict[str, object]:
    if isinstance(record.get("analysis"), dict):
        return dict(record["analysis"])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prompt = str(record.get("prompt") or "")
    requested = None
    match = re.search(r"\b(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, flags=re.I)
    if match:
        requested = int(match.group(1))
    slur_terms = [match.group(0).lower() for match in SLUR_RE.finditer(text)]
    return {
        "line_count": len(lines),
        "word_count": len(words(text)),
        "repeated_line_ratio": round(repeated_line_ratio(lines), 4),
        "non_ascii_count": sum(1 for char in text if ord(char) > 127),
        "requested_line_count": requested,
        "line_count_delta": (len(lines) - requested) if requested is not None else None,
        "exact_line_match": requested == len(lines) if requested is not None else None,
        "is_hook_prompt": "hook" in prompt.lower(),
        "hook_line_cap_ok": (len(lines) <= 8) if "hook" in prompt.lower() else None,
        "no_slurs_requested": prompt_requests_clean(prompt),
        "slur_count": len(slur_terms),
        "slur_terms": sorted(set(slur_terms)),
        "no_slurs_passed": (len(slur_terms) == 0) if prompt_requests_clean(prompt) else None,
    }


def candidate_score(record: dict[str, object], *, min_lines: int, max_line_words: int) -> tuple[float, list[str]]:
    text = text_for(record)
    prompt = str(record.get("prompt") or "")
    prompt_l = prompt.lower()
    text_l = text.lower()
    analysis = analysis_for(record, text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_words = [len(words(line)) for line in lines]
    total_words = len(words(text))
    score = 50.0
    flags: list[str] = []

    if len(lines) < min_lines:
        score -= 18
        flags.append("too_short")
    if len(lines) >= 8:
        score += 8
    if 12 <= len(lines) <= 20:
        score += 8

    requested = analysis.get("requested_line_count")
    if isinstance(requested, int):
        delta = abs(len(lines) - requested)
        score += max(0, 16 - 3 * delta)
        if delta == 0:
            flags.append("exact_line_count")
        elif delta <= 2:
            flags.append("near_line_count")
        else:
            flags.append("line_count_miss")

    is_hook = bool(analysis.get("is_hook_prompt"))
    if is_hook:
        if len(lines) <= 8:
            score += 10
        else:
            score -= 14
            flags.append("hook_too_long")

    if any(count > max_line_words for count in line_words):
        score -= 14
        flags.append("long_line")
    elif line_words and max(line_words) <= 24:
        score += 6

    if total_words < 35:
        score -= 12
        flags.append("low_word_count")
    elif 70 <= total_words <= 220:
        score += 6

    repeated = float(analysis.get("repeated_line_ratio") or 0)
    score -= repeated * 35
    if repeated > 0:
        flags.append("repeated_lines")

    if int(analysis.get("non_ascii_count") or 0) == 0:
        score += 4
    else:
        score -= 10
        flags.append("non_ascii")

    if analysis.get("no_slurs_requested"):
        if analysis.get("no_slurs_passed"):
            score += 12
            flags.append("no_slur_pass")
        else:
            score -= 28
            flags.append("no_slur_fail")

    quote_count = text.count('"')
    if quote_count > 2:
        score -= 8
        flags.append("dialogue_like")

    question_lines = sum(1 for line in lines if QUESTION_LINE_RE.search(line))
    if question_lines >= 2:
        score -= 8
        flags.append("question_drift")

    if DIALOGUE_DRIFT_RE.search(text):
        score -= 10
        flags.append("dialogue_drift")

    if ABSURD_DRIFT_RE.search(text):
        score -= 14
        flags.append("absurd_drift")

    if SEVERE_VIOLENCE_RE.search(text):
        if "battle" in prompt_l:
            score -= 8
        else:
            score -= 22
        flags.append("violent_derailment")

    if ARTIST_LEAK_RE.search(text):
        score -= 8
        flags.append("name_or_brand_leak")

    if lines:
        last_line = lines[-1].strip()
        last_line_l = last_line.lower()
        last_words = words(lines[-1])
        if last_words and last_words[-1] in BAD_END_WORDS:
            score -= 16
            flags.append("weak_ending")
        if lines[-1].endswith(("...", ",", "-", "and", "but")):
            score -= 10
            flags.append("unfinished_punctuation")
        if any(last_line_l.startswith(prefix) for prefix in BAD_END_PREFIXES):
            score -= 18
            flags.append("unfinished_thought")
        if FRAGMENT_END_RE.search(last_line):
            score -= 18
            flags.append("unfinished_fragment")
        if last_line.count("(") != last_line.count(")") or last_line.count('"') % 2:
            score -= 10
            flags.append("unclosed_phrase")
        if len(last_words) <= 5 and not last_line.endswith((".", "!", "?", ")")):
            score -= 12
            flags.append("abrupt_short_ending")
        if last_line.endswith("?") and not str(record.get("prompt") or "").lower().startswith("write a hook"):
            score -= 5
            flags.append("question_ending")

    for keyword in ["rain", "train", "pressure", "lonely", "failure", "fame", "money"]:
        if keyword in prompt_l and keyword in text_l:
            score += 3

    return round(score, 2), flags


def judge_candidate(record: dict[str, object], *, min_lines: int, max_line_words: int) -> dict[str, object]:
    text = text_for(record)
    prompt = str(record.get("prompt") or "")
    analysis = analysis_for(record, text)
    score, raw_tags = candidate_score(record, min_lines=min_lines, max_line_words=max_line_words)
    positive_tags = {"exact_line_count", "near_line_count", "no_slur_pass"}
    failure_tags = [tag for tag in raw_tags if tag not in positive_tags]
    slur_present = int(analysis.get("slur_count") or 0) > 0
    hard_reject_slur = bool(prompt_requests_clean(prompt) and slur_present)
    strength_tags: list[str] = []

    line_count = int(analysis.get("line_count") or 0)
    if analysis.get("exact_line_match") is True:
        strength_tags.append("exact_line_count")
    elif analysis.get("requested_line_count") is not None and abs(int(analysis.get("line_count_delta") or 0)) <= 2:
        strength_tags.append("near_line_count")
    if analysis.get("hook_line_cap_ok") is True:
        strength_tags.append("hook_cap_pass")
    if 12 <= line_count <= 20 and not analysis.get("is_hook_prompt"):
        strength_tags.append("complete_verse_shape")
    if 4 <= line_count <= 8 and analysis.get("is_hook_prompt"):
        strength_tags.append("compact_hook_shape")
    if float(analysis.get("repeated_line_ratio") or 0.0) == 0.0:
        strength_tags.append("no_repeated_lines")
    if not slur_present:
        strength_tags.append("no_slurs_present")
    if any(keyword in text.lower() for keyword in ["train", "pressure", "lonely", "failure", "money", "fame"]):
        strength_tags.append("prompt_keyword_present")

    hard_fail_tags = {
        "no_slur_fail",
        "hook_too_long",
        "too_short",
        "long_line",
        "repeated_lines",
        "dialogue_drift",
        "absurd_drift",
        "violent_derailment",
        "name_or_brand_leak",
    }
    ending_fail_tags = {
        "weak_ending",
        "unfinished_punctuation",
        "unfinished_thought",
        "unfinished_fragment",
        "unclosed_phrase",
        "abrupt_short_ending",
    }

    if hard_reject_slur and "no_slur_fail" not in failure_tags:
        failure_tags.append("no_slur_fail")
    if hard_reject_slur or any(tag in hard_fail_tags for tag in failure_tags):
        decision_label = "reject"
    elif any(tag in ending_fail_tags for tag in failure_tags) or score < 78:
        decision_label = "fixable" if score >= 58 else "reject"
    elif score >= 82:
        decision_label = "keeper"
    else:
        decision_label = "fixable"

    return {
        "score": score,
        "decision_label": decision_label,
        "failure_tags": sorted(set(failure_tags)),
        "strength_tags": sorted(set(strength_tags)),
        "slur_present": slur_present,
        "hard_reject_slur": hard_reject_slur,
        "analysis": analysis,
    }


def main() -> None:
    args = parse_args()
    records = []
    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        judgment = judge_candidate(
            record,
            min_lines=args.min_lines,
            max_line_words=args.max_line_words,
        )
        record["candidate_score"] = judgment["score"]
        record["candidate_flags"] = judgment["failure_tags"]
        record["score"] = judgment["score"]
        record["decision_label"] = judgment["decision_label"]
        record["failure_tags"] = judgment["failure_tags"]
        record["strength_tags"] = judgment["strength_tags"]
        record["slur_present"] = judgment["slur_present"]
        record["hard_reject_slur"] = judgment["hard_reject_slur"]
        record["analysis"] = judgment["analysis"]
        records.append(record)

    records.sort(key=lambda item: (item.get("prompt_index", 0), -float(item["candidate_score"])))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_prompt: dict[int, list[dict[str, object]]] = {}
    for record in records:
        by_prompt.setdefault(int(record.get("prompt_index") or 0), []).append(record)

    md = ["# Ranked Generation Candidates", ""]
    md.append(f"- Source: `{args.input_jsonl}`")
    md.append(f"- Candidates: `{len(records)}`")
    md.append(f"- Top per prompt: `{args.top_per_prompt}`")
    md.append("")
    for prompt_index in sorted(by_prompt):
        prompt_records = by_prompt[prompt_index]
        prompt = prompt_records[0].get("prompt", "")
        md.extend([f"## Prompt {prompt_index}", "", str(prompt), ""])
        for rank, record in enumerate(prompt_records[: args.top_per_prompt], start=1):
            analysis = record.get("analysis", {})
            md.extend(
                [
                    (
                        f"### #{rank} {record.get('decision_label')} score={record['candidate_score']} "
                        f"sample={record.get('sample_index')}"
                    ),
                    "",
                    f"- failure_tags: `{', '.join(record.get('failure_tags') or []) or 'none'}`",
                    f"- strength_tags: `{', '.join(record.get('strength_tags') or []) or 'none'}`",
                    f"- lines: `{analysis.get('line_count')}` words: `{analysis.get('word_count')}` slurs: `{analysis.get('slur_count')}`",
                    "",
                    "```text",
                    str(record.get("generated_text") or ""),
                    "```",
                    "",
                ]
            )
    args.output_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(records), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
