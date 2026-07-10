"""Curate a local Qwen3-4B generation sweep with heuristic labels.

This replaces remote/manual LLM judging for the rebuild path. It reads the raw
sweep JSONL, assigns keeper/fixable/reject labels, and writes split JSONL files
that can be passed directly into ``package_qwen3_sweep_datasets.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SLUR_RE = re.compile(
    r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b",
    re.IGNORECASE,
)
ARTIFACT_RE = re.compile(
    r"(lyrics taken from|lyrics from|you might also like|genius\.com|https?://|embed)",
    re.IGNORECASE,
)
INSTRUCTION_ECHO_RE = re.compile(
    r"\b(?:return only|generated rap lyrics|write exactly|write only|no intro|no commentary|"
    r"no bracket labels|avoid slurs|avoid hate speech|no profanity|radio-safe|"
    r"user(?:'s)? prompt|assistant|system prompt)\b",
    re.IGNORECASE,
)
QUESTION_DRIFT_RE = re.compile(
    r"\b(what do you think|should i|can you|do you want|would you like|is this enough)\b",
    re.IGNORECASE,
)
PROFANITY_RE = re.compile(
    r"\b(?:fuck(?:ing|in|ed)?|shit(?:ty)?|bitch(?:es)?|asshole|motherfucker|damn)\b",
    re.IGNORECASE,
)
DIALOGUE_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z0-9_ -]{1,24}:|[\"']).+")
VIOLENT_DERAILMENT_RE = re.compile(
    r"\b(?:kill(?:ed|ing)?\s+(?:you|him|her|them)|murder\s+(?:you|him|her|them)|"
    r"shoot\s+(?:you|him|her|them)|rape|sexual threat)\b",
    re.IGNORECASE,
)
DANGLING_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "cause",
    "could",
    "for",
    "from",
    "have",
    "here",
    "his",
    "hold",
    "how",
    "i",
    "if",
    "in",
    "inside",
    "is",
    "like",
    "make",
    "might",
    "my",
    "of",
    "on",
    "or",
    "our",
    "outside",
    "push",
    "same",
    "save",
    "serious",
    "should",
    "so",
    "sure",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "while",
    "who",
    "why",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
}
DANGLING_PHRASE_RE = re.compile(
    r"(?:\bmake sure|\bstand beside|\bso it|\bwhy we should|\bsomeone else|\bcome outta nowhere and save|\bwasn'?t)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sweeps/qwen3_4b_rebuild/sweep_raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/curation/qwen3_4b_rebuild"))
    parser.add_argument("--keeper-min-score", type=float, default=0.72)
    parser.add_argument("--fixable-min-score", type=float, default=0.42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def repeated_line_ratio(output_lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in output_lines]
    normalized = [line for line in normalized if line]
    if not normalized:
        return 0.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(normalized)


def requested_line_count(prompt: str) -> int | None:
    match = re.search(r"\b(?:exactly\s+)?(\d+)\s*[- ]?(?:bar|bars|line|lines)\b", prompt, re.I)
    return int(match.group(1)) if match else None


def is_hook_prompt(prompt: str) -> bool:
    return bool(re.search(r"\bhook\b", prompt, re.I))


def output_kind(prompt: str) -> str:
    return "hook" if is_hook_prompt(prompt) else "verse"


def clean_requested(prompt: str) -> bool:
    return bool(re.search(r"\b(no slurs?|radio-safe|clean|no profanity|family pride)\b", prompt, re.I))


def profanity_block_requested(prompt: str) -> bool:
    return bool(re.search(r"\b(radio-safe|clean|no profanity)\b", prompt, re.I))


def has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def has_incomplete_terminal_line(last_line: str) -> bool:
    stripped = last_line.strip()
    if not stripped:
        return True
    last_words = words(stripped)
    if stripped.count("(") > stripped.count(")") or stripped.count("[") > stripped.count("]"):
        return True
    if stripped.endswith((",", ":", ";", "-", "...")):
        return True
    if not re.search(r"[.!?]$", stripped):
        return True
    if last_words and last_words[-1] in DANGLING_TERMS:
        return True
    return bool(DANGLING_PHRASE_RE.search(stripped))


def score_and_tags(row: dict[str, Any]) -> tuple[float, list[str], list[str], dict[str, Any]]:
    prompt = str(row.get("prompt") or "")
    text = str(row.get("generated_text") or "")
    raw_text = str(row.get("raw_generated_text") or text)
    output_lines = lines(text)
    word_count = len(words(text))
    line_count = len(output_lines)
    target_lines = requested_line_count(prompt)
    hook = is_hook_prompt(prompt)
    repeated_ratio = repeated_line_ratio(output_lines)
    slur_terms = sorted(set(match.group(0).lower() for match in SLUR_RE.finditer(text + "\n" + raw_text)))
    no_slur_required = clean_requested(prompt)
    profanity_terms = sorted(set(match.group(0).lower() for match in PROFANITY_RE.finditer(text + "\n" + raw_text)))
    profanity_required = profanity_block_requested(prompt)
    postprocess_actions = row.get("postprocess_actions") if isinstance(row.get("postprocess_actions"), list) else []
    hit_token_cap = bool(row.get("hit_token_cap")) and not bool(row.get("hit_eos"))

    failure_tags: list[str] = []
    strength_tags: list[str] = []
    score = 1.0

    if hit_token_cap:
        failure_tags.append("hit_token_cap")
        score -= 0.35
    if not text.strip() or word_count == 0:
        failure_tags.append("empty_output")
        score -= 0.8
    if ARTIFACT_RE.search(raw_text) or ARTIFACT_RE.search(text):
        failure_tags.append("source_artifact")
        score -= 0.5
    if INSTRUCTION_ECHO_RE.search(text):
        failure_tags.extend(["instruction_echo", "prompt_leakage", "meta_commentary"])
        score -= 0.65
    if slur_terms:
        failure_tags.append("slur_present")
        score -= 0.15
    if no_slur_required and slur_terms:
        failure_tags.append("no_slur_fail")
        score -= 0.65
    if profanity_required and profanity_terms:
        failure_tags.extend(["profanity_fail", "clean_prompt_profanity_fail"])
        score -= 0.55
    if VIOLENT_DERAILMENT_RE.search(text):
        failure_tags.append("violent_derailment")
        score -= 0.45
    if has_non_ascii(text):
        failure_tags.append("non_ascii")
        score -= 0.08
    if repeated_ratio > 0.12:
        failure_tags.append("repeated_lines")
        score -= min(0.35, repeated_ratio)
    else:
        strength_tags.append("no_repeated_lines")

    if hook:
        if 2 <= line_count <= 8:
            strength_tags.append("hook_cap_pass")
            if line_count <= 6:
                strength_tags.append("compact_hook_shape")
        else:
            failure_tags.append("line_count_miss")
            score -= 0.22
        if word_count < 10:
            failure_tags.append("too_short")
            score -= 0.2
    else:
        if target_lines is not None:
            delta = abs(line_count - target_lines)
            if delta == 0:
                strength_tags.append("complete_verse_shape")
            elif delta == 1:
                failure_tags.extend(["line_count_miss", "exact_line_hard_fail"])
                score -= 0.22
            else:
                failure_tags.extend(["line_count_miss", "exact_line_hard_fail"])
                score -= min(0.45, 0.12 * delta)
        elif 10 <= line_count <= 20:
            strength_tags.append("complete_verse_shape")
        if line_count < 8 or word_count < 45:
            failure_tags.append("too_short")
            score -= 0.22

    question_count = text.count("?")
    last_line = output_lines[-1] if output_lines else ""
    if last_line.endswith("?"):
        failure_tags.append("question_ending")
        score -= 0.25
    if question_count >= 2 or QUESTION_DRIFT_RE.search(text):
        failure_tags.append("question_drift")
        score -= 0.35
    if output_lines:
        last_words = words(last_line)
        if not hook and len(last_words) <= 3 and not re.search(r"[.!]$", last_line):
            failure_tags.append("weak_ending")
            score -= 0.18
        if not hook and has_incomplete_terminal_line(last_line):
            failure_tags.extend(["incomplete_ending", "terminal_fragment"])
            score -= 0.4

    dialogue_like_lines = [line for line in output_lines if DIALOGUE_RE.search(line)]
    if output_lines and len(dialogue_like_lines) / len(output_lines) > 0.25:
        failure_tags.append("dialogue_like")
        score -= 0.22

    if postprocess_actions:
        strength_tags.append("postprocess_fixed")
        if "drop_dangling_final_line" in postprocess_actions:
            failure_tags.append("weak_ending")
            score -= 0.08

    score = round(max(0.0, min(1.0, score)), 4)
    metrics = {
        "line_count": line_count,
        "word_count": word_count,
        "requested_line_count": target_lines,
        "line_count_delta": (line_count - target_lines) if target_lines is not None else None,
        "is_hook_prompt": hook,
        "repeated_line_ratio": round(repeated_ratio, 4),
        "slur_count": len(slur_terms),
        "slur_terms": slur_terms,
        "no_slurs_requested": no_slur_required,
        "profanity_count": len(profanity_terms),
        "profanity_terms": profanity_terms,
        "profanity_block_requested": profanity_required,
        "output_kind": output_kind(prompt),
        "hit_eos": bool(row.get("hit_eos")),
        "hit_token_cap": hit_token_cap,
    }
    return score, sorted(set(failure_tags)), sorted(set(strength_tags)), metrics


def label_row(
    *,
    row: dict[str, Any],
    score: float,
    failure_tags: list[str],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    hard_reject_tags = {
        "empty_output",
        "instruction_echo",
        "no_slur_fail",
        "question_drift",
        "source_artifact",
        "violent_derailment",
    }
    if set(failure_tags) & hard_reject_tags:
        return "reject"
    if "line_count_miss" in failure_tags and metrics.get("requested_line_count") is not None:
        delta = abs(int(metrics.get("line_count_delta") or 0))
        return "fixable" if delta == 1 and score >= args.fixable_min_score else "reject"
    if "incomplete_ending" in failure_tags:
        return "fixable" if score >= args.fixable_min_score else "reject"
    if "hit_token_cap" in failure_tags:
        return "fixable" if score >= args.fixable_min_score else "reject"
    if "profanity_fail" in failure_tags:
        return "fixable" if score >= args.fixable_min_score else "reject"
    if score >= args.keeper_min_score and "too_short" not in failure_tags:
        return "keeper"
    if score >= args.fixable_min_score:
        return "fixable"
    if row.get("postprocess_applied") and score >= 0.35:
        return "fixable"
    return "reject"


def curate(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    curated: list[dict[str, Any]] = []
    for row in rows:
        score, failure_tags, strength_tags, metrics = score_and_tags(row)
        decision = label_row(row=row, score=score, failure_tags=failure_tags, metrics=metrics, args=args)
        hard_reject_slur = "no_slur_fail" in failure_tags
        slur_present = metrics["slur_count"] > 0
        curated.append(
            {
                **row,
                "decision_label": decision,
                "score": score,
                "failure_tags": failure_tags,
                "strength_tags": strength_tags,
                "slur_present": slur_present,
                "hard_reject_slur": hard_reject_slur,
                "analysis": metrics,
            }
        )
    return curated


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    decisions = Counter(row["decision_label"] for row in rows)
    failures = Counter(tag for row in rows for tag in row["failure_tags"])
    strengths = Counter(tag for row in rows for tag in row["strength_tags"])
    postprocess = Counter(action for row in rows for action in row.get("postprocess_actions", []))
    keeper_kinds = Counter(row.get("analysis", {}).get("output_kind", "unknown") for row in rows if row["decision_label"] == "keeper")
    lines_out = [
        "# Qwen3-4B Local Sweep Curation",
        "",
        "Heuristic local curation. No OpenAI or external judge calls were used.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- Keepers: {decisions.get('keeper', 0)}",
        f"- Fixable: {decisions.get('fixable', 0)}",
        f"- Rejects: {decisions.get('reject', 0)}",
        f"- Postprocessed: {sum(1 for row in rows if row.get('postprocess_applied'))}",
        f"- Slur present: {sum(1 for row in rows if row.get('slur_present'))}",
        f"- Hard no-slur rejects: {sum(1 for row in rows if row.get('hard_reject_slur'))}",
        f"- Hook keepers: {keeper_kinds.get('hook', 0)}",
        f"- Verse keepers: {keeper_kinds.get('verse', 0)}",
        "",
        "## Failure Tags",
        "",
    ]
    lines_out.extend(f"- {tag}: {count}" for tag, count in failures.most_common(30)) if failures else lines_out.append("- none")
    lines_out.extend(["", "## Strength Tags", ""])
    lines_out.extend(f"- {tag}: {count}" for tag, count in strengths.most_common(30)) if strengths else lines_out.append("- none")
    lines_out.extend(["", "## Postprocess Actions", ""])
    lines_out.extend(f"- {tag}: {count}" for tag, count in postprocess.most_common(30)) if postprocess else lines_out.append("- none")
    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    curated = curate(rows, args)
    keepers = [row for row in curated if row["decision_label"] == "keeper"]
    fixable = [row for row in curated if row["decision_label"] == "fixable"]
    rejects = [row for row in curated if row["decision_label"] == "reject"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "annotated_sweep.jsonl", curated)
    write_jsonl(args.output_dir / "keepers.jsonl", keepers)
    write_jsonl(args.output_dir / "fixable.jsonl", fixable)
    write_jsonl(args.output_dir / "rejects.jsonl", rejects)
    write_report(args.output_dir / "curation_report.md", curated)
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "rows": len(curated),
        "keepers": len(keepers),
        "fixable": len(fixable),
        "rejects": len(rejects),
        "hook_keepers": sum(
            1 for row in keepers if row.get("analysis", {}).get("output_kind") == "hook"
        ),
        "verse_keepers": sum(
            1 for row in keepers if row.get("analysis", {}).get("output_kind") == "verse"
        ),
        "local_only": True,
        "base_model_scope": "Qwen/Qwen3-4B",
    }
    (args.output_dir / "curation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
