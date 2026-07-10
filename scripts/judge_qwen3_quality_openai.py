#!/usr/bin/env python3
"""Auto-judge ranked Qwen3 rap generations and produce review artifacts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ISSUES = {
    "weak_imagery",
    "scene_drift",
    "generic",
    "awkward",
    "low_rhyme",
    "weak_payoff",
    "other",
}
DIMENSIONS = [
    "theme_adherence",
    "imagery",
    "rhyme_cadence",
    "originality",
    "scene_coherence",
    "ending_payoff",
    "naturalness",
]
SYSTEM_PROMPT = (
    "You are a strict rap lyric quality judge. Score only the supplied candidate. "
    "Do not rewrite lyrics. Return compact JSON only."
)
CALIBRATION_VERSION = "manual_audit_v1"
CALIBRATION_ISSUE_PENALTIES = {
    "low_rhyme": 0.18,
    "weak_imagery": 0.10,
}
LOW_HEURISTIC_HIGH_JUDGE_PENALTY = 0.22
BORDERLINE_HIGH_JUDGE_PENALTY = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ranked", type=Path, required=True, help="Ranked JSONL from rank_qwen3_quality.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qwen3_4b_base_12line_v1_auto_quality_judge"))
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini")
    parser.add_argument("--top-count", type=int, default=200)
    parser.add_argument("--stratified-sample", type=int, default=60)
    parser.add_argument("--per-prompt-top-k", type=int, default=8)
    parser.add_argument("--top-review-count", type=int, default=100)
    parser.add_argument("--disagreement-review-count", type=int, default=30)
    parser.add_argument("--borderline-review-count", type=int, default=30)
    parser.add_argument("--winners-per-prompt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reports-only", action="store_true", help="Rebuild reports from existing judged/pairwise JSONL.")
    parser.add_argument("--pairwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-judge", action="store_true", help="Use deterministic local mock responses for tests.")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv as python_dotenv

        python_dotenv(path)
        return
    except Exception:
        pass
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("row_id") or row.get("candidate_index"))


def prompt_key(row: dict[str, Any]) -> str:
    return str(row.get("prompt_key") or row.get("prompt") or "")


def heuristic_score(row: dict[str, Any]) -> float:
    return float(row.get("quality_score") or 0.0)


def select_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    structural = [row for row in rows if row.get("structural_metrics", {}).get("structural_pass") is True]
    ranked = sorted(structural, key=heuristic_score, reverse=True)
    selected: dict[str, dict[str, Any]] = {}
    for row in ranked[: max(0, args.top_count)]:
        selected[candidate_id(row)] = row

    rng = random.Random(args.seed)
    if args.stratified_sample > 0 and ranked:
        bucket_count = min(5, len(ranked))
        per_bucket = max(1, args.stratified_sample // bucket_count)
        for bucket_index in range(bucket_count):
            start = len(ranked) * bucket_index // bucket_count
            end = len(ranked) * (bucket_index + 1) // bucket_count
            bucket = ranked[start:end]
            rng.shuffle(bucket)
            for row in bucket[:per_bucket]:
                selected[candidate_id(row)] = row

    if args.per_prompt_top_k > 0:
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ranked:
            by_prompt[prompt_key(row)].append(row)
        for items in by_prompt.values():
            for row in items[: args.per_prompt_top_k]:
                selected[candidate_id(row)] = row

    return sorted(selected.values(), key=heuristic_score, reverse=True)


def issue_from_tags(tags: list[str]) -> str:
    mapping = {
        "weak_imagery": "weak_imagery",
        "scene_drift": "scene_drift",
        "generic_motivation": "generic",
        "awkward_phrase": "awkward",
        "low_rhyme_density": "low_rhyme",
        "weak_payoff": "weak_payoff",
    }
    for tag in tags:
        if tag in mapping:
            return mapping[tag]
    return "other"


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def score(value: Any, default: int = 3) -> int:
        try:
            return max(1, min(5, int(round(float(value)))))
        except Exception:
            return default

    usable = payload.get("usable_as_is")
    if isinstance(usable, bool):
        usable_text = "yes" if usable else "no"
    else:
        usable_text = str(usable or "").strip().lower()
        usable_text = "yes" if usable_text in {"yes", "true", "1"} else "no"
    issue = str(payload.get("main_issue") or "other").strip().lower()
    if issue not in ISSUES:
        issue = "other"
    dimensions = payload.get("dimension_scores") if isinstance(payload.get("dimension_scores"), dict) else {}
    return {
        "overall_quality": score(payload.get("overall_quality")),
        "usable_as_is": usable_text,
        "main_issue": issue,
        "short_reason": str(payload.get("short_reason") or "").strip()[:280],
        "dimension_scores": {name: score(dimensions.get(name)) for name in DIMENSIONS},
    }


def judge_prompt(row: dict[str, Any]) -> str:
    tags = ", ".join(row.get("quality_tags") or []) or "none"
    return (
        "Judge this rap lyric candidate using the rubric below.\n\n"
        "Return JSON with exactly these keys:\n"
        "overall_quality: integer 1-5\n"
        "usable_as_is: yes or no\n"
        "main_issue: weak_imagery | scene_drift | generic | awkward | low_rhyme | weak_payoff | other\n"
        "short_reason: one sentence\n"
        "dimension_scores: object with integer 1-5 scores for theme_adherence, imagery, rhyme_cadence, "
        "originality, scene_coherence, ending_payoff, naturalness\n\n"
        "Rubric:\n"
        "5 = strong and usable as-is; 4 = good with minor edits; 3 = structurally valid but bland; "
        "2 = weak/fixable; 1 = reject.\n"
        "Do not reward generic motivational filler. Reward concrete imagery, rap-like cadence/rhyme, "
        "scene coherence, natural wording, and a strong final line.\n\n"
        f"Prompt:\n{row.get('prompt')}\n\n"
        f"Heuristic quality score: {row.get('quality_score')}\n"
        f"Heuristic tags: {tags}\n\n"
        f"Lyrics:\n{row.get('lyrics') or row.get('generated_text')}\n"
    )


def create_openai_client() -> Any:
    try:
        import openai
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("openai package is unavailable.") from exc
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)


def call_openai_json(client: Any, *, model: str, prompt: str, temperature: float, max_retries: int) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Judge returned non-object JSON.")
            usage = getattr(response, "usage", None)
            if usage is not None:
                payload["_usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return payload
        except Exception as exc:  # pragma: no cover - network/API behavior
            last_exc = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"OpenAI judge failed after retries: {last_exc}") from last_exc


def call_json_judge(client: Any, *, model: str, prompt: str, temperature: float, max_retries: int) -> dict[str, Any]:
    payload = call_openai_json(
        client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_retries=max_retries,
    )
    usage = payload.pop("_usage", None)
    out = normalize_judge_payload(payload)
    if usage is not None:
        out["usage"] = usage
    return out


def mock_judge(row: dict[str, Any]) -> dict[str, Any]:
    h = heuristic_score(row)
    overall = max(1, min(5, int(round(1 + 4 * h))))
    tags = row.get("quality_tags") or []
    issue = issue_from_tags(tags)
    usable = "yes" if overall >= 4 and issue not in {"awkward", "weak_payoff"} else "no"
    base = max(1, min(5, overall))
    return {
        "overall_quality": base,
        "usable_as_is": usable,
        "main_issue": issue,
        "short_reason": f"Mock judge based on heuristic score {h:.3f}.",
        "dimension_scores": {name: base for name in DIMENSIONS},
    }


def judge_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    judged_path = args.output_dir / "judged_candidates.jsonl"
    existing = {str(row.get("candidate_id")): row for row in read_jsonl(judged_path)} if args.resume else {}
    client = None if args.mock_judge else create_openai_client()
    judged: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cid = candidate_id(row)
        if cid in existing:
            judged.append(apply_manual_calibration(existing[cid]))
            continue
        if args.mock_judge:
            judge = mock_judge(row)
        else:
            judge = call_json_judge(
                client,
                model=args.model,
                prompt=judge_prompt(row),
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
        enriched = {
            **row,
            "candidate_id": cid,
            "judge": judge,
            "heuristic_5": round(1 + 4 * heuristic_score(row), 4),
        }
        enriched["combined_quality_score"] = round(
            0.45 * enriched["heuristic_5"] + 0.55 * float(judge["overall_quality"]),
            4,
        )
        enriched["confidence_bucket"] = confidence_bucket(enriched)
        apply_manual_calibration(enriched)
        append_jsonl(judged_path, enriched)
        judged.append(enriched)
        if index % 25 == 0:
            print(json.dumps({"judged": index, "remaining": len(rows) - index}))
    return judged


def confidence_bucket(row: dict[str, Any]) -> str:
    h = heuristic_score(row)
    j = int(row["judge"]["overall_quality"])
    usable = row["judge"]["usable_as_is"] == "yes"
    high_h = h >= 0.72
    low_h = h <= 0.56
    high_j = j >= 4 and usable
    low_j = j <= 2 or (j <= 3 and not usable)
    if high_h and high_j:
        return "auto_keep"
    if low_h and low_j:
        return "auto_reject"
    return "needs_review"


def manual_calibration_penalty(row: dict[str, Any]) -> tuple[float, list[str]]:
    bucket = row.get("confidence_bucket") or confidence_bucket(row)
    row["confidence_bucket"] = bucket
    if bucket == "auto_keep":
        return 0.0, []

    h = heuristic_score(row)
    judge = row["judge"]
    j = int(judge["overall_quality"])
    usable = judge["usable_as_is"] == "yes"
    issue = str(judge.get("main_issue") or "other")
    penalty = 0.0
    signals: list[str] = []

    if j >= 4 and usable and h <= 0.56:
        penalty += LOW_HEURISTIC_HIGH_JUDGE_PENALTY
        signals.append("low_heuristic_high_judge")
    elif j >= 4 and usable and h < 0.72:
        penalty += BORDERLINE_HIGH_JUDGE_PENALTY
        signals.append("borderline_heuristic_high_judge")

    issue_penalty = CALIBRATION_ISSUE_PENALTIES.get(issue, 0.0)
    if issue_penalty:
        penalty += issue_penalty
        signals.append(f"issue_{issue}")

    return round(min(penalty, 0.55), 4), signals


def apply_manual_calibration(row: dict[str, Any]) -> dict[str, Any]:
    penalty, signals = manual_calibration_penalty(row)
    combined = float(row.get("combined_quality_score") or 0.0)
    row["calibrated_review_score"] = round(combined - penalty, 4)
    row["manual_calibration"] = {
        "version": CALIBRATION_VERSION,
        "penalty": penalty,
        "signals": signals,
    }
    return row


def ranking_score(row: dict[str, Any]) -> float:
    if "calibrated_review_score" not in row:
        apply_manual_calibration(row)
    return float(row.get("calibrated_review_score") or row.get("combined_quality_score") or 0.0)


def disagreement_reason(row: dict[str, Any]) -> str | None:
    h = heuristic_score(row)
    j = int(row["judge"]["overall_quality"])
    usable = row["judge"]["usable_as_is"] == "yes"
    if h >= 0.72 and (j <= 3 or not usable):
        return "high_heuristic_low_judge"
    if h <= 0.56 and j >= 4 and usable:
        return "low_heuristic_high_judge"
    return None


def borderline_reason(row: dict[str, Any]) -> str | None:
    h = heuristic_score(row)
    j = int(row["judge"]["overall_quality"])
    if 0.56 < h < 0.72:
        return "borderline_heuristic"
    if j == 3:
        return "borderline_judge"
    return None


def pairwise_prompt(left: dict[str, Any], right: dict[str, Any]) -> str:
    return (
        "Given the same prompt, choose the better rap lyric candidate. "
        "Reward concrete imagery, rap-like cadence/rhyme, coherent scene, originality, natural wording, "
        "and a strong final line. Return JSON only with winner: A or B, and short_reason.\n\n"
        f"Prompt:\n{left.get('prompt')}\n\n"
        f"Candidate A:\n{left.get('lyrics')}\n\n"
        f"Candidate B:\n{right.get('lyrics')}\n"
    )


def normalize_pairwise(payload: dict[str, Any]) -> dict[str, str]:
    winner = str(payload.get("winner") or "").strip().upper()
    if winner not in {"A", "B"}:
        winner = "A"
    return {"winner": winner, "short_reason": str(payload.get("short_reason") or "").strip()[:280]}


def pairwise_judge(client: Any, args: argparse.Namespace, left: dict[str, Any], right: dict[str, Any]) -> dict[str, str]:
    if args.mock_judge:
        winner = "A" if ranking_score(left) >= ranking_score(right) else "B"
        return {"winner": winner, "short_reason": "Mock pairwise winner by calibrated score."}
    payload = call_openai_json(
        client,
        model=args.model,
        prompt=pairwise_prompt(left, right)
        + "\nReturn JSON with keys winner and short_reason. winner must be A or B.",
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    return normalize_pairwise(payload)


def run_pairwise_tournaments(judged: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not args.pairwise:
        return [], []
    client = None if args.mock_judge else create_openai_client()
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged:
        by_prompt[prompt_key(row)].append(row)
    comparisons: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for _, items in sorted(by_prompt.items()):
        items = sorted(items, key=lambda row: (ranking_score(row), heuristic_score(row)), reverse=True)
        contenders = items[: max(1, args.per_prompt_top_k)]
        if not contenders:
            continue
        champion = contenders[0]
        for challenger in contenders[1:]:
            result = pairwise_judge(client, args, champion, challenger)
            comparison = {
                "prompt_key": prompt_key(champion),
                "prompt": champion.get("prompt"),
                "left_candidate_id": champion["candidate_id"],
                "right_candidate_id": challenger["candidate_id"],
                "winner": result["winner"],
                "short_reason": result["short_reason"],
            }
            comparisons.append(comparison)
            champion = champion if result["winner"] == "A" else challenger
            if args.sleep_seconds and not args.mock_judge:
                time.sleep(args.sleep_seconds)
        prompt_winners = [champion]
        for row in items:
            if row["candidate_id"] != champion["candidate_id"]:
                prompt_winners.append(row)
            if len(prompt_winners) >= args.winners_per_prompt:
                break
        for rank, row in enumerate(prompt_winners, start=1):
            winners.append({**row, "prompt_winner_rank": rank, "pairwise_champion": row["candidate_id"] == champion["candidate_id"]})
    return comparisons, winners


def markdown_record(row: dict[str, Any], rank: int | None = None) -> list[str]:
    judge = row["judge"]
    title = f"## {rank}. {row['candidate_id']}" if rank is not None else f"## {row['candidate_id']}"
    tags = ", ".join(row.get("quality_tags") or []) or "none"
    apply_manual_calibration(row)
    calibration = row["manual_calibration"]
    signals = ", ".join(calibration["signals"]) or "none"
    return [
        title,
        "",
        f"- bucket: `{row.get('confidence_bucket')}`",
        f"- heuristic: `{row.get('quality_score')}` | judge: `{judge['overall_quality']}` | combined: `{row.get('combined_quality_score')}`",
        f"- calibrated_review: `{row.get('calibrated_review_score')}` | calibration_penalty: `{calibration['penalty']}` | signals: `{signals}`",
        f"- usable_as_is: `{judge['usable_as_is']}` | main_issue: `{judge['main_issue']}`",
        f"- heuristic_tags: `{tags}`",
        f"- reason: {judge['short_reason']}",
        f"- prompt: {row.get('prompt')}",
        "",
        "```text",
        str(row.get("lyrics") or "").strip(),
        "```",
        "",
    ]


def write_top_judged(path: Path, rows: list[dict[str, Any]], count: int) -> None:
    selected = sorted(rows, key=ranking_score, reverse=True)[:count]
    lines = [
        "# Top Auto-Judged Qwen3-4B Candidates",
        "",
        f"- candidates: {len(selected)}",
        "- scoring: combined = 45% heuristic_5 + 55% model judge overall_quality",
        f"- calibrated_review_score: {CALIBRATION_VERSION} penalties apply only outside auto_keep",
        "",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.extend(markdown_record(row, rank))
    path.write_text("\n".join(lines), encoding="utf-8")


def review_priority(row: dict[str, Any]) -> float:
    return abs(float(row["heuristic_5"]) - float(row["judge"]["overall_quality"]))


def write_disagreements(path: Path, rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    disagreements = []
    for row in rows:
        reason = disagreement_reason(row)
        if reason:
            disagreements.append({**row, "disagreement_reason": reason})
    disagreements.sort(
        key=lambda row: (
            row["disagreement_reason"] != "high_heuristic_low_judge",
            review_priority(row),
        ),
        reverse=True,
    )
    selected = disagreements[:limit]
    lines = [
        "# Auto-Judge Disagreement Queue",
        "",
        f"- candidates: {len(selected)}",
        f"- total_true_disagreements: {len(disagreements)}",
        "- prioritized for small human audit before changing decoding, prompts, or training data",
        "",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.append(f"<!-- disagreement_reason: {row['disagreement_reason']} -->")
        lines.extend(markdown_record(row, rank))
    path.write_text("\n".join(lines), encoding="utf-8")
    return disagreements


def write_borderline_review(path: Path, rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    borderline = []
    for row in rows:
        reason = borderline_reason(row)
        if reason:
            distance = min(abs(float(row["quality_score"]) - 0.56), abs(float(row["quality_score"]) - 0.72))
            borderline.append({**row, "borderline_reason": reason, "borderline_distance": distance})
    borderline.sort(key=lambda row: (row["judge"]["overall_quality"] == 3, -row["borderline_distance"]), reverse=True)
    selected = borderline[:limit]
    lines = [
        "# Auto-Judge Borderline Review",
        "",
        f"- candidates: {len(selected)}",
        f"- total_borderline_cases: {len(borderline)}",
        "- optional second-pass audit after true disagreements",
        "",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.append(f"<!-- borderline_reason: {row['borderline_reason']} -->")
        lines.extend(markdown_record(row, rank))
    path.write_text("\n".join(lines), encoding="utf-8")
    return borderline


def write_prompt_winners(path: Path, winners: list[dict[str, Any]]) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in winners:
        by_prompt[prompt_key(row)].append(row)
    lines = ["# Per-Prompt Winners", ""]
    for _, items in sorted(by_prompt.items()):
        items = sorted(items, key=lambda row: int(row.get("prompt_winner_rank") or 999))
        if not items:
            continue
        items = [apply_manual_calibration(row) for row in items]
        lines.extend([f"## {items[0].get('prompt')}", ""])
        for row in items:
            judge = row["judge"]
            champ = "yes" if row.get("pairwise_champion") else "no"
            calibration = row["manual_calibration"]
            lines.extend(
                [
                    f"### Winner {row.get('prompt_winner_rank')}: {row['candidate_id']}",
                    "",
                    f"- pairwise_champion: `{champ}`",
                    f"- heuristic: `{row.get('quality_score')}` | judge: `{judge['overall_quality']}` | combined: `{row.get('combined_quality_score')}`",
                    f"- calibrated_review: `{row.get('calibrated_review_score')}` | calibration_penalty: `{calibration['penalty']}`",
                    f"- usable_as_is: `{judge['usable_as_is']}` | main_issue: `{judge['main_issue']}`",
                    "",
                    "```text",
                    str(row.get("lyrics") or "").strip(),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    *,
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    judged: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    borderline: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    winners: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets = Counter(row["confidence_bucket"] for row in judged)
    issues = Counter(row["judge"]["main_issue"] for row in judged)
    judge_scores = Counter(str(row["judge"]["overall_quality"]) for row in judged)
    calibration_penalties = [float(row.get("manual_calibration", {}).get("penalty") or 0.0) for row in judged]
    calibration_signals = Counter(
        signal
        for row in judged
        for signal in row.get("manual_calibration", {}).get("signals", [])
    )
    top50 = sorted(judged, key=ranking_score, reverse=True)[:50]
    return {
        "ranker": "qwen3_4b_base_12line_v1_auto_quality_judge_v1",
        "input_ranked": str(args.input_ranked),
        "output_dir": str(args.output_dir),
        "model": args.model if not args.mock_judge else "mock-judge",
        "mock_judge": bool(args.mock_judge),
        "selected_candidates": len(selected),
        "judged_candidates": len(judged),
        "confidence_buckets": dict(buckets),
        "judge_score_counts": dict(judge_scores),
        "main_issue_counts": dict(issues),
        "true_disagreement_count": len(disagreements),
        "borderline_count": len(borderline),
        "manual_review_queue_count": min(args.disagreement_review_count, len(disagreements)),
        "borderline_review_queue_count": min(args.borderline_review_count, len(borderline)),
        "pairwise_comparison_count": len(comparisons),
        "per_prompt_winner_count": len(winners),
        "calibration_version": CALIBRATION_VERSION,
        "calibrated_rows": sum(1 for penalty in calibration_penalties if penalty > 0),
        "avg_calibration_penalty": round(
            sum(calibration_penalties) / max(1, len(calibration_penalties)),
            4,
        ),
        "calibration_signal_counts": dict(calibration_signals),
        "top50_judge_usable_rate": round(
            sum(1 for row in top50 if row["judge"]["usable_as_is"] == "yes") / max(1, len(top50)),
            4,
        ),
        "top50_avg_combined_score": round(
            sum(float(row["combined_quality_score"]) for row in top50) / max(1, len(top50)),
            4,
        ),
        "top50_avg_calibrated_review_score": round(
            sum(ranking_score(row) for row in top50) / max(1, len(top50)),
            4,
        ),
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    if args.top_count <= 0:
        raise ValueError("--top-count must be > 0")
    if not args.mock_judge:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; live auto-judge cannot run.")
        if not args.model:
            raise SystemExit("Set --model or OPENAI_JUDGE_MODEL for live auto-judge.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_ranked)
    if args.reports_only:
        judged = [apply_manual_calibration(row) for row in read_jsonl(args.output_dir / "judged_candidates.jsonl")]
        selected = judged
    else:
        selected = select_candidates(rows, args)
        judged = judge_candidates(selected, args)
    judged.sort(key=ranking_score, reverse=True)

    if args.reports_only:
        comparisons = read_jsonl(args.output_dir / "pairwise_comparisons.jsonl")
        winners = [apply_manual_calibration(row) for row in read_jsonl(args.output_dir / "per_prompt_winners.jsonl")]
    else:
        comparisons, winners = run_pairwise_tournaments(judged, args)
    if args.reports_only:
        write_jsonl(args.output_dir / "judged_candidates.jsonl", judged)
    disagreements = write_disagreements(
        args.output_dir / "top_100_disagreements.md",
        judged,
        limit=args.disagreement_review_count,
    )
    borderline = write_borderline_review(
        args.output_dir / "top_100_borderline.md",
        judged,
        limit=args.borderline_review_count,
    )
    write_top_judged(args.output_dir / "top_100_auto_judged.md", judged, args.top_review_count)
    write_prompt_winners(args.output_dir / "per_prompt_winners.md", winners)
    if not args.reports_only:
        write_jsonl(args.output_dir / "pairwise_comparisons.jsonl", comparisons)
        write_jsonl(args.output_dir / "per_prompt_winners.jsonl", winners)
    summary = build_summary(
        args=args,
        selected=selected,
        judged=judged,
        disagreements=disagreements,
        borderline=borderline,
        comparisons=comparisons,
        winners=winners,
    )
    (args.output_dir / "quality_judge_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
