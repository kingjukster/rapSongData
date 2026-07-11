#!/usr/bin/env python3
"""Blindly judge quality-goal comparison packets with an automated OpenAI consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a strict rap lyric comparison judge. Compare only the anonymous candidates. "
    "Do not infer model identity. Reward concrete imagery, coherent scene movement, natural rap cadence and rhyme, "
    "originality, and an earned final-line payoff. Return JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini")
    parser.add_argument("--votes-per-comparison", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--mock-judge", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    try:
        from dotenv import load_dotenv as dotenv_load

        dotenv_load(path)
    except Exception:
        return


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object at {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_prompt(comparison: dict[str, Any]) -> str:
    metadata = comparison.get("prompt_metadata") or {}
    candidates = comparison.get("candidates") or []
    rendered = "\n\n".join(
        f"Candidate {candidate['candidate_alias']}:\n{candidate['lyrics']}" for candidate in candidates
    )
    aliases = [str(candidate["candidate_alias"]) for candidate in candidates]
    return (
        "Compare these anonymous rap verses for the supplied instruction.\n\n"
        "Return JSON with exactly these fields:\n"
        f"winner_alias: one of {aliases}\n"
        f"ranking: every alias exactly once, best to worst\n"
        "tie: boolean; true only if the top candidates are genuinely indistinguishable\n"
        "reason: one concise sentence\n"
        "dimension_winners: object with imagery, rhyme_cadence, scene_coherence, originality, ending_payoff; "
        "each value must be one alias\n\n"
        f"Instruction:\n{metadata.get('prompt')}\n\n{rendered}"
    )


def create_client() -> Any:
    import openai

    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)


def call_json(client: Any, *, model: str, prompt: str, temperature: float, max_retries: int) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Judge response is not an object")
            usage = getattr(response, "usage", None)
            if usage is not None:
                payload["_usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return payload
        except Exception as exc:
            last = exc
            if attempt < max_retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Automated judge failed after retries: {last}") from last


def mock_payload(comparison: dict[str, Any], vote: int) -> dict[str, Any]:
    candidates = comparison.get("candidates") or []
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            len(set(str(candidate.get("lyrics") or "").lower().split())),
            len(str(candidate.get("lyrics") or "")),
            str(candidate.get("candidate_alias")),
        ),
        reverse=True,
    )
    aliases = [str(candidate["candidate_alias"]) for candidate in ordered]
    return {
        "winner_alias": aliases[0],
        "ranking": aliases,
        "tie": False,
        "reason": f"deterministic mock vote {vote}",
        "dimension_winners": {name: aliases[0] for name in ("imagery", "rhyme_cadence", "scene_coherence", "originality", "ending_payoff")},
    }


def normalize_vote(payload: dict[str, Any], aliases: list[str]) -> dict[str, Any]:
    winner = str(payload.get("winner_alias") or "").strip()
    ranking = [str(value).strip() for value in payload.get("ranking") or []]
    tie = payload.get("tie") is True
    if winner not in aliases:
        raise ValueError(f"Judge winner {winner!r} is not a candidate alias")
    if len(ranking) != len(aliases) or set(ranking) != set(aliases):
        raise ValueError("Judge ranking must contain every candidate alias exactly once")
    dimensions = payload.get("dimension_winners") if isinstance(payload.get("dimension_winners"), dict) else {}
    normalized_dimensions = {}
    for name in ("imagery", "rhyme_cadence", "scene_coherence", "originality", "ending_payoff"):
        value = str(dimensions.get(name) or winner)
        normalized_dimensions[name] = value if value in aliases else winner
    return {
        "winner_alias": winner,
        "ranking": ranking,
        "tie": tie,
        "reason": str(payload.get("reason") or "")[:500],
        "dimension_winners": normalized_dimensions,
        "usage": payload.get("_usage") or {},
    }


def consensus(votes: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any]:
    non_tie = [vote for vote in votes if not vote["tie"]]
    first_counts = Counter(vote["winner_alias"] for vote in non_tie)
    borda = Counter()
    for vote in votes:
        for index, alias in enumerate(vote["ranking"]):
            borda[alias] += len(aliases) - index - 1
    ordered = sorted(aliases, key=lambda alias: (first_counts[alias], borda[alias], alias), reverse=True)
    top = ordered[0]
    tied = len(ordered) > 1 and (first_counts[top], borda[top]) == (first_counts[ordered[1]], borda[ordered[1]])
    return {
        "winner_alias": top,
        "ranking": ordered,
        "tie": tied,
        "vote_count": len(votes),
        "first_place_counts": dict(first_counts),
        "borda_scores": dict(borda),
        "votes": votes,
    }


def judge_packet(
    packet: dict[str, Any],
    private_key: dict[str, Any],
    *,
    votes_per_comparison: int,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    mock: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if votes_per_comparison < 1 or votes_per_comparison % 2 == 0:
        raise ValueError("votes-per-comparison must be a positive odd integer")
    comparisons = packet.get("comparisons") or []
    assignments = {str(row.get("comparison_id")): row for row in private_key.get("assignments") or []}
    if {str(row.get("comparison_id")) for row in comparisons} != set(assignments):
        raise ValueError("Packet and private-key comparison ids differ")
    client = None if mock else create_client()
    judged: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    label_wins: Counter[str] = Counter()
    label_borda: Counter[str] = Counter()
    for comparison in comparisons:
        comparison_id = str(comparison["comparison_id"])
        aliases = [str(candidate["candidate_alias"]) for candidate in comparison.get("candidates") or []]
        votes = []
        prompt = comparison_prompt(comparison)
        for vote_index in range(votes_per_comparison):
            raw = mock_payload(comparison, vote_index) if mock else call_json(
                client,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_retries=max_retries,
            )
            votes.append(normalize_vote(raw, aliases))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        result = consensus(votes, aliases)
        assignment = assignments[comparison_id]
        alias_to_label = {
            str(row["candidate_alias"]): str(row["generation_label"])
            for row in assignment.get("candidates") or []
        }
        winner_label = "tie" if result["tie"] else alias_to_label[result["winner_alias"]]
        ranking_labels = [alias_to_label[alias] for alias in result["ranking"]]
        if winner_label != "tie":
            label_wins[winner_label] += 1
        for index, label in enumerate(ranking_labels):
            label_borda[label] += len(ranking_labels) - index - 1
        judged.append({"comparison_id": comparison_id, "consensus": result})
        preferences.append(
            {
                "comparison_id": comparison_id,
                "row_id": assignment.get("source_row_id"),
                "winner": winner_label,
                "ranking": ranking_labels,
                "judge_model": "mock" if mock else model,
                "votes_per_comparison": votes_per_comparison,
            }
        )
    adapter_labels = sorted(label for label in label_borda if label != "base")
    selected_adapter = (
        max(adapter_labels, key=lambda label: (label_wins[label], label_borda[label], label))
        if adapter_labels
        else None
    )
    summary = {
        "comparison_count": len(comparisons),
        "judge_model": "mock" if mock else model,
        "votes_per_comparison": votes_per_comparison,
        "label_first_place_counts": dict(label_wins),
        "label_borda_scores": dict(label_borda),
        "selected_adapter": selected_adapter,
        "selection_policy": "highest automated consensus first-place count, then Borda, then stable label",
        "human_review_used": False,
    }
    return judged, preferences, summary


def main() -> int:
    args = parse_args()
    load_dotenv(Path(".env"))
    packet = read_object(args.packet)
    private_key = read_object(args.private_key)
    judged, preferences, summary = judge_packet(
        packet,
        private_key,
        votes_per_comparison=args.votes_per_comparison,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        mock=args.mock_judge,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judged_comparisons.json").write_text(json.dumps(judged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "resolved_preferences.json").write_text(
        json.dumps({"preferences": preferences}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary.update(
        {
            "packet_sha256": sha256_file(args.packet),
            "private_key_sha256": sha256_file(args.private_key),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
