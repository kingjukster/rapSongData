#!/usr/bin/env python3
"""DEPRECATED: resolve legacy human evaluation reviews to model labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-packet", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_reviews(packet: dict[str, Any], private_key: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = packet.get("comparisons")
    assignments = private_key.get("assignments")
    if not isinstance(comparisons, list) or not isinstance(assignments, list):
        raise ValueError("Packet and private key must contain comparison/assignment arrays")
    key_by_id = {str(row.get("comparison_id")): row for row in assignments if isinstance(row, dict)}
    if len(key_by_id) != len(assignments):
        raise ValueError("Private key contains missing or duplicate comparison ids")
    packet_ids = {str(row.get("comparison_id")) for row in comparisons if isinstance(row, dict)}
    if packet_ids != set(key_by_id):
        raise ValueError("Reviewed packet and private key comparison-id sets differ")

    resolved: list[dict[str, Any]] = []
    for row in comparisons:
        comparison_id = str(row.get("comparison_id") or "")
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        tie = review.get("tie") is True
        winner_alias = str(review.get("winner_alias") or "").strip()
        if tie and winner_alias:
            raise ValueError(f"{comparison_id}: review cannot select both tie and winner_alias")
        if not tie and not winner_alias:
            raise ValueError(f"{comparison_id}: review is incomplete")
        assignment = key_by_id[comparison_id]
        alias_to_label = {
            str(candidate.get("candidate_alias")): str(candidate.get("generation_label"))
            for candidate in assignment.get("candidates") or []
            if isinstance(candidate, dict)
        }
        if winner_alias and winner_alias not in alias_to_label:
            raise ValueError(f"{comparison_id}: winner alias is not present in the private key")
        ranking_aliases = review.get("ranking") if isinstance(review.get("ranking"), list) else []
        if any(str(alias) not in alias_to_label for alias in ranking_aliases):
            raise ValueError(f"{comparison_id}: ranking contains an unknown alias")
        resolved.append(
            {
                "comparison_id": comparison_id,
                "row_id": assignment.get("source_row_id"),
                "winner": "tie" if tie else alias_to_label[winner_alias],
                "ranking": [alias_to_label[str(alias)] for alias in ranking_aliases],
                "notes": str(review.get("notes") or ""),
            }
        )
    return resolved


def main() -> int:
    print(
        "DEPRECATED: active evaluation uses judge_quality_goal_eval_packet_openai.py.",
        file=sys.stderr,
    )
    args = parse_args()
    packet = read_object(args.reviewed_packet)
    private_key = read_object(args.private_key)
    resolved = resolve_reviews(packet, private_key)
    output = {
        "schema_version": 1,
        "reviewed_packet_sha256": sha256_file(args.reviewed_packet),
        "private_key_sha256": sha256_file(args.private_key),
        "review_count": len(resolved),
        "preferences": resolved,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("schema_version", "review_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
