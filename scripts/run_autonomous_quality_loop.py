#!/usr/bin/env python3
"""Resumable autonomous quality loop for rapSongData.

This script implements a documented, stateful improvement cycle using the
existing repo scripts as components:
- prompt generation
- structural evaluation
- local heuristic ranking
- judge pass-through (mock or live when API key is available)
- smoke tests

All runs write artifacts to:
- runs/qwen3_rap_quality_autonomous_loop/<timestamp>_<slug>/
- docs/qwen3_rap_quality_autonomous_loop/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs" / "qwen3_rap_quality_autonomous_loop"
RUNS_ROOT = ROOT / "runs" / "qwen3_rap_quality_autonomous_loop"
STATE_PATH = RUNS_ROOT / "state.json"
LABELED_ARTIFACT_ROOT = ROOT / "codex-autonomous-loop-evidence"
LABELED_DOCS_ROOT = LABELED_ARTIFACT_ROOT / "docs"
LABELED_RUNS_ROOT = LABELED_ARTIFACT_ROOT / "runs"

BIBLE_MARKER_FILES = [
    ROOT / "codex-bible" / "rapSongData Documentation and Governance Milestone Plan.pdf",
    ROOT / "codex-bible" / "Comprehensive Codex Goal and Research Program for rapSongData.pdf",
]
BIBLE_TEXT_FILES = [
    ROOT / "codex-bible" / "rapSongData Documentation and Governance Milestone Plan.pdf.txt",
    ROOT / "codex-bible" / "Comprehensive Codex Goal and Research Program for rapSongData.pdf.txt",
]
GOAL_PATH = ROOT / ".codex" / "attachments" / "573fb76a-8b55-488f-9bef-b4daae93c1c6" / "pasted-text.txt"

DEFAULT_INPUT_SWEEP = (
    ROOT / "data" / "sweeps" / "qwen3_4b_base_no_adapter_12line_expanded_eval_line_enforced_retry2" / "sweep_raw.jsonl"
)
DEFAULT_PROMPTS = ROOT / "data" / "prompts" / "qwen3_4b_12line_expanded_eval_prompts.json"
DEFAULT_TRAIN_CORPUS = ROOT / "data" / "training" / "qwen3_4b_sweep_sft" / "train.jsonl"
DEFAULT_BASELINE_EVAL = (
    ROOT / "reports" / "qwen3_4b_base_no_adapter_12line_expanded_eval_line_enforced_retry2_metrics.json"
)
DEFAULT_BASELINE_JUDGE = (
    ROOT / "reports" / "qwen3_4b_base_12line_v1_auto_quality_judge" / "quality_judge_metrics.json"
)
DEFAULT_SWEEP_ROOT = ROOT / "data" / "sweeps"

DEFAULT_SWEEP_BATCH_SIZE = 2
POST_SWEEP_BATCH_SIZE = 1

POST_SWEEP_RANK_VARIANTS = [
    {"slug": "sensitivity-top80", "top_n": "80", "similarity_threshold": "0.9", "ngram_size": "6"},
    {"slug": "sensitivity-top120", "top_n": "120", "similarity_threshold": "0.85", "ngram_size": "5"},
    {"slug": "sensitivity-top160", "top_n": "160", "similarity_threshold": "0.92", "ngram_size": "7"},
]

POST_SWEEP_ROUND_VARIANTS = [
    POST_SWEEP_RANK_VARIANTS,
    [
        {
            "slug": "priority-top90-strict",
            "top_n": "90",
            "similarity_threshold": "0.8",
            "ngram_size": "6",
            "judge_top_review_count": "70",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
        {
            "slug": "priority-top210-broad",
            "top_n": "210",
            "similarity_threshold": "0.96",
            "ngram_size": "8",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "25",
            "judge_borderline_count": "25",
        },
    ],
    [
        {
            "slug": "priority-top120-rerank",
            "top_n": "120",
            "similarity_threshold": "0.88",
            "ngram_size": "4",
            "judge_top_review_count": "95",
            "judge_disagreement_count": "15",
            "judge_borderline_count": "10",
        },
        {
            "slug": "priority-top180-rank-bias",
            "top_n": "180",
            "similarity_threshold": "0.9",
            "ngram_size": "6",
            "judge_top_review_count": "120",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
    ],
    [
        {
            "slug": "precision-top100-tight",
            "top_n": "100",
            "similarity_threshold": "0.94",
            "ngram_size": "9",
            "judge_top_review_count": "80",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "recall-top260-wide",
            "top_n": "260",
            "similarity_threshold": "0.84",
            "ngram_size": "4",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "24",
            "judge_borderline_count": "24",
        },
    ],
    [
        {
            "slug": "rank-early-top90",
            "top_n": "90",
            "similarity_threshold": "0.88",
            "ngram_size": "7",
            "judge_top_review_count": "75",
            "judge_disagreement_count": "22",
            "judge_borderline_count": "22",
        },
        {
            "slug": "rank-late-top240",
            "top_n": "240",
            "similarity_threshold": "0.9",
            "ngram_size": "5",
            "judge_top_review_count": "120",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "clarity-top110-tightly",
            "top_n": "110",
            "similarity_threshold": "0.93",
            "ngram_size": "7",
            "judge_top_review_count": "90",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "diversity-top310-balanced",
            "top_n": "310",
            "similarity_threshold": "0.87",
            "ngram_size": "5",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "26",
            "judge_borderline_count": "26",
        },
    ],
    [
        {
            "slug": "scene-top80-priority",
            "top_n": "80",
            "similarity_threshold": "0.89",
            "ngram_size": "8",
            "judge_top_review_count": "85",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "scene-top260-relaxed",
            "top_n": "260",
            "similarity_threshold": "0.83",
            "ngram_size": "3",
            "judge_top_review_count": "125",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "cadence-top130-strong",
            "top_n": "130",
            "similarity_threshold": "0.95",
            "ngram_size": "10",
            "judge_top_review_count": "100",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "cadence-top290-wide",
            "top_n": "290",
            "similarity_threshold": "0.79",
            "ngram_size": "4",
            "judge_top_review_count": "125",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "copy-safe-top90-strict",
            "top_n": "90",
            "similarity_threshold": "0.91",
            "ngram_size": "9",
            "judge_top_review_count": "95",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "copy-safe-top280-relaxed",
            "top_n": "280",
            "similarity_threshold": "0.82",
            "ngram_size": "2",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "26",
            "judge_borderline_count": "26",
        },
    ],
    [
        {
            "slug": "focus-top120-balanced",
            "top_n": "120",
            "similarity_threshold": "0.92",
            "ngram_size": "8",
            "judge_top_review_count": "100",
            "judge_disagreement_count": "21",
            "judge_borderline_count": "21",
        },
        {
            "slug": "focus-top280-open",
            "top_n": "280",
            "similarity_threshold": "0.81",
            "ngram_size": "6",
            "judge_top_review_count": "132",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "length-top90-precision",
            "top_n": "90",
            "similarity_threshold": "0.94",
            "ngram_size": "7",
            "judge_top_review_count": "90",
            "judge_disagreement_count": "17",
            "judge_borderline_count": "17",
        },
        {
            "slug": "length-top310-coverage",
            "top_n": "310",
            "similarity_threshold": "0.8",
            "ngram_size": "3",
            "judge_top_review_count": "128",
            "judge_disagreement_count": "27",
            "judge_borderline_count": "27",
        },
    ],
    [
        {
            "slug": "semantic-top140-precision",
            "top_n": "140",
            "similarity_threshold": "0.94",
            "ngram_size": "7",
            "judge_top_review_count": "104",
            "judge_disagreement_count": "21",
            "judge_borderline_count": "21",
        },
        {
            "slug": "semantic-top300-coverage",
            "top_n": "300",
            "similarity_threshold": "0.8",
            "ngram_size": "4",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "fluency-top100-lean",
            "top_n": "100",
            "similarity_threshold": "0.92",
            "ngram_size": "6",
            "judge_top_review_count": "92",
            "judge_disagreement_count": "16",
            "judge_borderline_count": "16",
        },
        {
            "slug": "fluency-top270-broad",
            "top_n": "270",
            "similarity_threshold": "0.78",
            "ngram_size": "5",
            "judge_top_review_count": "126",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "imagery-top125-focused",
            "top_n": "125",
            "similarity_threshold": "0.93",
            "ngram_size": "7",
            "judge_top_review_count": "105",
            "judge_disagreement_count": "22",
            "judge_borderline_count": "22",
        },
        {
            "slug": "imagery-top295-wide",
            "top_n": "295",
            "similarity_threshold": "0.81",
            "ngram_size": "6",
            "judge_top_review_count": "129",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "rhythm-top85-strict",
            "top_n": "85",
            "similarity_threshold": "0.95",
            "ngram_size": "8",
            "judge_top_review_count": "98",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "rhythm-top275-broad",
            "top_n": "275",
            "similarity_threshold": "0.8",
            "ngram_size": "4",
            "judge_top_review_count": "124",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "story-top140-lean",
            "top_n": "140",
            "similarity_threshold": "0.93",
            "ngram_size": "7",
            "judge_top_review_count": "108",
            "judge_disagreement_count": "21",
            "judge_borderline_count": "21",
        },
        {
            "slug": "story-top360-broad",
            "top_n": "360",
            "similarity_threshold": "0.79",
            "ngram_size": "5",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "hook-top95-precision",
            "top_n": "95",
            "similarity_threshold": "0.96",
            "ngram_size": "8",
            "judge_top_review_count": "102",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "hook-top300-open",
            "top_n": "300",
            "similarity_threshold": "0.81",
            "ngram_size": "4",
            "judge_top_review_count": "132",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "cohesion-top120-focused",
            "top_n": "120",
            "similarity_threshold": "0.92",
            "ngram_size": "9",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "cohesion-top330-wide",
            "top_n": "330",
            "similarity_threshold": "0.82",
            "ngram_size": "3",
            "judge_top_review_count": "128",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "energy-top80-strong",
            "top_n": "80",
            "similarity_threshold": "0.94",
            "ngram_size": "8",
            "judge_top_review_count": "96",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "energy-top250-wide",
            "top_n": "250",
            "similarity_threshold": "0.8",
            "ngram_size": "5",
            "judge_top_review_count": "126",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "narrative-top100-tuned",
            "top_n": "100",
            "similarity_threshold": "0.95",
            "ngram_size": "9",
            "judge_top_review_count": "102",
            "judge_disagreement_count": "16",
            "judge_borderline_count": "16",
        },
        {
            "slug": "narrative-top280-open",
            "top_n": "280",
            "similarity_threshold": "0.82",
            "ngram_size": "4",
            "judge_top_review_count": "128",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "motif-top120-focused",
            "top_n": "120",
            "similarity_threshold": "0.93",
            "ngram_size": "7",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "motif-top320-broad",
            "top_n": "320",
            "similarity_threshold": "0.81",
            "ngram_size": "3",
            "judge_top_review_count": "132",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "cadence-top75-lean",
            "top_n": "75",
            "similarity_threshold": "0.96",
            "ngram_size": "8",
            "judge_top_review_count": "92",
            "judge_disagreement_count": "16",
            "judge_borderline_count": "16",
        },
        {
            "slug": "cadence-top300-open",
            "top_n": "300",
            "similarity_threshold": "0.79",
            "ngram_size": "4",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "hook-top110-rigid",
            "top_n": "110",
            "similarity_threshold": "0.94",
            "ngram_size": "8",
            "judge_top_review_count": "108",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "hook-top330-open",
            "top_n": "330",
            "similarity_threshold": "0.8",
            "ngram_size": "4",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "bridge-top95-lean",
            "top_n": "95",
            "similarity_threshold": "0.95",
            "ngram_size": "9",
            "judge_top_review_count": "102",
            "judge_disagreement_count": "16",
            "judge_borderline_count": "16",
        },
        {
            "slug": "bridge-top300-broad",
            "top_n": "300",
            "similarity_threshold": "0.82",
            "ngram_size": "5",
            "judge_top_review_count": "132",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "impact-top90-intense",
            "top_n": "90",
            "similarity_threshold": "0.96",
            "ngram_size": "7",
            "judge_top_review_count": "96",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "impact-top290-wide",
            "top_n": "290",
            "similarity_threshold": "0.79",
            "ngram_size": "4",
            "judge_top_review_count": "128",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "syntax-top120-stable",
            "top_n": "120",
            "similarity_threshold": "0.92",
            "ngram_size": "7",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "syntax-top330-wide",
            "top_n": "330",
            "similarity_threshold": "0.81",
            "ngram_size": "3",
            "judge_top_review_count": "134",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "punch-top95-focused",
            "top_n": "95",
            "similarity_threshold": "0.95",
            "ngram_size": "8",
            "judge_top_review_count": "98",
            "judge_disagreement_count": "17",
            "judge_borderline_count": "17",
        },
        {
            "slug": "punch-top280-wide",
            "top_n": "280",
            "similarity_threshold": "0.8",
            "ngram_size": "4",
            "judge_top_review_count": "126",
            "judge_disagreement_count": "28",
            "judge_borderline_count": "28",
        },
    ],
    [
        {
            "slug": "flow-top110-tuned",
            "top_n": "110",
            "similarity_threshold": "0.94",
            "ngram_size": "9",
            "judge_top_review_count": "104",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "flow-top320-open",
            "top_n": "320",
            "similarity_threshold": "0.81",
            "ngram_size": "3",
            "judge_top_review_count": "132",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "vibe-top85-lean",
            "top_n": "85",
            "similarity_threshold": "0.96",
            "ngram_size": "8",
            "judge_top_review_count": "94",
            "judge_disagreement_count": "16",
            "judge_borderline_count": "16",
        },
        {
            "slug": "vibe-top290-broad",
            "top_n": "290",
            "similarity_threshold": "0.79",
            "ngram_size": "4",
            "judge_top_review_count": "128",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
    [
        {
            "slug": "wording-top130-sane",
            "top_n": "130",
            "similarity_threshold": "0.93",
            "ngram_size": "7",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "wording-top300-open",
            "top_n": "300",
            "similarity_threshold": "0.82",
            "ngram_size": "5",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "emotion-top105-focused",
            "top_n": "105",
            "similarity_threshold": "0.94",
            "ngram_size": "8",
            "judge_top_review_count": "108",
            "judge_disagreement_count": "18",
            "judge_borderline_count": "18",
        },
        {
            "slug": "emotion-top300-open",
            "top_n": "300",
            "similarity_threshold": "0.79",
            "ngram_size": "4",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "30",
            "judge_borderline_count": "30",
        },
    ],
    [
        {
            "slug": "metaphor-top120-rich",
            "top_n": "120",
            "similarity_threshold": "0.93",
            "ngram_size": "9",
            "judge_top_review_count": "110",
            "judge_disagreement_count": "20",
            "judge_borderline_count": "20",
        },
        {
            "slug": "metaphor-top310-wide",
            "top_n": "310",
            "similarity_threshold": "0.8",
            "ngram_size": "5",
            "judge_top_review_count": "130",
            "judge_disagreement_count": "29",
            "judge_borderline_count": "29",
        },
    ],
]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_slug(value: str, *, max_len: int = 48) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "-", value.lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] if text else "sweep"


def ensure_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_candidate_sweep_profiles() -> list[dict[str, Any]]:
    if not DEFAULT_SWEEP_ROOT.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for sweep_dir in sorted(DEFAULT_SWEEP_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not sweep_dir.is_dir():
            continue
        sweep_raw = sweep_dir / "sweep_raw.jsonl"
        if not sweep_raw.exists():
            continue
        if sweep_raw == DEFAULT_INPUT_SWEEP:
            continue
        candidates.append(
            {
                "display_name": sweep_dir.name,
                "slug": sanitize_slug(sweep_dir.name),
                "sweep_path": str(sweep_raw),
                "size_kb": round(sweep_raw.stat().st_size / 1024.0, 1),
                "status": "pending",
            }
        )
    return candidates


def build_sweep_profile_experiments(
    profile_id: str,
    cycle: int,
    profile: dict[str, Any],
    *,
    judge_seed: int,
) -> list[dict[str, Any]]:
    input_path = profile["sweep_path"]
    slug = profile["slug"]
    label = f"Profile #{cycle} ({profile['display_name']})"
    return [
        {
            "id": f"{profile_id}_eval_v1",
            "slug": f"{slug}-eval",
            "type": "evaluation",
            "name": f"{label}: replay eval on alternate sweep",
            "hypothesis": "Alternate sweeps can reveal profile-level improvements.",
            "status": "pending",
            "candidate_id": profile_id,
            "candidate_name": profile["display_name"],
            "input": input_path,
            "command": [
                sys.executable,
                ROOT / "scripts" / "evaluate_generation_outputs.py",
                "--input",
                "${input}",
                "--prompts",
                DEFAULT_PROMPTS,
                "--train-corpus",
                DEFAULT_TRAIN_CORPUS,
                "--out",
                "${run_dir}/eval_metrics.json",
                "--sample-md",
                "${run_dir}/sample_outputs.md",
            ],
        },
        {
            "id": f"{profile_id}_rank_v1",
            "slug": f"{slug}-rank",
            "type": "ranking",
            "name": f"{label}: rank this profile",
            "hypothesis": "Consistent reranking can surface higher-quality candidates.",
            "status": "pending",
            "candidate_id": profile_id,
            "candidate_name": profile["display_name"],
            "input": input_path,
            "command": [
                sys.executable,
                ROOT / "scripts" / "rank_qwen3_quality.py",
                "--input",
                "${input}",
                "--prompts",
                DEFAULT_PROMPTS,
                "--output-md",
                "${run_dir}/rank_review_queue.md",
                "--output-jsonl",
                "${run_dir}/ranked_quality.jsonl",
                "--summary-json",
                "${run_dir}/rank_summary.json",
                "--top-n",
                "150",
            ],
        },
        {
            "id": f"{profile_id}_judge_v1",
            "slug": f"{slug}-judge",
            "type": "judge",
            "name": f"{label}: judge this profile",
            "hypothesis": "Candidate quality should improve versus incumbent top-50 usable band.",
            "status": "pending",
            "candidate_id": profile_id,
            "candidate_name": profile["display_name"],
            "input": input_path,
            "command": [
                sys.executable,
                ROOT / "scripts" / "judge_qwen3_quality_openai.py",
                "--input-ranked",
                "${run_dir}/ranked_quality.jsonl",
                "--output-dir",
                "${run_dir}/judge",
                "--mock-judge",
                "--top-review-count",
                "100",
                "--disagreement-review-count",
                "30",
                "--borderline-review-count",
                "20",
                "--seed",
                str(judge_seed),
            ],
        },
    ]


def get_post_sweep_round_variants(round_index: int) -> list[dict[str, Any]]:
    if round_index < 0 or round_index >= len(POST_SWEEP_ROUND_VARIANTS):
        return []
    return POST_SWEEP_ROUND_VARIANTS[round_index]


def select_next_post_sweep_profile(state: dict[str, Any]) -> tuple[dict[str, Any], int] | None:
    candidates = [p for p in state.get("sweep_profiles", []) if p.get("status") == "complete"]
    if not candidates:
        return None
    progress = {str(k): int(v or 0) for k, v in (state.get("post_sweep_profile_rounds", {}) or {}).items()}
    next_candidates: list[tuple[dict[str, Any], int]] = []
    for candidate in candidates:
        rounds_done = progress.get(str(candidate.get("id")), 0)
        next_round = rounds_done + 1
        if get_post_sweep_round_variants(next_round - 1):
            next_candidates.append((candidate, next_round))
    if not next_candidates:
        return None
    return sorted(
        next_candidates,
        key=lambda item: (item[0].get("best_top50_judge_usable_rate") or 0.0, int(item[0].get("cycle") or 0)),
        reverse=True,
    )[0]


def find_latest_ranked_json(state: dict[str, Any], candidate_id: str) -> str:
    for run in reversed(state.get("run_history", [])):
        if run.get("type") == "ranking" and run.get("candidate_id") == candidate_id:
            run_dir = run.get("run_dir")
            if not run_dir:
                continue
            ranked_path = ROOT / run_dir / "ranked_quality.jsonl"
            if ranked_path.exists():
                return str(ranked_path)
    return ""


def build_post_sweep_experiments(
    profile_id: str,
    profile_name: str,
    profile_path: str,
    state: dict[str, Any],
    *,
    judge_seed: int,
    profile_round: int,
) -> list[dict[str, Any]]:
    candidate_slug = sanitize_slug(profile_name)
    ranked_reference = find_latest_ranked_json(state, profile_id)
    experiments: list[dict[str, Any]] = []
    variants = get_post_sweep_round_variants(profile_round - 1)
    for variant_index, variant in enumerate(variants, start=1):
        variant_suffix = variant["slug"]
        experiments.append(
            {
                "id": f"{profile_id}_post_{variant_suffix}_rank_v1",
                "slug": f"post-{candidate_slug}-{variant_suffix}",
                "type": "ranking",
                "name": f"Post-sweep sensitivity profile: {profile_name} ({variant_suffix})",
                "hypothesis": "Post-sweep rank tuning can improve top-50 quality without new data.",
                "status": "pending",
                "candidate_id": None,
                "candidate_name": profile_name,
                "post_sweep_round": profile_round,
                "input": profile_path,
                "command": [
                    sys.executable,
                    ROOT / "scripts" / "rank_qwen3_quality.py",
                    "--input",
                    "${input}",
                    "--prompts",
                    DEFAULT_PROMPTS,
                    "--output-md",
                    "${run_dir}/rank_review_queue.md",
                    "--output-jsonl",
                    "${run_dir}/ranked_quality.jsonl",
                    "--summary-json",
                    "${run_dir}/rank_summary.json",
                    "--top-n",
                    variant["top_n"],
                    "--similarity-threshold",
                    variant["similarity_threshold"],
                    "--ngram-size",
                    variant["ngram_size"],
                ],
            }
        )
        if ranked_reference:
            experiments.append(
                {
                    "id": f"{profile_id}_post_{variant_suffix}_judge_v1",
                    "slug": f"post-{candidate_slug}-{variant_suffix}-judge",
                    "type": "judge",
                    "name": f"Post-sweep judge policy: {profile_name} ({variant_suffix})",
                    "hypothesis": "Judge thresholds can materially change usable-band selection in top-50.",
                    "status": "pending",
                "candidate_id": None,
                "candidate_name": profile_name,
                "post_sweep_round": profile_round,
                "input": ranked_reference,
                "command": [
                    sys.executable,
                    ROOT / "scripts" / "judge_qwen3_quality_openai.py",
                        "--input-ranked",
                        "${input}",
                    "--output-dir",
                    "${run_dir}/judge",
                    "--mock-judge",
                    "--top-review-count",
                    variant.get("judge_top_review_count", "90"),
                    "--disagreement-review-count",
                    variant.get("judge_disagreement_count", "20"),
                    "--borderline-review-count",
                    variant.get("judge_borderline_count", "20"),
                    "--seed",
                    str(judge_seed + variant_index),
                ],
            }
        )
    return experiments


def enqueue_post_sweep_experiments(state: dict[str, Any], *, batch_size: int) -> int:
    if batch_size <= 0:
        return 0

    processed = {str(item) for item in state.get("post_sweep_processed_candidates", [])}
    next_candidate = select_next_post_sweep_profile(state)
    if not next_candidate:
        return 0

    selected_profile, selected_round = next_candidate
    selected_id = str(selected_profile["id"])

    profile_name = str(selected_profile.get("display_name") or selected_profile.get("sweep_path"))
    profile_path = str(selected_profile.get("sweep_path"))
    experiments = build_post_sweep_experiments(
        profile_id=selected_id,
        profile_name=profile_name,
        profile_path=profile_path,
        state=state,
        judge_seed=20260706 + len(processed) * 100,
        profile_round=selected_round,
    )
    processed_rounds = {
        str(key): int(value or 0)
        for key, value in (state.get("post_sweep_profile_rounds", {}) or {}).items()
    }
    processed_rounds[selected_id] = selected_round
    state["post_sweep_profile_rounds"] = processed_rounds
    state["post_sweep_processed_candidates"] = sorted(list(processed | {selected_id}))
    state["post_sweep_round_counter"] = int(state.get("post_sweep_round_counter", 0)) + 1
    state["post_sweep_profiles"] = state.get("post_sweep_profiles", [])
    state["post_sweep_profiles"].append(f"{selected_id}:r{selected_round}")
    for experiment in experiments[: max(1, batch_size) * len(get_post_sweep_round_variants(selected_round - 1)) * 2]:
        state["experiment_queue"].append(experiment)

    return len(experiments)


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in value]
    return value


def safe_dumps(value: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> str:
    return json.dumps(_normalize_jsonable(value), ensure_ascii=ensure_ascii, indent=indent)


def write_json(path: Path, payload: Any) -> None:
    normalized = _normalize_jsonable(payload)
    ensure_path(path).write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    ensure_path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_or_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return
    ensure_path(dst)
    shutil.copy2(src, dst)


def refresh_labeled_artifact_mirror(state: dict[str, Any], run_dir: Path | None = None, run_record: dict[str, Any] | None = None) -> None:
    """Mirror core loop artifacts into a labeled folder at repo root."""
    LABELED_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LABELED_DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    LABELED_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    copy_tree_or_file(DOCS_ROOT, LABELED_DOCS_ROOT)
    if STATE_PATH.exists():
        copy_tree_or_file(STATE_PATH, LABELED_ARTIFACT_ROOT / "state.json")

    if run_record:
        target_dir = LABELED_RUNS_ROOT / str(run_record.get("run_id", "unknown"))
        if run_dir is None and run_record.get("run_dir"):
            run_dir = ROOT / str(run_record["run_dir"])
        if run_dir is not None and run_dir.exists():
            copy_tree_or_file(run_dir, target_dir)

    index_payload_path = LABELED_ARTIFACT_ROOT / "artifact_index.json"
    prior_index = load_json(index_payload_path, default={}) if index_payload_path.exists() else {}
    latest_run_dir = None
    latest_run_id = None
    if not run_dir and state.get("run_history"):
        latest = state["run_history"][-1]
        latest_run_dir = latest.get("run_dir")
        latest_run_id = latest.get("run_id")
        if latest_run_dir:
            latest_run_dir = str(ROOT / latest_run_dir)
    index_payload = {
        "updated_at": now_iso(),
        "state_path": str(STATE_PATH),
        "labeled_root": str(LABELED_ARTIFACT_ROOT),
        "run_dir": str(run_dir) if run_dir is not None else latest_run_dir or prior_index.get("run_dir", None),
        "run_id": run_record.get("run_id") if run_record is not None else latest_run_id or prior_index.get("run_id", None),
        "recent_state_snapshot": "state.json",
        "bible_references": [str(p) for p in BIBLE_MARKER_FILES],
    }
    write_json(index_payload_path, index_payload)


def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> CommandResult:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        shell=False,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, duration_seconds=round(elapsed, 4))


def read_lines_from_jsonl(path: Path, count: int = 5) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= count:
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def gather_repo_fingerprint() -> dict[str, Any]:
    import subprocess as sp

    git_rev = "unknown"
    git_branch = "unknown"
    git_dirty = False
    try:
        git_rev = sp.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    try:
        git_branch = sp.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    try:
        status = sp.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
        git_dirty = bool(status)
    except Exception:
        pass
    return {
        "git": {"commit": git_rev, "branch": git_branch, "dirty": git_dirty},
        "python": sys.version,
        "platform": os.name,
    }


def build_incumbent_profile() -> dict[str, Any]:
    eval_baseline = load_json(DEFAULT_BASELINE_EVAL, default={})
    judge_baseline = load_json(DEFAULT_BASELINE_JUDGE, default={})
    structural_rate = eval_baseline.get("summary", {}).get("exact_line_match_rate")
    if structural_rate is None:
        structural_rate = eval_baseline.get("summary", {}).get("structural_pass_rate")
    return {
        "name": "qwen3_4b_base_12line_v1",
        "model": "Qwen/Qwen3-4B base",
        "adapter": "none",
        "status": "active",
        "gate_targets": {
            "structural_pass_rate": float(structural_rate) if structural_rate is not None else 0.995,
            "slur_violation_rate": 0.0,
            "prompt_leakage_rate": 0.0,
            "incomplete_ending_rate": float(eval_baseline.get("summary", {}).get("incomplete_ending_rate", 0.0)),
            "copy_similarity_rate": float(eval_baseline.get("summary", {}).get("high_copy_similarity_rate", 0.0)),
            "top50_judge_usable_rate": float(judge_baseline.get("top50_judge_usable_rate", 1.0)),
        },
        "baseline_eval_report": str(DEFAULT_BASELINE_EVAL),
        "baseline_judge_report": str(DEFAULT_BASELINE_JUDGE),
    }


def build_default_queue() -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen3_prompt_library_refresh_v1",
            "slug": "prompt-library",
            "type": "prompt_library",
            "name": "Refresh 12-line prompt bank",
            "hypothesis": "Stable prompt inputs improve comparability and experiment reproducibility.",
            "status": "pending",
            "command": [
                sys.executable,
                ROOT / "scripts" / "build_qwen3_eval_prompts.py",
            ],
            "outputs": {
                "prompt_set": str(ROOT / "data" / "prompts" / "qwen3_4b_12line_expanded_eval_prompts.json")
            },
        },
        {
            "id": "qwen3_eval_replay_line_enforced_v1",
            "slug": "eval-replay",
            "type": "evaluation",
            "name": "Replay incumbent eval metrics on existing sweep",
            "hypothesis": "Recomputing baseline metrics confirms contract compliance before experimenting.",
            "status": "pending",
            "candidate_id": "baseline",
            "input": str(DEFAULT_INPUT_SWEEP),
            "command": [
                sys.executable,
                ROOT / "scripts" / "evaluate_generation_outputs.py",
                "--input",
                "${input}",
                "--prompts",
                DEFAULT_PROMPTS,
                "--train-corpus",
                DEFAULT_TRAIN_CORPUS,
                "--out",
                "${run_dir}/eval_metrics.json",
                "--sample-md",
                "${run_dir}/sample_outputs.md",
            ],
        },
        {
            "id": "qwen3_ranker_sweep_replay_v1",
            "slug": "ranker-pass",
            "type": "ranking",
            "name": "Run ranker over incumbent eval sweep",
            "hypothesis": "Current heuristics should remain stable and identify strongest baseline candidates.",
            "status": "pending",
            "candidate_id": "baseline",
            "input": str(DEFAULT_INPUT_SWEEP),
            "command": [
                sys.executable,
                ROOT / "scripts" / "rank_qwen3_quality.py",
                "--input",
                "${input}",
                "--prompts",
                DEFAULT_PROMPTS,
                "--output-md",
                "${run_dir}/rank_review_queue.md",
                "--output-jsonl",
                "${run_dir}/ranked_quality.jsonl",
                "--summary-json",
                "${run_dir}/rank_summary.json",
            ],
        },
        {
            "id": "qwen3_judge_mock_v1",
            "slug": "judge-mock",
            "type": "judge",
            "name": "Mock OpenAI quality judge pass over ranked rows",
            "hypothesis": "Mock judging should produce deterministic pass/fail signals for comparison gating.",
            "status": "pending",
            "command": [
                sys.executable,
                ROOT / "scripts" / "judge_qwen3_quality_openai.py",
                "--input-ranked",
                "${run_dir}/ranked_quality.jsonl",
                "--output-dir",
                "${run_dir}/judge",
                "--mock-judge",
                "--top-review-count",
                "80",
                "--disagreement-review-count",
                "20",
                "--borderline-review-count",
                "20",
                "--seed",
                "20260706",
            ],
        },
        {
            "id": "qwen3_rap_quality_tools_smoke_v1",
            "slug": "tools-smoke",
            "type": "tests",
            "name": "Smoke test quality toolchain",
            "hypothesis": "Focused tool test keeps the loop mechanically safe for structural metrics.",
            "status": "pending",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_rap_quality_tools.py::RapQualityToolTests::test_generation_evaluator_reports_line_and_copy_metrics",
                "-q",
            ],
        },
    ]


def write_docs_scaffold(state: dict[str, Any]) -> None:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    (DOCS_ROOT / "judge_prompts").mkdir(exist_ok=True)
    (DOCS_ROOT / "judge_transcripts").mkdir(exist_ok=True)
    (DOCS_ROOT / "repo_inventory").mkdir(exist_ok=True)

    readme = [
        "# qwen3_rap_quality_autonomous_loop",
        "",
        "This folder is the governance boundary for the autonomous quality-improvement loop.",
        "",
        "## Canonical Sources (always referenced)",
        f"- Documentation milestone: {BIBLE_MARKER_FILES[0]}",
        f"- Comprehensive goal: {BIBLE_MARKER_FILES[1]}",
        f"- Short goal summary: {GOAL_PATH}",
        "",
        "## Operating Rules",
        "- Never print secrets or dump `.env` values.",
        "- Preserve unknown-license data as experimental-only.",
        "- Keep all meaningful experiments in `state.json` and this folder.",
        "- Every experiment must emit: run_manifest.json, config_snapshot/, metrics.json, summary.md, decision.md,",
        "  stdout.log, stderr.log, sample_outputs.md.",
        f"- All loop artifacts are mirrored under `{LABELED_ARTIFACT_ROOT}` for repository-root evidence review.",
        "",
        "## Current Objective",
        "- Keep the incumbent baseline intact.",
        "- Improve 12-line rap quality under quality gates.",
        "- Stop only on explicit hard safety/legal blocker, time exhaustion, queue exhaustion, or promotion decision.",
    ]
    write_markdown(DOCS_ROOT / "README.md", readme)

    write_markdown(
        DOCS_ROOT / "dataset_lineage.md",
        [
            "# Dataset Lineage",
            "",
            "- Source corpus: `data/processed/` and `data/curated/...` pipelines.",
            "- SFT lineage: `data/training/` outputs.",
            "- Preference lineage: `data/preferences*/` outputs.",
            "- Risk/unknown-license artifacts remain marked `experimental_only` until explicitly approved.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "model_registry.md",
        [
            "# Model Registry",
            "",
            "- Baseline model: Qwen/Qwen3-4B base (`qwen3_4b_base_12line_v1`).",
            "- Adapter candidates are logged in `state.json` and `iteration_log.md` only when tested.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "prompt_registry.md",
        [
            "# Prompt Registry",
            "",
            "- Canonical prompt bank path: `configs/prompts/qwen3_4b_12line_expanded_eval_prompts.json`.",
            "- Prompt refresh experiments should update this file and record snapshots in `run_dir/config_snapshot/`.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "judge_registry.md",
        [
            "# Judge Registry",
            "",
            "- Primary judge path: `scripts/judge_qwen3_quality_openai.py`.",
            "- Judge runs log to: `advisor_prompts.md`, `advisor_responses.md`, `advisor_transcripts/`.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "quality_registry.md",
        [
            "# Quality Registry",
            "",
            "- Gates are maintained in state and evaluated per experiment.",
            "- `validation_matrix.md` is updated at each iteration with gate-by-gate status.",
            "- New sweep profiles from `data/sweeps` are appended from the loop as long as candidate profiles remain.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "sweep_profiles.md",
        [
            "# Sweep Profiles",
            "",
            "- Profiles are discovered from `data/sweeps/*/sweep_raw.jsonl`.",
            "- The loop appends evaluation/ranking/judge profile cycles in configurable batches until exhausted or the budget ends.",
        ],
    )
    write_markdown(
        DOCS_ROOT / "validation_matrix.md",
        [
            "# Validation Matrix",
            "",
            "| Gate | Target | Value | Status |",
            "| --- | --- | --- | --- |",
            "| exact 12-line structural pass | >= 99.5% | unknown | pending |",
            "| slur violations | 0 | unknown | pending |",
            "| prompt leakage | 0 | unknown | pending |",
            "| incomplete endings | <=0.10% | unknown | pending |",
            "| high copy similarity | 0 | unknown | pending |",
            "| top-50 judge usable | > incumbent | unknown | pending |",
        ],
    )

    for fname in [
        "iteration_log.md",
        "experiment_queue.md",
        "decision_log.md",
        "failed_ideas.md",
        "best_results.md",
        "open_questions.md",
        "artifact_manifest.md",
        "advisor_prompts.md",
        "advisor_responses.md",
        "sweep_profiles.md",
    ]:
        target = DOCS_ROOT / fname
        if not target.exists():
            target.write_text(f"# {fname.replace('_', ' ').replace('.md','').title()}\n\n", encoding="utf-8")

    (DOCS_ROOT / "README.md").write_text(
        (DOCS_ROOT / "README.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if state:
        update_experiment_queue_markdown(state)
        update_validation_matrix(state)
        update_artifact_manifest(state)
        update_sweep_profile_log(state)


def update_experiment_queue_markdown(state: dict[str, Any]) -> None:
    rows = ["# Experiment Queue", "", "| id | name | status | attempts | type |", "| --- | --- | --- | --- | --- |"]
    for exp in state.get("experiment_queue", []):
        rows.append(
            f"| {exp['id']} | {exp.get('name', '')} | {exp.get('status', 'pending')} | "
            f"{exp.get('attempts', 0)} | {exp.get('type', '')} |"
        )
    write_markdown(DOCS_ROOT / "experiment_queue.md", rows)


def update_validation_matrix(state: dict[str, Any]) -> None:
    latest = (state.get("run_history") or [])[-1:] or []
    if not latest:
        return
    gates = latest[0].get("gate_results", {})
    rows = [
        "# Validation Matrix",
        "",
        "| Gate | Target | Value | Status |",
        "| --- | --- | --- | --- |",
        f"| exact 12-line structural pass | >= 99.5% | {gates.get('exact_line_match_rate', 'N/A')} | "
        f"{'pass' if gates.get('exact_line_match_rate_pass', False) else 'fail'} |",
        f"| slur violations | 0 | {gates.get('slur_violation_rate', 'N/A')} | "
        f"{'pass' if gates.get('slur_violation_rate_pass', False) else 'fail'} |",
        f"| prompt leakage | 0 | {gates.get('prompt_leakage_rate', 'N/A')} | "
        f"{'pass' if gates.get('prompt_leakage_rate_pass', False) else 'fail'} |",
        f"| incomplete endings | <=0.1% | {gates.get('incomplete_ending_rate', 'N/A')} | "
        f"{'pass' if gates.get('incomplete_ending_rate_pass', False) else 'fail'} |",
        f"| high copy similarity | 0 | {gates.get('high_copy_similarity_rate', 'N/A')} | "
        f"{'pass' if gates.get('high_copy_similarity_rate_pass', False) else 'fail'} |",
        f"| top-50 judge usable | > incumbent | {gates.get('top50_judge_usable_rate', 'N/A')} | "
        f"{'pass' if gates.get('top50_judge_usable_rate_pass', False) else 'fail'} |",
    ]
    write_markdown(DOCS_ROOT / "validation_matrix.md", rows)


def update_artifact_manifest(state: dict[str, Any]) -> None:
    lines = ["# Artifact Manifest", ""]
    lines.append("| run_id | status | decision | artifact_path |")
    lines.append("| --- | --- | --- | --- |")
    for run in state.get("run_history", []):
        lines.append(f"| {run['run_id']} | {run['status']} | {run.get('decision', 'continue_research')} | {run['run_dir']} |")
    write_markdown(DOCS_ROOT / "artifact_manifest.md", lines)


def update_sweep_profile_log(state: dict[str, Any]) -> None:
    lines = ["# Sweep Profiles", ""]
    lines.append("| profile_id | display_name | status | cycle | last_run | top50_usable |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for profile in state.get("sweep_profiles", []):
        lines.append(
            f"| {profile['id']} | {profile.get('display_name', 'unknown')} | {profile.get('status', 'pending')} "
            f"| {profile.get('cycle', 'n/a')} | {profile.get('last_run', 'n/a')} | "
            f"{profile.get('best_top50_judge_usable_rate', profile.get('top50_judge_usable_rate', 'n/a'))} |"
        )
    write_markdown(DOCS_ROOT / "sweep_profiles.md", lines)


def update_iteration_log(state: dict[str, Any], run_record: dict[str, Any]) -> None:
    lines = [
        "# Iteration Log",
        "",
        f"- timestamp: {now_iso()}",
        f"- run_id: {run_record['run_id']}",
        f"- experiment: {run_record['experiment_id']} ({run_record['name']})",
        f"- candidate: {run_record.get('candidate_id', 'n/a')}",
        f"- status: {run_record['status']}",
        f"- wall_seconds: {run_record.get('wall_seconds', 0.0)}",
        f"- decision: {run_record.get('decision', 'continue_research')}",
        "",
    ]
    decision = run_record.get("decision", "continue_research")
    if decision and decision != "continue_research":
        lines.append(f"- promotion decision: {decision}")
        lines.append(f"- incumbent: {state.get('incumbent', {}).get('name','unknown')}")
    lines.append("")
    lines.extend(
        [
            "## Command",
            f"- `{run_record.get('command', 'N/A')}`",
            "",
            "## Metrics Snapshot",
            f"- `run_summary`: {safe_dumps(run_record.get('summary', {}), ensure_ascii=False)}",
            "",
        ]
    )
    marker = DOCS_ROOT / "iteration_log.md"
    existing = marker.read_text(encoding="utf-8") if marker.exists() else "# Iteration Log\n\n"
    marker.write_text(existing + "\n" + "\n".join(lines), encoding="utf-8")


def update_decision_log(state: dict[str, Any], run_record: dict[str, Any]) -> None:
    lines = [
        "# Decision Log",
        "",
        f"- {now_iso()}: run {run_record['run_id']} -> {run_record.get('decision', 'continue_research')}",
        f"- experiment {run_record['experiment_id']} ({run_record['name']})",
        f"- candidate {run_record.get('candidate_id', 'n/a')}",
        f"- next_resume_command: {state.get('next_resume_command', '')}",
        "",
    ]
    marker = DOCS_ROOT / "decision_log.md"
    existing = marker.read_text(encoding="utf-8") if marker.exists() else "# Decision Log\n\n"
    marker.write_text(existing + "\n" + "\n".join(lines), encoding="utf-8")


def write_advisor_records(prompt: str, response_record: dict[str, Any]) -> None:
    prompt_file = DOCS_ROOT / "advisor_prompts.md"
    response_file = DOCS_ROOT / "advisor_responses.md"
    transcript_dir = DOCS_ROOT / "advisor_transcripts"
    transcript_dir.mkdir(exist_ok=True)

    run_token = uuid.uuid4().hex[:8]
    prompt_path = transcript_dir / f"{run_token}_prompt_messages.json"
    request_path = transcript_dir / f"{run_token}_request.json"
    raw_path = transcript_dir / f"{run_token}_raw_response.json"
    parsed_path = transcript_dir / f"{run_token}_parsed_response.json"
    transcript_path = transcript_dir / f"{run_token}_transcript.md"

    prompt_path.write_text(safe_dumps({"prompt": prompt}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    request_path.write_text(
        safe_dumps(response_record.get("request", {}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_path.write_text(
        safe_dumps(response_record.get("raw_response", {}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    parsed_path.write_text(
        safe_dumps(response_record.get("parsed_response", {}), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    transcript_path.write_text(
        "\n".join(
            [
                "# advisor transcript",
                "",
                "## prompt",
                "```",
                prompt,
                "```",
                "## parsed_response",
                "```json",
                safe_dumps(response_record.get("parsed_response", {}), indent=2, ensure_ascii=False),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    previous = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else "# Advisor Prompts\n\n"
    prompt_file.write_text(previous + f"\n## {run_token}\n```text\n{prompt}\n```\n", encoding="utf-8")
    previous = response_file.read_text(encoding="utf-8") if response_file.exists() else "# Advisor Responses\n\n"
    response_file.write_text(
        previous
        + f"\n## {run_token}\n```json\n{safe_dumps(response_record.get('parsed_response', {}), ensure_ascii=False)}\n```\n",
        encoding="utf-8",
    )


def run_advisor(state: dict[str, Any], run_dir: Path, run_record: dict[str, Any]) -> dict[str, Any]:
    latest = state.get("run_history", [])[-1:] or []
    metrics = latest[0].get("summary", {}) if latest else {}
    prompt = (
        "Given the latest metrics, failures, best samples, and prior attempts, "
        "what are the next 3-5 highest-leverage experiments to improve rap generation quality "
        "without hurting structure?\n\n"
        f"State: {safe_dumps(state.get('incumbent', {}), ensure_ascii=False)}\n"
        f"Latest run: {safe_dumps(metrics, ensure_ascii=False)}\n"
        f"Queue depth: {len(state.get('experiment_queue', []))}\n"
    )

    if not os.getenv("OPENAI_API_KEY"):
        parsed = {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not configured",
            "suggestions": [
                "Run more mock-gated local experiments first (eval+rank+judge).",
                "Prioritize underlength-retry policy and judge/ranker calibration changes after passing structure gates.",
            ],
        }
        write_advisor_records(prompt, {"request": {"reason": "no_api_key"}, "raw_response": {}, "parsed_response": parsed})
        return {"advisor": parsed, "status": "skipped"}

    try:
        import openai
    except Exception as exc:
        parsed = {
            "status": "error",
            "error": f"openai package unavailable: {exc}",
            "suggestions": [],
        }
        write_advisor_records(prompt, {"request": {"reason": "import_error"}, "raw_response": {}, "parsed_response": parsed})
        return {"advisor": parsed, "status": "skipped"}

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
    request_payload = {
        "model": os.getenv("OPENAI_ADVISOR_MODEL", "gpt-4.1-mini"),
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    raw = client.chat.completions.create(**request_payload)
    raw_content = raw.choices[0].message.content if raw and raw.choices else "{}"
    try:
        parsed = json.loads(str(raw_content or "{}"))
    except Exception:
        parsed = {"status": "parse_error", "content": str(raw_content), "suggestions": []}

    raw_response = raw.dict() if hasattr(raw, "dict") else {}
    write_advisor_records(
        prompt,
        {
            "request": request_payload,
            "raw_response": raw_response,
            "parsed_response": parsed,
        },
    )
    return {"advisor": parsed, "status": "ok"}


def run_single_experiment(state: dict[str, Any], experiment: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_id = f"{timestamp()}_{experiment['slug']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": experiment["id"],
        "name": experiment.get("name"),
        "started_at": now_iso(),
        "state_before": state,
    }

    # Configuration snapshot
    config_snapshot_dir = run_dir / "config_snapshot"
    config_snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_snapshot_dir / "loop_config.json", {"state": state, "experiment": experiment, "command_template": experiment["command"]})
    write_json(config_snapshot_dir / "repo_fingerprint.json", gather_repo_fingerprint())
    write_markdown(run_dir / "summary.md", [f"# Experiment {run_id}", f"- name: {experiment.get('name')}", f"- type: {experiment['type']}"])

    def resolve_command(command_template: list[Any], run_dir_path: Path, experiment_ctx: dict[str, Any]) -> list[str]:
        resolved: list[str] = []
        replacements = {
            "${run_dir}": str(run_dir_path),
            "${state_json}": str(STATE_PATH),
            "${input}": str(experiment_ctx.get("input", "")),
        }
        for part in command_template:
            text = str(part)
            for token, replacement in replacements.items():
                text = text.replace(token, replacement)
            resolved.append(text)
        return resolved

    command = resolve_command(experiment["command"], run_dir, experiment)

    manifest["command"] = " ".join(command)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    metrics: dict[str, Any] = {"type": experiment["type"], "status": "running"}
    gate_results: dict[str, Any] = {}
    decision = "continue_research"
    status = "failed"

    if experiment["type"] in {"prompt_library", "evaluation", "ranking", "judge", "tests"}:
        result = run_command(command, ROOT, stdout_log, stderr_log)
        metrics["command"] = {"returncode": result.returncode, "duration_seconds": result.duration_seconds}
        manifest["command_result"] = {"returncode": result.returncode, "duration_seconds": result.duration_seconds}
        manifest["status"] = "pass" if result.returncode == 0 else "fail"

        if experiment["type"] == "prompt_library":
            prompt_path = run_dir / "prompt_set.json"
            if (run_dir / "eval_summary" ).exists():
                pass
            if DEFAULT_PROMPTS.exists():
                prompt_path = DEFAULT_PROMPTS
            count = 0
            if prompt_path.exists():
                prompts = load_json(prompt_path, default=[])
                if isinstance(prompts, list):
                    count = len(prompts)
                metrics["prompt_count"] = count
                metrics["prompt_set_hash"] = sha256_file(prompt_path)

        elif experiment["type"] == "evaluation":
            eval_metrics_path = run_dir / "eval_metrics.json"
            if eval_metrics_path.exists():
                report = load_json(eval_metrics_path, default={})
                metrics["summary"] = report.get("summary", {})
                summary = metrics["summary"]
                gate_results = {
                    "exact_line_match_rate": summary.get("exact_line_match_rate"),
                    "slur_violation_rate": summary.get("slur_violation_rate"),
                    "prompt_leakage_rate": summary.get("prompt_leakage_rate"),
                    "incomplete_ending_rate": summary.get("incomplete_ending_rate"),
                    "high_copy_similarity_rate": summary.get("high_copy_similarity_rate"),
                    "exact_line_match_rate_pass": (summary.get("exact_line_match_rate", 0.0) >= 0.995),
                    "slur_violation_rate_pass": (summary.get("slur_violation_rate", 1.0) == 0.0),
                    "prompt_leakage_rate_pass": (summary.get("prompt_leakage_rate", 1.0) == 0.0),
                    "incomplete_ending_rate_pass": (summary.get("incomplete_ending_rate", 1.0) <= 0.001),
                    "high_copy_similarity_rate_pass": (summary.get("high_copy_similarity_rate", 1.0) == 0.0),
                    "top50_judge_usable_rate_pass": False,
                }
                metrics["rows"] = report.get("summary", {}).get("row_count")
            else:
                gate_results = {"summary_missing": True}

        elif experiment["type"] == "ranking":
            summary_path = run_dir / "rank_summary.json"
            summary = load_json(summary_path, default={})
            metrics["summary"] = summary
            if summary:
                top_rows = read_lines_from_jsonl(run_dir / "ranked_quality.jsonl", count=8)
                write_markdown(run_dir / "sample_outputs.md", [ "# Ranked sample", "", *top_rows])
                # ranking metrics are not directly gates for current promotion phase
                gate_results = {
                    "exact_line_match_rate": summary.get("structural_pass_rate", 0.0),
                    "slur_violation_rate": 0.0,
                    "prompt_leakage_rate": 0.0,
                    "incomplete_ending_rate": 0.0,
                    "high_copy_similarity_rate": 0.0,
                    "top50_judge_usable_rate": None,
                    "exact_line_match_rate_pass": True,
                    "slur_violation_rate_pass": True,
                    "prompt_leakage_rate_pass": True,
                    "incomplete_ending_rate_pass": True,
                    "high_copy_similarity_rate_pass": True,
                    "top50_judge_usable_rate_pass": False,
                }

        elif experiment["type"] == "judge":
            judge_summary = load_json(run_dir / "judge" / "quality_judge_metrics.json", default={})
            metrics["judge_summary"] = judge_summary
            if judge_summary:
                top50 = judge_summary.get("top50_judge_usable_rate", 0.0)
                metrics["top50_judge_usable_rate"] = top50
                incumbent = state.get("incumbent", {})
                incumbent_top50 = incumbent.get("gate_targets", {}).get("top50_judge_usable_rate", 0.0)
                gate_results = {
                    "top50_judge_usable_rate": top50,
                    "top50_judge_usable_rate_pass": top50 >= incumbent_top50,
                }
                if (run_dir / "judge" / "top_100_auto_judged.md").exists():
                    judge_samples = read_text_file(run_dir / "judge" / "top_100_auto_judged.md")
                    write_markdown(
                        run_dir / "sample_outputs.md",
                        [
                            "# Judge sample outputs",
                            "",
                            judge_samples[:4000],
                        ],
                    )

        elif experiment["type"] == "tests":
            metrics["pytest_passed"] = result.returncode == 0
            gate_results = {"tests_pass": result.returncode == 0}

        status = "success" if result.returncode == 0 else "failed"

    decision = (
        "continue_research"
        if status != "success"
        or not (all(v is not False for v in gate_results.values() if isinstance(v, bool) and v is not None))
        else (
            "promote_prompt_policy"
            if experiment["type"] in {"ranking", "judge"} and gate_results.get("top50_judge_usable_rate_pass")
            else "continue_research"
        )
    )
    if gate_results.get("top50_judge_usable_rate") is not None and \
            gate_results.get("top50_judge_usable_rate", 0.0) > state.get("incumbent", {}).get("gate_targets", {}).get(
            "top50_judge_usable_rate", 0.0
        ):
        decision = "promote_prompt_policy"

    manifest["status"] = status
    manifest["ended_at"] = now_iso()
    manifest["wall_seconds"] = metrics.get("command", {}).get("duration_seconds", 0.0)
    manifest["metrics"] = metrics
    manifest["gate_results"] = gate_results
    manifest["decision"] = decision
    manifest["finished"] = now_iso()

    write_json(run_dir / "run_manifest.json", manifest)
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, {"run_id": run_id, "metrics": metrics, "gate_results": gate_results, "decision": decision})
    write_markdown(
        run_dir / "decision.md",
        [f"# Decision for {run_id}", "", f"decision = {decision}", "", safe_dumps(manifest, indent=2)],
    )
    write_markdown(
        run_dir / "summary.md",
        [
            "# Run Summary",
            "",
            f"- experiment: {experiment.get('name')}",
            f"- decision: {decision}",
            f"- command status: {manifest['status']}",
            f"- returncode: {metrics.get('command', {}).get('returncode', 'n/a')}",
            "",
            "## Metrics",
            safe_dumps(metrics, indent=2),
            "",
            "## Gate Results",
            safe_dumps(gate_results, indent=2),
        ],
    )

    run_record = {
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "experiment_id": experiment["id"],
        "name": experiment.get("name"),
        "type": experiment["type"],
        "status": manifest["status"],
        "decision": decision,
        "wall_seconds": manifest.get("wall_seconds", 0.0),
        "command": manifest["command"],
        "candidate_id": experiment.get("candidate_id"),
        "candidate_name": experiment.get("candidate_name"),
        "candidate_path": experiment.get("input"),
        "summary": {
            "metrics": metrics.get("summary", {}),
            "judge_summary": metrics.get("judge_summary", {}),
            "top50_judge_usable_rate": metrics.get("top50_judge_usable_rate", None),
        },
        "gate_results": gate_results,
    }

    if experiment["type"] == "judge":
        advisor_result = run_advisor(state, run_dir, run_record)
        run_record["advisor"] = advisor_result
        metrics["advisor_status"] = advisor_result.get("status")

    return run_record


def load_or_create_state(force_init: bool = False) -> dict[str, Any]:
    if STATE_PATH.exists() and not force_init:
        state = load_json(STATE_PATH, default={}) or {}
    else:
        state = {}

    if not state:
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        state = {
            "version": "1.0",
            "name": "qwen3_rap_quality_autonomous_loop",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "status": "running",
            "sweep_cycle_counter": 0,
            "sweep_batch_size": DEFAULT_SWEEP_BATCH_SIZE,
            "sweep_profiles": [],
            "incumbent": build_incumbent_profile(),
            "legal_status": "experimental_only",
            "experiment_queue": build_default_queue(),
            "run_history": [],
            "state_fingerprint": {
                "bible_references": [str(p) for p in BIBLE_MARKER_FILES],
                "goal_reference": str(GOAL_PATH),
            },
            "post_sweep_round_counter": 0,
            "post_sweep_batch_size": POST_SWEEP_BATCH_SIZE,
            "post_sweep_processed_candidates": [],
            "post_sweep_profile_rounds": {},
            "post_sweep_profiles": [],
            "next_resume_command": (
                f'python {ROOT / "scripts/run_autonomous_quality_loop.py"} '
                f'--hours 4 --resume --state {STATE_PATH}'
            ),
        }
        state["sweep_profiles"] = [
            {
                **candidate,
                "id": f"profile_{candidate['slug']}_v1",
                "status": "pending",
                "cycle": None,
                "last_run": None,
                "best_top50_judge_usable_rate": None,
            }
            for candidate in discover_candidate_sweep_profiles()
        ]
    else:
        if "sweep_profiles" not in state:
            state["sweep_profiles"] = [
                {
                    **candidate,
                    "id": f"profile_{candidate['slug']}_v1",
                    "status": "pending",
                    "cycle": None,
                    "last_run": None,
                    "best_top50_judge_usable_rate": None,
                }
                for candidate in discover_candidate_sweep_profiles()
            ]
        if "sweep_cycle_counter" not in state:
            state["sweep_cycle_counter"] = 0
        if "sweep_batch_size" not in state:
            state["sweep_batch_size"] = DEFAULT_SWEEP_BATCH_SIZE
        state.setdefault("post_sweep_round_counter", 0)
        state.setdefault("post_sweep_batch_size", POST_SWEEP_BATCH_SIZE)
        state.setdefault("post_sweep_processed_candidates", [])
        state.setdefault("post_sweep_profile_rounds", {})
        state.setdefault("post_sweep_profiles", [])
        if not state.get("post_sweep_profile_rounds"):
            round_progress = {}
            for profile in state.get("sweep_profiles", []):
                if profile.get("status") == "complete":
                    profile_id = str(profile.get("id"))
                    if profile_id in state.get("post_sweep_processed_candidates", []):
                        round_progress[profile_id] = 1
            if round_progress:
                state["post_sweep_profile_rounds"] = round_progress
    return state


def pick_next_experiment(state: dict[str, Any]) -> dict[str, Any] | None:
    for exp in state.get("experiment_queue", []):
        if exp.get("status") in {None, "pending", "retry"}:
            return exp
    return None


def enqueue_next_sweep_profiles(state: dict[str, Any], *, batch_size: int) -> int:
    """Append the next unresolved sweep profiles to the queue and return how many were added."""
    if batch_size <= 0:
        return 0

    known_status = {entry["sweep_path"]: entry["status"] for entry in state.get("sweep_profiles", [])}
    candidate_profiles = discover_candidate_sweep_profiles()
    available = [profile for profile in candidate_profiles if known_status.get(profile["sweep_path"]) in {None, "pending"}]
    if not available:
        return 0

    state["sweep_cycle_counter"] = int(state.get("sweep_cycle_counter", 0)) + 1
    cycle_id = state["sweep_cycle_counter"]
    selected = available[:batch_size]
    appended = 0
    for index, candidate in enumerate(selected, start=1):
        profile_id = f"cycle{cycle_id}_{candidate['slug']}_{index}"
        candidate["status"] = "queued"
        candidate["id"] = profile_id
        candidate["cycle"] = cycle_id
        candidate["last_run"] = None
        candidate["best_top50_judge_usable_rate"] = None

        for current in state.get("sweep_profiles", []):
            if current["sweep_path"] == candidate["sweep_path"]:
                current["id"] = profile_id
                current["status"] = "queued"
                current["cycle"] = cycle_id
                current["last_run"] = None
                current["best_top50_judge_usable_rate"] = None
                current["display_name"] = candidate["display_name"]
                break
        else:
            state["sweep_profiles"].append(candidate)

        for profile_exp in build_sweep_profile_experiments(
            profile_id=profile_id,
            cycle=cycle_id,
            profile=candidate,
            judge_seed=20260706 + (cycle_id * 100) + index,
        ):
            state["experiment_queue"].append(profile_exp)
            appended += 1

    return appended


def update_state_after_run(state: dict[str, Any], run_record: dict[str, Any], experiment_id: str, result: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    state["run_history"].append(run_record)
    for exp in state.get("experiment_queue", []):
        if exp["id"] == experiment_id:
            exp["attempts"] = int(exp.get("attempts", 0)) + 1
            exp["status"] = "done" if run_record["status"] == "success" else "failed"
            exp["last_run"] = run_record["run_id"]
            exp["last_summary"] = result.get("summary")
            break

    candidate_id = run_record.get("candidate_id")
    if candidate_id:
        profile = None
        for existing in state.get("sweep_profiles", []):
            if existing.get("id") == candidate_id:
                profile = existing
                break
        if profile is None:
            profile = {
                "id": candidate_id,
                "display_name": run_record.get("candidate_name") or candidate_id,
                "sweep_path": run_record.get("candidate_path"),
                "status": "queued",
                "cycle": None,
                "last_run": None,
                "best_top50_judge_usable_rate": None,
            }
            state["sweep_profiles"].append(profile)

        profile["last_run"] = run_record["run_id"]
        experiment_type = run_record.get("type")
        if run_record["status"] != "success":
            profile["status"] = "failed"
        elif experiment_type == "evaluation":
            profile["status"] = "evaluated"
        elif experiment_type == "ranking":
            profile["status"] = "ranked"
        elif experiment_type == "judge":
            top50 = run_record.get("summary", {}).get("top50_judge_usable_rate")
            profile["status"] = "complete"
            profile["best_top50_judge_usable_rate"] = top50

    if run_record.get("decision", "").startswith("promote_"):
        state["status"] = "promotion_ready"
        state["incumbent"] = {
            **state.get("incumbent", {}),
            "name": run_record.get("name", state.get("incumbent", {}).get("name", "")),
            "last_promotion_decision": run_record.get("decision"),
            "promoted_at": now_iso(),
        }
    update_experiment_queue_markdown(state)
    update_validation_matrix(state)
    update_artifact_manifest(state)
    update_sweep_profile_log(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an autonomous rap quality improvement loop.")
    parser.add_argument("--hours", type=float, default=1.0, help="Loop budget in hours.")
    parser.add_argument("--state", type=Path, default=STATE_PATH, help="Path to loop state.json.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state file.")
    parser.add_argument("--force-init", action="store_true", help="Reinitialize loop state and docs.")
    parser.add_argument("--run-everything", action="store_true", help="Run the full initial queue immediately.")
    parser.add_argument(
        "--auto-refill",
        action="store_true",
        help="When queue is empty, auto-enqueue alternate sweep profiles.",
    )
    parser.add_argument(
        "--sweep-batch",
        type=int,
        default=DEFAULT_SWEEP_BATCH_SIZE,
        help="How many new sweep profiles to enqueue each refill.",
    )
    return parser.parse_args()


def build_resume_command(hours: float, state_path: Path, *, auto_refill: bool, sweep_batch: int) -> str:
    command = f'python "{ROOT / "scripts" / "run_autonomous_quality_loop.py"} --hours {hours} --resume --state "{state_path}"'
    if auto_refill:
        command += " --auto-refill --sweep-batch " + str(sweep_batch)
    return command

def main() -> int:
    args = parse_args()
    global STATE_PATH
    STATE_PATH = args.state
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    state = load_or_create_state(force_init=args.force_init)
    state["sweep_batch_size"] = args.sweep_batch
    state["next_resume_command"] = build_resume_command(
        args.hours,
        STATE_PATH,
        auto_refill=args.auto_refill,
        sweep_batch=args.sweep_batch,
    )
    write_docs_scaffold(state)

    if args.force_init and args.state:
        STATE_PATH = args.state
    if not args.state.parent.exists():
        args.state.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + max(0.0, args.hours) * 3600.0
    ran_any = False
    while time.time() < deadline:
        experiment = pick_next_experiment(state)
        if experiment is None:
            if args.auto_refill:
                enqueued = enqueue_next_sweep_profiles(
                    state,
                    batch_size=args.sweep_batch,
                )
                if enqueued > 0:
                    update_experiment_queue_markdown(state)
                    update_sweep_profile_log(state)
                    continue
                enqueued_post = enqueue_post_sweep_experiments(
                    state,
                    batch_size=args.sweep_batch,
                )
                if enqueued_post > 0:
                    update_experiment_queue_markdown(state)
                    update_sweep_profile_log(state)
                    update_artifact_manifest(state)
                    continue
            state["status"] = "complete"
            break

        run_dir = RUNS_ROOT / f"{timestamp()}_{experiment['slug']}"
        run_record = run_single_experiment(state, experiment, run_dir)
        ran_any = True

        update_state_after_run(state, run_record, experiment["id"], run_record)
        write_json(STATE_PATH, state)
        update_iteration_log(state, run_record)
        update_decision_log(state, run_record)
        refresh_labeled_artifact_mirror(state, run_dir=run_dir, run_record=run_record)

        if run_record["decision"].startswith("promote_"):
            state["status"] = "promotion_ready"
            write_json(STATE_PATH, state)
            break

        if not args.run_everything and run_record["status"] != "success":
            # stop on first concrete failure unless explicitly forced.
            break

    if not ran_any and state.get("status") != "complete":
        state["status"] = "idle"
    state["updated_at"] = now_iso()
    state["next_resume_command"] = build_resume_command(
        args.hours,
        STATE_PATH,
        auto_refill=args.auto_refill,
        sweep_batch=args.sweep_batch,
    )
    update_experiment_queue_markdown(state)
    update_artifact_manifest(state)
    update_sweep_profile_log(state)
    refresh_labeled_artifact_mirror(state)
    write_json(STATE_PATH, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
