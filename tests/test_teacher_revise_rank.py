from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.run_teacher_revise_rank import (
    ApiBudget,
    cmd_critique,
    cmd_generate,
    cmd_judge,
    cmd_report,
    cmd_revise,
    ensure_manifest,
    jaccard_ngrams,
    load_prompts,
    resolve_returned_id,
    strict_pass,
    structural_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        prompts=ROOT / "configs/prompts/qwen3_4b_12line_targeted_stage2_prompts.json",
        baseline=ROOT / "data/sweeps/qwen3_4b_12line_targeted_stage2_smoke96/sweep_raw.jsonl",
        output_dir=tmp_path,
        teacher_model="teacher-test",
        critic_model="critic-test",
        judge_model="judge-test",
        candidates_per_prompt=4,
        discovery_themes=1,
        families=["technical", "clean"],
        judge_batch_size=6,
        seed=20260712,
        max_api_calls=80,
        max_total_tokens=500000,
        timeout=5,
        retries=0,
        mock=True,
    )


def test_prompt_split_and_manifest_are_deterministic(tmp_path: Path) -> None:
    options = args(tmp_path)
    prompts = load_prompts(options.prompts, 8)
    assert len(prompts) == 16
    first = ensure_manifest(options, prompts)
    assert ensure_manifest(options, prompts) == first


def test_structural_metrics_and_strict_family_rules() -> None:
    lyrics = "\n".join(
        [
            "Silver rain taps softly on the rusted fire escape",
            "A night bus exhales while the corner stores awake",
            "I fold the old map where the river splits the town",
            "Neon swims in puddles as the traffic settles down",
            "A locksmith turns his sign beneath a tired amber bulb",
            "The bakery sends warm air through the avenues of cold",
            "My shadow keeps its distance by the boarded picture show",
            "A train writes sparks above me, then disappears below",
            "I carry one small promise in the pocket of my coat",
            "Past windows full of strangers and a ferry's distant note",
            "At sunrise every rooftop catches fire without a flame",
            "I cross the bridge still moving, but I leave with my own name",
        ]
    )
    metrics = structural_metrics(lyrics)
    assert metrics["line_count"] == 12
    row = {
        "family": "clean",
        "lyrics": lyrics,
        "scores": {
            "overall": 5,
            "technical_rhyme": 4,
            "flow_cadence": 4,
            "coherence": 4,
            "thematic_depth": 4,
            "imagery": 4,
            "ending_strength": 4,
            "family_compliance": 5,
            "cleanliness": 5,
            "genericness": 1,
        },
        "critical_failure_flags": [],
    }
    passed, failures = strict_pass(row)
    assert passed, failures
    row["scores"]["genericness"] = 3
    assert not strict_pass(row)[0]


def test_diversity_similarity_detects_identical_text() -> None:
    assert jaccard_ngrams("one two three four five six", "one two three four five six") == 1.0


def test_returned_id_repairs_one_character_only() -> None:
    assert resolve_returned_id("abcdef1234567890abce", ["abcdef1234567890abcd"]) == ("abcdef1234567890abcd", True)


def test_api_budget_enforces_call_limit(tmp_path: Path) -> None:
    budget = ApiBudget(tmp_path, max_calls=1, max_total_tokens=100)
    budget.record({"usage": {"total_tokens": 5}})
    try:
        budget.reserve()
    except RuntimeError as exc:
        assert "call budget" in str(exc)
    else:
        raise AssertionError("budget should fail")


def test_mock_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    options = args(tmp_path)
    cmd_generate(options)
    cmd_critique(options)
    cmd_revise(options)
    cmd_judge(options)
    cmd_report(options)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["groups"]["technical"]["teacher_generate"]["rows"] == 4
    assert report["groups"]["clean"]["independent_critique_revision"]["rows"] == 4
    assert (tmp_path / "blind_map.json").exists()
