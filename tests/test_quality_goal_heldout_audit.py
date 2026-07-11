from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_heldout_prompts.py"
DEV = ROOT / "configs" / "prompts" / "quality_goal_dev_prompts.json"
CONFIRMATION = ROOT / "configs" / "prompts" / "quality_goal_confirmation_prompts.json"


def run_audit(tmp_path: Path, training_rows: list[dict]) -> tuple[subprocess.CompletedProcess[str], dict]:
    training = tmp_path / "training.jsonl"
    output = tmp_path / "audit.json"
    training.write_text("\n".join(json.dumps(row) for row in training_rows) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt-file",
            str(DEV),
            "--prompt-file",
            str(CONFIRMATION),
            "--training-jsonl",
            str(training),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def test_locked_prompt_banks_have_valid_schema_counts_and_hashes(tmp_path: Path) -> None:
    result, report = run_audit(tmp_path, [])

    assert result.returncode == 0, result.stderr
    assert report["status"] == "pass"
    assert report["evaluation_split_counts"] == {"development": 48, "confirmation": 24}
    assert report["schema_error_count"] == 0
    assert all(item["sha256"] for item in report["prompt_inputs"])


def test_heldout_audit_rejects_prompt_and_theme_overlap(tmp_path: Path) -> None:
    prompt = json.loads(DEV.read_text(encoding="utf-8"))[0]
    training_row = {
        "prompt": prompt["prompt"],
        "metadata": {"theme": prompt["theme"]},
    }

    result, report = run_audit(tmp_path, [training_row])

    assert result.returncode != 0
    assert report["normalized_prompt_overlap_count"] == 1
    assert report["theme_overlap_count"] >= 1
