"""Run the locked two-stage Qwen3-4B quality-goal evaluation matrix.

Development compares base/E1/E2/E3 and intentionally stops without selecting a
winner. Confirmation compares base with an explicitly supplied development
winner. Generation is always fresh, raw, sequential, and seed-identical.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/evaluation/quality_goal_generation_v1.json")
SWEEP_SCRIPT = Path("scripts/run_qwen3_generation_sweep.py")
PINNED_BASE_MODEL = "Qwen/Qwen3-4B"
PINNED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PINNED_SEED = 20260710
ADAPTER_LABELS = ("e1", "e2", "e3")
STAGES = ("development", "confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="development")
    parser.add_argument("--winner", choices=ADAPTER_LABELS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/qwen3_4b_12line_quality_goal_eval_v1"),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and validate the plan without requiring adapters, CUDA, or training data.",
    )
    return parser.parse_args()


def absolute(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def read_json(path: Path | str) -> dict[str, Any]:
    source = absolute(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {source}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    target = absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    source = absolute(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_stage(stage: str, winner: str | None) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "development":
        if winner is not None:
            raise ValueError("--winner is only valid for the confirmation stage")
        return ["base", *ADAPTER_LABELS]
    if winner not in ADAPTER_LABELS:
        raise ValueError("Confirmation requires the automated development winner e1, e2, or e3")
    return ["base", winner]


def resolve_stage_winner(stage: str, winner: str | None, run_dir: Path) -> str | None:
    if stage != "confirmation" or winner is not None:
        return winner
    summary_path = absolute(run_dir / "development" / "automated_judge" / "summary.json")
    if not summary_path.is_file():
        raise ValueError(
            "Confirmation needs a completed automated development judge summary or an explicit --winner."
        )
    selected = str(read_json(summary_path).get("selected_adapter") or "")
    if selected not in ADAPTER_LABELS:
        raise ValueError("Automated development summary has no valid selected_adapter")
    return selected


def load_and_validate_prompts(stage: str, stage_config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    prompt_path = Path(str(stage_config.get("prompt_file") or ""))
    source = absolute(prompt_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prompt file must contain a JSON list: {source}")
    expected_count = int(stage_config.get("expected_prompt_count") or 0)
    samples = int(stage_config.get("samples_per_prompt") or 0)
    if len(payload) != expected_count:
        raise ValueError(f"{stage}: expected {expected_count} prompts, found {len(payload)}")
    if samples != 2:
        raise ValueError(f"{stage}: samples_per_prompt must be exactly 2")

    keys: set[str] = set()
    prompts: set[str] = set()
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: prompt {index} must be an object")
        prompt_key = str(row.get("prompt_key") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        if not prompt_key or not prompt:
            raise ValueError(f"{source}: prompt {index} is missing prompt_key or prompt")
        if prompt_key in keys or prompt in prompts:
            raise ValueError(f"{source}: prompt keys and prompt text must be unique")
        if row.get("evaluation_split") != stage:
            raise ValueError(f"{source}: prompt {prompt_key} has the wrong evaluation_split")
        if int(row.get("target_line_count") or 0) != 12:
            raise ValueError(f"{source}: prompt {prompt_key} must target exactly 12 lines")
        if int(row.get("samples_per_model") or 0) != samples:
            raise ValueError(f"{source}: prompt {prompt_key} must declare samples_per_model={samples}")
        keys.add(prompt_key)
        prompts.add(prompt)
    return payload, sha256_file(prompt_path)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("base_model") != PINNED_BASE_MODEL:
        raise ValueError(f"base_model must be locked to {PINNED_BASE_MODEL}")
    if config.get("model_revision") != PINNED_MODEL_REVISION:
        raise ValueError(f"model_revision must be locked to {PINNED_MODEL_REVISION}")
    if int(config.get("seed") or -1) != PINNED_SEED:
        raise ValueError(f"seed must be locked to {PINNED_SEED}")

    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    locked_values = {
        "batch_size": 1,
        "strict_row_seeds": True,
        "block_slurs": False,
        "enforce_target_line_count": False,
        "underlength_retries": 0,
        "resume": False,
    }
    for key, expected in locked_values.items():
        if generation.get(key) != expected:
            raise ValueError(f"generation.{key} must be {expected!r}")
    for required in (
        "max_input_tokens",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "load_in_4bit",
        "disable_thinking",
    ):
        if required not in generation:
            raise ValueError(f"generation.{required} is required")

    models = config.get("models")
    if not isinstance(models, dict) or set(models) != {"base", *ADAPTER_LABELS}:
        raise ValueError("models must define exactly base, e1, e2, and e3")
    if models["base"].get("adapter") is not False:
        raise ValueError("models.base must disable adapters")
    for expected_epoch, label in enumerate(ADAPTER_LABELS, start=1):
        model = models[label]
        if model.get("adapter") is not True or not model.get("adapter_dir") or not model.get("training_config"):
            raise ValueError(f"models.{label} must define an adapter directory and training config")
        training_config = read_json(model["training_config"])
        if training_config.get("base_model") != PINNED_BASE_MODEL:
            raise ValueError(f"models.{label} training config has the wrong base model")
        if training_config.get("model_revision") != PINNED_MODEL_REVISION:
            raise ValueError(f"models.{label} training config has the wrong model revision")
        if Path(str(training_config.get("output_dir"))) != Path(str(model["adapter_dir"])):
            raise ValueError(f"models.{label} adapter_dir must match its training config output_dir")
        epochs = float((training_config.get("training") or {}).get("num_train_epochs") or 0)
        if epochs != float(expected_epoch):
            raise ValueError(f"models.{label} must reference the {expected_epoch}-epoch training config")

    stages = config.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ValueError("stages must define exactly development and confirmation")
    if stages["development"].get("models") != ["base", "e1", "e2", "e3"]:
        raise ValueError("Development must compare base, e1, e2, and e3")
    if stages["confirmation"].get("models") != ["base", "selected_development_winner"]:
        raise ValueError("Confirmation must compare base with an explicitly selected winner")
    for stage in STAGES:
        load_and_validate_prompts(stage, stages[stage])

    policy = config.get("selection_policy") or {}
    if policy.get("selection_mode") != "automated_blind_openai_judge":
        raise ValueError("selection_policy.selection_mode must use the automated blind judge")
    if policy.get("auto_select_winner") is not True:
        raise ValueError("selection_policy.auto_select_winner must be true")
    if policy.get("confirmation_requires_explicit_winner") is not False:
        raise ValueError("confirmation must consume the automated winner")
    if policy.get("allowed_winners") != list(ADAPTER_LABELS):
        raise ValueError("allowed_winners must be e1, e2, and e3")
    promotion = config.get("promotion") if isinstance(config.get("promotion"), dict) else {}
    locked_promotion = {
        "preference_wilson_95_lower_bound": 0.5,
        "maximum_exact_line_rate_drop": 0.02,
        "allow_new_slur_prompt_leak_or_high_copy_rows": False,
        "maximum_incomplete_ending_rate_increase": 0.01,
        "minimum_relative_target_issue_burden_improvement": 0.1,
        "maximum_individual_issue_rate_increase": 0.03,
    }
    for key, expected in locked_promotion.items():
        if promotion.get(key) != expected:
            raise ValueError(f"promotion.{key} must be locked to {expected!r}")
    automated_judge = config.get("automated_judge") if isinstance(config.get("automated_judge"), dict) else {}
    if not str(automated_judge.get("script") or "").strip():
        raise ValueError("automated_judge.script is required")
    if int(automated_judge.get("votes_per_comparison") or 0) < 3 or int(
        automated_judge.get("votes_per_comparison") or 0
    ) % 2 == 0:
        raise ValueError("automated_judge.votes_per_comparison must be an odd integer >= 3")
    if automated_judge.get("human_review_used") is not False:
        raise ValueError("automated_judge.human_review_used must be false")
    for key in ("gate_script",):
        if not str(promotion.get(key) or "").strip():
            raise ValueError(f"promotion.{key} is required")
    return config


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def generation_command(
    config: dict[str, Any],
    *,
    label: str,
    prompt_file: Path,
    num_candidates: int,
    output_jsonl: Path,
    summary_json: Path,
    run_manifest: Path,
) -> list[str]:
    generation = config["generation"]
    model = config["models"][label]
    command = [
        sys.executable,
        "-u",
        str(SWEEP_SCRIPT),
        "--base-model",
        str(config["base_model"]),
        "--model-revision",
        str(config["model_revision"]),
    ]
    if model["adapter"]:
        command.extend(["--adapter", "--adapter-dir", str(model["adapter_dir"])])
    else:
        command.append("--no-adapter")
    command.extend(
        [
            "--prompt-file",
            str(prompt_file),
            "--num-candidates",
            str(num_candidates),
            "--batch-size",
            str(generation["batch_size"]),
            "--max-input-tokens",
            str(generation["max_input_tokens"]),
            "--max-new-tokens",
            str(generation["max_new_tokens"]),
            "--temperature",
            str(generation["temperature"]),
            "--top-p",
            str(generation["top_p"]),
            "--top-k",
            str(generation["top_k"]),
            "--repetition-penalty",
            str(generation["repetition_penalty"]),
            "--no-repeat-ngram-size",
            str(generation["no_repeat_ngram_size"]),
            "--seed",
            str(config["seed"]),
            bool_flag("load-in-4bit", bool(generation["load_in_4bit"])),
            bool_flag("disable-thinking", bool(generation["disable_thinking"])),
            bool_flag("block-slurs", bool(generation["block_slurs"])),
            bool_flag("enforce-target-line-count", bool(generation["enforce_target_line_count"])),
            "--underlength-retries",
            str(generation["underlength_retries"]),
            bool_flag("strict-row-seeds", bool(generation["strict_row_seeds"])),
            "--no-resume",
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--run-manifest",
            str(run_manifest),
        ]
    )
    return command


def packet_command(
    config: dict[str, Any],
    *,
    labels: list[str],
    generation_paths: dict[str, Path],
    prompt_file: Path,
    output_dir: Path,
) -> list[str]:
    packet = config["evaluation_packet"]
    command = [sys.executable, "-u", str(packet["script"])]
    for label in labels:
        command.extend(["--generation", f"{label}={generation_paths[label]}"])
    command.extend(["--prompt-file", str(prompt_file)])
    for train_jsonl in packet.get("training_jsonl") or []:
        command.extend(["--train-jsonl", str(train_jsonl)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--seed",
            str(config["seed"]),
            "--ngram-size",
            str(packet["ngram_size"]),
            "--similarity-threshold",
            str(packet["similarity_threshold"]),
            "--max-train-records",
            str(packet["max_train_records"]),
        ]
    )
    return command


def automated_judge_command(config: dict[str, Any], *, packet_dir: Path, output_dir: Path) -> list[str]:
    judge = config["automated_judge"]
    return [
        sys.executable,
        "-u",
        str(judge["script"]),
        "--packet",
        str(packet_dir / "comparison_packet.json"),
        "--private-key",
        str(packet_dir / "comparison_alias_key.private.json"),
        "--output-dir",
        str(output_dir),
        "--model",
        str(judge.get("default_model") or "gpt-4.1-mini"),
        "--votes-per-comparison",
        str(judge["votes_per_comparison"]),
        "--temperature",
        str(judge.get("temperature", 0.0)),
        "--max-retries",
        str(judge.get("max_retries", 3)),
    ]


def promotion_command(
    config: dict[str, Any],
    *,
    packet_dir: Path,
    judge_dir: Path,
    winner: str,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(config["promotion"]["gate_script"]),
        "--base-scored",
        str(packet_dir / "raw_scored" / "base.jsonl"),
        "--candidate-scored",
        str(packet_dir / "raw_scored" / f"{winner}.jsonl"),
        "--preferences",
        str(judge_dir / "resolved_preferences.json"),
        "--base-label",
        "base",
        "--candidate-label",
        winner,
        "--out",
        str(output_path),
    ]


def build_plan(
    config_path: Path,
    config: dict[str, Any],
    *,
    stage: str,
    winner: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    labels = validate_stage(stage, winner)
    stage_config = config["stages"][stage]
    prompts, prompt_hash = load_and_validate_prompts(stage, stage_config)
    prompt_file = Path(stage_config["prompt_file"])
    num_candidates = len(prompts) * int(stage_config["samples_per_prompt"])
    stage_dir = run_dir / stage
    generation_paths: dict[str, Path] = {}
    runs: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "evaluation_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "prompt_file": {"path": str(prompt_file), "sha256": prompt_hash},
        "training_configs": {},
        "adapter_artifacts": {},
        "training_jsonl": {},
    }
    for label in labels:
        model = config["models"][label]
        model_dir = stage_dir / "generations" / label
        output_jsonl = model_dir / "raw.jsonl"
        summary_json = model_dir / "summary.json"
        # Keep the sweep's conventional sidecar name so the packet builder can
        # independently discover and hash it from raw.jsonl.
        run_manifest = model_dir / "raw.run_manifest.json"
        generation_paths[label] = output_jsonl
        command = generation_command(
            config,
            label=label,
            prompt_file=prompt_file,
            num_candidates=num_candidates,
            output_jsonl=output_jsonl,
            summary_json=summary_json,
            run_manifest=run_manifest,
        )
        runs.append(
            {
                "label": label,
                "adapter": bool(model["adapter"]),
                "adapter_dir": model.get("adapter_dir"),
                "expected_rows": num_candidates,
                "output_jsonl": str(output_jsonl),
                "summary_json": str(summary_json),
                "run_manifest": str(run_manifest),
                "command": command,
            }
        )
        if model["adapter"]:
            training_config = Path(model["training_config"])
            hashes["training_configs"][label] = {
                "path": str(training_config),
                "sha256": sha256_file(training_config),
            }

    for path_text in config["evaluation_packet"].get("training_jsonl") or []:
        path = Path(path_text)
        hashes["training_jsonl"][str(path)] = sha256_file(path) if absolute(path).exists() else None

    packet_dir = stage_dir / "eval_packet"
    packet = packet_command(
        config,
        labels=labels,
        generation_paths=generation_paths,
        prompt_file=prompt_file,
        output_dir=packet_dir,
    )
    judge_dir = stage_dir / "automated_judge"
    judge = automated_judge_command(config, packet_dir=packet_dir, output_dir=judge_dir)
    promotion = (
        promotion_command(
            config,
            packet_dir=packet_dir,
            judge_dir=judge_dir,
            winner=str(winner),
            output_path=stage_dir / "promotion_result.json",
        )
        if stage == "confirmation" and winner
        else None
    )
    return {
        "schema_version": 1,
        "status": "prepared",
        "created_at": timestamp(),
        "stage": stage,
        "selected_development_winner": winner,
        "winner_selection": "automated_consensus" if stage == "development" else "automated_development_winner",
        "base_model": config["base_model"],
        "model_revision": config["model_revision"],
        "seed": config["seed"],
        "prompt_count": len(prompts),
        "samples_per_prompt": int(stage_config["samples_per_prompt"]),
        "expected_rows_per_model": num_candidates,
        "labels": labels,
        "generation_mode": {
            "raw": True,
            "fresh_outputs": True,
            **config["generation"],
        },
        "runs": runs,
        "packet": {"output_dir": str(packet_dir), "command": packet},
        "automated_judge": {"output_dir": str(judge_dir), "command": judge},
        "promotion": {"command": promotion, "output": str(stage_dir / "promotion_result.json")} if promotion else None,
        "input_hashes": hashes,
    }


def validate_adapter(label: str, config: dict[str, Any]) -> dict[str, Any]:
    model = config["models"][label]
    adapter_dir = absolute(model["adapter_dir"])
    required = {
        "training_summary": adapter_dir / "training_summary.json",
        "adapter_config": adapter_dir / "adapter_config.json",
        "adapter_weights": adapter_dir / "adapter_model.safetensors",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required completed adapter artifacts: {missing}")
    summary = read_json(required["training_summary"])
    result = summary.get("result") if isinstance(summary.get("result"), dict) else {}
    if result.get("status") != "complete_full_budget":
        raise ValueError(f"{label} training summary status must be complete_full_budget")
    if summary.get("base_model") != PINNED_BASE_MODEL:
        raise ValueError(f"{label} training summary has the wrong base model")
    if summary.get("model_revision") != PINNED_MODEL_REVISION:
        raise ValueError(f"{label} training summary has the wrong model revision")
    return {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for name, path in required.items()
    }


def validate_runtime_inputs(plan: dict[str, Any], config: dict[str, Any]) -> None:
    for label in plan["labels"]:
        if label != "base":
            plan["input_hashes"]["adapter_artifacts"][label] = validate_adapter(label, config)
    packet_script = absolute(config["evaluation_packet"]["script"])
    if not packet_script.is_file():
        raise FileNotFoundError(f"Evaluation packet builder not found: {packet_script}")
    for script_path_text in (config["automated_judge"]["script"], config["promotion"]["gate_script"]):
        script_path = absolute(script_path_text)
        if not script_path.is_file():
            raise FileNotFoundError(f"Automated evaluation workflow script not found: {script_path}")
    for path_text in config["evaluation_packet"].get("training_jsonl") or []:
        path = absolute(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation training JSONL not found: {path}")
        plan["input_hashes"]["training_jsonl"][str(Path(path_text))] = sha256_file(path)


def assert_fresh_outputs(plan: dict[str, Any]) -> None:
    existing: list[str] = []
    for run in plan["runs"]:
        for key in ("output_jsonl", "summary_json", "run_manifest"):
            if absolute(run[key]).exists():
                existing.append(str(absolute(run[key])))
    packet_dir = absolute(plan["packet"]["output_dir"])
    if packet_dir.exists():
        existing.append(str(packet_dir))
    judge_dir = absolute(plan["automated_judge"]["output_dir"])
    if judge_dir.exists():
        existing.append(str(judge_dir))
    if plan.get("promotion") and absolute(plan["promotion"]["output"]).exists():
        existing.append(str(absolute(plan["promotion"]["output"])))
    if existing:
        raise FileExistsError(
            "Fresh evaluation outputs are required; choose a new --run-dir or archive these paths: "
            + ", ".join(existing)
        )


def run_logged(command: list[str], log_dir: Path) -> dict[str, Any]:
    target = absolute(log_dir)
    target.mkdir(parents=True, exist_ok=True)
    stdout_path = target / "stdout.txt"
    stderr_path = target / "stderr.txt"
    started = dt.datetime.now(dt.timezone.utc)
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "w", encoding="utf-8", buffering=1
    ) as stderr:
        process = subprocess.run(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        "command": command,
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": round((ended - started).total_seconds(), 2),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def count_jsonl(path: Path | str) -> int:
    source = absolute(path)
    with source.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_generation_result(run: dict[str, Any]) -> dict[str, Any]:
    output_path = absolute(run["output_jsonl"])
    summary_path = absolute(run["summary_json"])
    manifest_path = absolute(run["run_manifest"])
    if not output_path.is_file() or not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"{run['label']} did not write all required generation outputs")
    summary = read_json(summary_path)
    expected = int(run["expected_rows"])
    rows = count_jsonl(output_path)
    if summary.get("status") != "complete" or int(summary.get("unique_output_rows") or -1) != expected:
        raise RuntimeError(f"{run['label']} generation summary is incomplete")
    if rows != expected:
        raise RuntimeError(f"{run['label']} expected {expected} JSONL rows, found {rows}")
    return {
        "output_jsonl_sha256": sha256_file(output_path),
        "summary_json_sha256": sha256_file(summary_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    config = validate_config(read_json(args.config))
    winner = resolve_stage_winner(args.stage, args.winner, args.run_dir)
    plan = build_plan(
        args.config,
        config,
        stage=args.stage,
        winner=winner,
        run_dir=args.run_dir,
    )
    stage_dir = args.run_dir / args.stage
    write_json(stage_dir / "run_plan.json", plan)
    write_json(
        stage_dir / "commands.json",
        {
            "generation": [run["command"] for run in plan["runs"]],
            "evaluation_packet": plan["packet"]["command"],
            "automated_judge": plan["automated_judge"]["command"],
            "promotion": (plan.get("promotion") or {}).get("command"),
        },
    )
    print(json.dumps(plan, indent=2))
    if args.prepare_only:
        return 0

    try:
        validate_runtime_inputs(plan, config)
        assert_fresh_outputs(plan)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        result = {"status": "preflight_failed", "error": str(exc), "stage": args.stage}
        write_json(stage_dir / "final_result.json", result)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 2
    write_json(stage_dir / "run_plan.json", plan)

    attempts: list[dict[str, Any]] = []
    output_hashes: dict[str, Any] = {}
    for run in plan["runs"]:
        attempt = {"kind": "generation", "label": run["label"], **run_logged(run["command"], stage_dir / "logs" / run["label"])}
        attempts.append(attempt)
        write_json(stage_dir / "attempts.json", {"attempts": attempts})
        if attempt["returncode"] != 0:
            final = {"status": "generation_failed", "stage": args.stage, "attempts": attempts}
            write_json(stage_dir / "final_result.json", final)
            return int(attempt["returncode"] or 1)
        try:
            output_hashes[run["label"]] = validate_generation_result(run)
        except RuntimeError as exc:
            final = {"status": "generation_validation_failed", "error": str(exc), "attempts": attempts}
            write_json(stage_dir / "final_result.json", final)
            return 2

    packet_attempt = {
        "kind": "evaluation_packet",
        **run_logged(plan["packet"]["command"], stage_dir / "logs" / "evaluation_packet"),
    }
    attempts.append(packet_attempt)
    write_json(stage_dir / "attempts.json", {"attempts": attempts})
    if packet_attempt["returncode"] != 0:
        final = {"status": "evaluation_packet_failed", "stage": args.stage, "attempts": attempts}
        write_json(stage_dir / "final_result.json", final)
        return int(packet_attempt["returncode"] or 1)

    packet_manifest = absolute(plan["packet"]["output_dir"]) / "evaluation_manifest.json"
    if not packet_manifest.is_file():
        final = {"status": "evaluation_packet_validation_failed", "missing": str(packet_manifest)}
        write_json(stage_dir / "final_result.json", final)
        return 2
    output_hashes["evaluation_manifest"] = sha256_file(packet_manifest)
    judge_attempt = {
        "kind": "automated_judge",
        **run_logged(plan["automated_judge"]["command"], stage_dir / "logs" / "automated_judge"),
    }
    attempts.append(judge_attempt)
    write_json(stage_dir / "attempts.json", {"attempts": attempts})
    if judge_attempt["returncode"] != 0:
        final = {"status": "automated_judge_failed", "stage": args.stage, "attempts": attempts}
        write_json(stage_dir / "final_result.json", final)
        return int(judge_attempt["returncode"] or 1)
    judge_summary_path = absolute(plan["automated_judge"]["output_dir"]) / "summary.json"
    preferences_path = absolute(plan["automated_judge"]["output_dir"]) / "resolved_preferences.json"
    if not judge_summary_path.is_file() or not preferences_path.is_file():
        final = {"status": "automated_judge_validation_failed", "stage": args.stage}
        write_json(stage_dir / "final_result.json", final)
        return 2
    judge_summary = read_json(judge_summary_path)
    output_hashes["automated_judge_summary"] = sha256_file(judge_summary_path)
    output_hashes["automated_preferences"] = sha256_file(preferences_path)

    promotion_report = None
    if args.stage == "confirmation":
        promotion_attempt = {
            "kind": "promotion_gate",
            **run_logged(plan["promotion"]["command"], stage_dir / "logs" / "promotion_gate"),
        }
        attempts.append(promotion_attempt)
        write_json(stage_dir / "attempts.json", {"attempts": attempts})
        promotion_path = absolute(plan["promotion"]["output"])
        if not promotion_path.is_file():
            final = {"status": "promotion_evidence_missing", "stage": args.stage, "attempts": attempts}
            write_json(stage_dir / "final_result.json", final)
            return 2
        promotion_report = read_json(promotion_path)
        output_hashes["promotion_result"] = sha256_file(promotion_path)
        status = "promotion_passed" if promotion_report.get("passed") else "promotion_failed"
    else:
        selected = str(judge_summary.get("selected_adapter") or "")
        if selected not in ADAPTER_LABELS:
            final = {"status": "automated_winner_selection_failed", "judge_summary": judge_summary}
            write_json(stage_dir / "final_result.json", final)
            return 2
        status = "automated_winner_selected"
    final = {
        "status": status,
        "stage": args.stage,
        "selected_development_winner": winner if args.stage == "confirmation" else judge_summary.get("selected_adapter"),
        "auto_selected_winner": judge_summary.get("selected_adapter"),
        "human_review_used": False,
        "judge_summary": judge_summary,
        "promotion": promotion_report,
        "output_hashes": output_hashes,
        "evaluation_packet": str(plan["packet"]["output_dir"]),
        "attempts": attempts,
    }
    if args.stage == "development":
        final["next_step"] = (
            "Run confirmation; the orchestrator will consume this automated development winner."
        )
    else:
        final["next_step"] = (
            "Promote only when status is promotion_passed; otherwise retain the base model."
        )
    write_json(stage_dir / "final_result.json", final)
    print(json.dumps(final, indent=2))
    return 0 if status != "promotion_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
