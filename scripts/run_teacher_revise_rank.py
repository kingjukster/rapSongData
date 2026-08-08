#!/usr/bin/env python3
"""Run a resumable teacher -> critic -> revise -> blind-judge lyric experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_PROMPTS = Path("configs/prompts/qwen3_4b_12line_targeted_stage2_prompts.json")
DEFAULT_BASELINE = Path("data/sweeps/qwen3_4b_12line_targeted_stage2_smoke96/sweep_raw.jsonl")
DEFAULT_OUTPUT = Path("runs/qwen3_4b_teacher_revise_v4/phase1")
DIMENSIONS = (
    "overall",
    "technical_rhyme",
    "flow_cadence",
    "coherence",
    "thematic_depth",
    "imagery",
    "ending_strength",
    "family_compliance",
    "cleanliness",
    "genericness",
)
POSITIVE_DIMENSIONS = (
    "overall",
    "technical_rhyme",
    "flow_cadence",
    "coherence",
    "thematic_depth",
    "imagery",
    "ending_strength",
)
CRITICAL_FLAGS = {
    "off_prompt",
    "scene_drift",
    "forced_syntax",
    "filler_bar",
    "unfinished_ending",
    "unsafe_content",
    "prompt_leakage",
    "repetition_loop",
    "copy_risk",
}
BLOCKED_CLEAN_TERMS = {
    "fuck", "fucking", "shit", "bitch", "cunt", "nigga", "nigger", "faggot",
    "cock", "pussy", "motherfucker",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?")


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv as _load

        _load(path)
        return
    except Exception:
        pass
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


def resolve_returned_id(returned: str, expected: Iterable[str]) -> tuple[str, bool]:
    expected_ids = list(expected)
    if returned in expected_ids:
        return returned, False
    one_edit = [
        candidate
        for candidate in expected_ids
        if len(candidate) == len(returned)
        and sum(left != right for left, right in zip(candidate, returned)) == 1
    ]
    if len(one_edit) == 1:
        return one_edit[0], True
    raise RuntimeError(f"Model returned unknown or ambiguous id {returned!r}")


def parse_response_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                if isinstance(content.get("text"), str):
                    return json.loads(content["text"])
    raise ValueError("OpenAI response contained no JSON output text")


def integer_score_schema() -> dict[str, Any]:
    return {"type": "integer", "minimum": 1, "maximum": 5}


GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {"slot": {"type": "integer"}, "lyrics": {"type": "string"}},
                "required": ["slot", "lyrics"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}
CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "dimension_scores": {
                        "type": "object",
                        "properties": {name: integer_score_schema() for name in DIMENSIONS},
                        "required": list(DIMENSIONS),
                        "additionalProperties": False,
                    },
                    "critical_failure_flags": {"type": "array", "items": {"type": "string"}},
                    "weak_dimensions": {"type": "array", "items": {"type": "string"}},
                    "repair_instructions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidate_id", "dimension_scores", "critical_failure_flags", "weak_dimensions", "repair_instructions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["critiques"],
    "additionalProperties": False,
}
REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}, "lyrics": {"type": "string"}},
                "required": ["candidate_id", "lyrics"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["revisions"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "blind_id": {"type": "string"},
                    "scores": {
                        "type": "object",
                        "properties": {name: integer_score_schema() for name in DIMENSIONS},
                        "required": list(DIMENSIONS),
                        "additionalProperties": False,
                    },
                    "critical_failure_flags": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                },
                "required": ["blind_id", "scores", "critical_failure_flags", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


class ApiBudget:
    def __init__(self, output_dir: Path, max_calls: int, max_total_tokens: int):
        self.path = output_dir / "api_calls.jsonl"
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens

    def totals(self) -> tuple[int, int]:
        rows = read_jsonl(self.path)
        return len(rows), sum(int((row.get("usage") or {}).get("total_tokens") or 0) for row in rows)

    def reserve(self) -> None:
        calls, tokens = self.totals()
        if calls >= self.max_calls:
            raise RuntimeError(f"API call budget exhausted: {calls}/{self.max_calls}")
        if tokens >= self.max_total_tokens:
            raise RuntimeError(f"API token budget exhausted: {tokens}/{self.max_total_tokens}")

    def record(self, row: dict[str, Any]) -> None:
        append_jsonl(self.path, [row])


def api_json(
    *,
    model: str,
    name: str,
    schema: dict[str, Any],
    system: str,
    user: str,
    budget: ApiBudget,
    retries: int,
    timeout: int,
    mock_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mock_payload is not None:
        return mock_payload, {"id": f"mock-{stable_id(name, user)}", "usage": {"total_tokens": 0}}
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    request_payload = {
        "model": model,
        "store": False,
        "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        request_payload["reasoning"] = {"effort": "low"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        budget.reserve()
        started = time.perf_counter()
        try:
            response = requests.post(
                RESPONSES_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=request_payload,
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"retryable status {response.status_code}: {response.text[:300]}")
            if not response.ok:
                raise RuntimeError(f"status {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            raw = response.json()
            parsed = parse_response_json(raw)
            meta = {
                "stage": name,
                "model": model,
                "response_id": raw.get("id"),
                "usage": raw.get("usage") or {},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            budget.record(meta)
            return parsed, meta
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            budget.record(
                {
                    "stage": name,
                    "model": model,
                    "attempt": attempt + 1,
                    "usage": {},
                    "error": str(exc)[:500],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            if attempt >= retries:
                break
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"OpenAI stage {name} failed: {last_error}") from last_error


def load_prompts(path: Path, discovery_themes: int = 8, families: Iterable[str] = ("technical", "clean")) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ordered_themes: list[str] = []
    for row in rows:
        theme = str(row["theme"])
        if theme not in ordered_themes:
            ordered_themes.append(theme)
    allowed = set(ordered_themes[:discovery_themes])
    allowed_families = set(families)
    return [row for row in rows if str(row["theme"]) in allowed and str(row["prompt_family"]) in allowed_families]


def ensure_manifest(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "prompts_sha256": sha256_file(args.prompts),
        "prompt_count": len(prompts),
        "teacher_model": args.teacher_model,
        "critic_model": args.critic_model,
        "judge_model": args.judge_model,
        "candidates_per_prompt": args.candidates_per_prompt,
        "seed": args.seed,
        "families": sorted(args.families),
        "prompt_version": "teacher_revise_v2_explicit_rhyme_contract",
        "metric_version": "rap_structural_v1",
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = args.output_dir / "manifest.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("fingerprint") != fingerprint:
            raise RuntimeError("Resume refused: manifest fingerprint changed")
        return current
    manifest = {"fingerprint": fingerprint, "spec": spec}
    write_json(path, manifest)
    return manifest


def record_command(args: argparse.Namespace) -> None:
    command = " ".join([sys.executable, *sys.argv])
    row = {
        "command": command,
        "command_id": stable_id(command),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = args.output_dir / "commands.jsonl"
    if row["command_id"] not in {item.get("command_id") for item in read_jsonl(path)}:
        append_jsonl(path, [row])


def baseline_by_prompt(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[str(row.get("prompt") or "")].append(row)
    return {prompt: min(rows, key=lambda row: (int(row.get("sample_index") or 0), int(row.get("candidate_index") or 0))) for prompt, rows in groups.items()}


def teacher_system(family: str) -> str:
    common = (
        "You are a top-tier rap lyric writer. Return four distinct original 12-line candidates. "
        "Every line must contribute; never include commentary, labels, or copied lyrics."
    )
    if family == "technical":
        return common + (
            " This is a technical rap test, not scene prose. Build three coherent four-bar movements. "
            "Use audible end-rhyme relationships on at least 8 of 12 lines, internal rhyme on at least 6 lines, "
            "and at least three repeated multisyllabic rhyme chains spanning adjacent lines. Keep cadence stable, "
            "syntax natural, and the scene semantically continuous. Never satisfy the rhyme counts with random word stacking. "
            "Resolve the narrative and rhyme pattern with a decisive final-bar payoff."
        )
    return common + " Produce radio-safe writing with natural language, tension, imagery, personality, punchlines, and technical quality equal to unrestricted rap."


def cmd_preflight(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.discovery_themes, args.families)
    ensure_manifest(args, prompts)
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    progress_path = args.output_dir / "preflight_progress.jsonl"
    results = read_jsonl(progress_path)
    completed = {(row.get("role"), row.get("model")) for row in results if row.get("ok") is True}
    # Recover successful checks from an older interrupted run that predates the
    # progress file. A successful API log entry means schema parsing completed.
    for call in read_jsonl(args.output_dir / "api_calls.jsonl"):
        stage = str(call.get("stage") or "")
        if stage.startswith("preflight_") and not call.get("error"):
            role = stage.removeprefix("preflight_")
            pair = (role, call.get("model"))
            if pair not in completed:
                recovered = {"role": role, "model": call.get("model"), "ok": True, "recovered": True}
                append_jsonl(progress_path, [recovered])
                results.append(recovered)
                completed.add(pair)
    for role, model in (("teacher", args.teacher_model), ("critic", args.critic_model), ("judge", args.judge_model)):
        if (role, model) in completed:
            continue
        parsed, meta = api_json(
            model=model,
            name=f"preflight_{role}",
            schema=schema,
            system="Return the requested JSON only.",
            user="Return {\"ok\": true}.",
            budget=budget,
            retries=0,
            timeout=args.timeout,
            mock_payload={"ok": True} if args.mock else None,
        )
        result = {"role": role, "model": model, "ok": parsed.get("ok") is True, **meta}
        append_jsonl(progress_path, [result])
        results.append(result)
    write_json(args.output_dir / "preflight.json", {"results": results})
    if not all(row["ok"] for row in results):
        raise RuntimeError("One or more model preflights failed")
    print(json.dumps({"status": "pass", "models": [row["model"] for row in results]}))


def cmd_generate(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.discovery_themes, args.families)
    ensure_manifest(args, prompts)
    out = args.output_dir / "originals.jsonl"
    existing = {row["prompt_id"] for row in read_jsonl(out)}
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    for prompt in prompts:
        prompt_id = stable_id(prompt["prompt"])
        if prompt_id in existing:
            continue
        mock = None
        if args.mock:
            mock = {"candidates": [{"slot": slot, "lyrics": "\n".join(f"mock {slot} concrete line {i} time" for i in range(12))} for slot in range(1, 5)]}
        parsed, meta = api_json(
            model=args.teacher_model,
            name="teacher_generate",
            schema=GENERATION_SCHEMA,
            system=teacher_system(str(prompt["prompt_family"])),
            user=str(prompt["prompt"]),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
            mock_payload=mock,
        )
        rows = []
        for item in parsed["candidates"]:
            cid = stable_id(prompt_id, item["slot"], args.seed, "teacher")
            rows.append({
                "prompt_id": prompt_id,
                "candidate_id": cid,
                "source": "teacher_generate",
                "family": prompt["prompt_family"],
                "theme": prompt["theme"],
                "prompt": prompt["prompt"],
                "lyrics": str(item["lyrics"]).strip(),
                "slot": item["slot"],
                "model": args.teacher_model,
                "response": meta,
            })
        append_jsonl(out, rows)
    print(json.dumps({"rows": len(read_jsonl(out)), "output": str(out)}))


def critique_system() -> str:
    return (
        "You are an independent rap editor. Diagnose exact failures in rhyme control, cadence, coherence, thematic depth, imagery, ending strength, and family compliance. "
        "For clean writing, cleanliness must not excuse generic language. For technical writing, explicitly inspect whether at least 8 line endings participate "
        "in audible rhyme, at least 6 lines contain internal rhyme, and three multisyllabic chains recur across adjacent lines; do not award technical quality "
        "for dense imagery or alliteration alone. Rhyme must not damage meaning or syntax. Return actionable repairs, not rewritten lyrics."
    )


def cmd_critique(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.discovery_themes, args.families)
    ensure_manifest(args, prompts)
    originals = read_jsonl(args.output_dir / "originals.jsonl")
    out = args.output_dir / "critiques.jsonl"
    existing = {row["prompt_id"] for row in read_jsonl(out)}
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in originals:
        by_prompt[row["prompt_id"]].append(row)
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    for prompt_id, rows in by_prompt.items():
        if prompt_id in existing:
            continue
        payload = [{"candidate_id": row["candidate_id"], "family": row["family"], "prompt": row["prompt"], "lyrics": row["lyrics"]} for row in rows]
        mock = None
        if args.mock:
            mock = {"critiques": [{"candidate_id": row["candidate_id"], "dimension_scores": {name: 3 for name in DIMENSIONS}, "critical_failure_flags": [], "weak_dimensions": ["imagery"], "repair_instructions": ["add concrete imagery"]} for row in rows]}
        parsed, meta = api_json(
            model=args.critic_model,
            name="critic_review",
            schema=CRITIQUE_SCHEMA,
            system=critique_system(),
            user=json.dumps(payload, ensure_ascii=False),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
            mock_payload=mock,
        )
        expected_ids = [row["candidate_id"] for row in rows]
        normalized = []
        for item in parsed["critiques"]:
            candidate_id, repaired = resolve_returned_id(str(item["candidate_id"]), expected_ids)
            normalized.append({"prompt_id": prompt_id, **item, "candidate_id": candidate_id, "id_repaired": repaired, "model": args.critic_model, "response": meta})
        if {row["candidate_id"] for row in normalized} != set(expected_ids):
            raise RuntimeError(f"Critic did not return exactly one critique per candidate for prompt {prompt_id}")
        append_jsonl(out, normalized)
    print(json.dumps({"rows": len(read_jsonl(out)), "output": str(out)}))


def cmd_revise(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.discovery_themes, args.families)
    ensure_manifest(args, prompts)
    originals = {row["candidate_id"]: row for row in read_jsonl(args.output_dir / "originals.jsonl")}
    critiques = read_jsonl(args.output_dir / "critiques.jsonl")
    out = args.output_dir / "revisions.jsonl"
    existing = {row["prompt_id"] for row in read_jsonl(out)}
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in critiques:
        if row["candidate_id"] not in originals:
            expected_ids = [candidate_id for candidate_id, original in originals.items() if original["prompt_id"] == row["prompt_id"]]
            corrected, repaired = resolve_returned_id(str(row["candidate_id"]), expected_ids)
            if repaired:
                append_jsonl(args.output_dir / "id_repairs.jsonl", [{"stage": "critic_review", "prompt_id": row["prompt_id"], "returned_id": row["candidate_id"], "resolved_id": corrected}])
                row = {**row, "candidate_id": corrected, "id_repaired": True}
        by_prompt[row["prompt_id"]].append(row)
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    for prompt_id, rows in by_prompt.items():
        if prompt_id in existing:
            continue
        payload = [{"candidate_id": row["candidate_id"], "original": originals[row["candidate_id"]]["lyrics"], "critique": row} for row in rows]
        family = originals[rows[0]["candidate_id"]]["family"]
        mock = None
        if args.mock:
            mock = {"revisions": [{"candidate_id": row["candidate_id"], "lyrics": originals[row["candidate_id"]]["lyrics"].replace("mock", "revised")} for row in rows]}
        parsed, meta = api_json(
            model=args.teacher_model,
            name="teacher_revise",
            schema=REVISION_SCHEMA,
            system=teacher_system(family) + " Revise each candidate only according to its independent critique; preserve strengths and return exactly 12 lines.",
            user=json.dumps(payload, ensure_ascii=False),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
            mock_payload=mock,
        )
        output_rows = []
        expected_ids = [row["candidate_id"] for row in rows]
        seen_ids: set[str] = set()
        for item in parsed["revisions"]:
            parent_id, repaired = resolve_returned_id(str(item["candidate_id"]), expected_ids)
            seen_ids.add(parent_id)
            parent = originals[parent_id]
            output_rows.append({
                **{key: parent[key] for key in ("prompt_id", "family", "theme", "prompt")},
                "candidate_id": stable_id(parent_id, "revision-1"),
                "parent_candidate_id": parent_id,
                "source": "independent_critique_revision",
                "lyrics": str(item["lyrics"]).strip(),
                "id_repaired": repaired,
                "model": args.teacher_model,
                "response": meta,
            })
        if seen_ids != set(expected_ids):
            raise RuntimeError(f"Teacher did not return exactly one revision per candidate for prompt {prompt_id}")
        append_jsonl(out, output_rows)
    print(json.dumps({"rows": len(read_jsonl(out)), "output": str(out)}))


def select_baselines(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = baseline_by_prompt(args.baseline)
    rows = []
    for prompt in prompts:
        source = available.get(str(prompt["prompt"]))
        if not source:
            raise RuntimeError(f"Saved baseline missing prompt: {prompt['prompt']}")
        rows.append({
            "prompt_id": stable_id(prompt["prompt"]),
            "candidate_id": stable_id(prompt["prompt"], "baseline"),
            "source": "current_generator",
            "family": prompt["prompt_family"],
            "theme": prompt["theme"],
            "prompt": prompt["prompt"],
            "lyrics": source.get("generated_text") or source.get("lyrics"),
            "model": source.get("settings", {}).get("base_model") or "Qwen/Qwen3-4B",
        })
    return rows


def judge_system() -> str:
    return (
        "You are a blind independent rap-quality judge. Candidate provenance and revision stage are hidden. Score evidence, not intent. "
        "An overall 4 cannot compensate for a dimension below 4. Family compliance is 5 only when every explicit family constraint is met. "
        "For technical lyrics, technical_rhyme must be 3 or lower unless the evidence names recurring end-rhyme relationships, internal rhymes, and "
        "multisyllabic chains; vivid scene prose and alliteration are not substitutes. Family compliance is 5 only if at least 8 line endings rhyme, "
        "at least 6 lines contain internal rhyme, and three multisyllabic chains recur across adjacent lines without harming meaning. "
        "For clean lyrics, cleanliness 5 also requires natural non-generic writing. Genericness 1 is best and 5 is worst."
    )


def cmd_judge(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.discovery_themes, args.families)
    ensure_manifest(args, prompts)
    candidates = select_baselines(args, prompts) + read_jsonl(args.output_dir / "originals.jsonl") + read_jsonl(args.output_dir / "revisions.jsonl")
    rng = random.Random(args.seed)
    mapping = []
    for row in candidates:
        mapping.append({"blind_id": stable_id(args.seed, row["candidate_id"], "blind"), **row})
    rng.shuffle(mapping)
    write_json(args.output_dir / "blind_map.json", {"rows": mapping})
    out = args.output_dir / "judgments.jsonl"
    existing = {row["blind_id"] for row in read_jsonl(out)}
    budget = ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens)
    for start in range(0, len(mapping), args.judge_batch_size):
        batch = [row for row in mapping[start : start + args.judge_batch_size] if row["blind_id"] not in existing]
        if not batch:
            continue
        payload = [{"blind_id": row["blind_id"], "family": row["family"], "prompt": row["prompt"], "lyrics": row["lyrics"]} for row in batch]
        mock = None
        if args.mock:
            mock = {"judgments": [{"blind_id": row["blind_id"], "scores": {name: (1 if name == "genericness" else 5) for name in DIMENSIONS}, "critical_failure_flags": [], "evidence": "mock evidence"} for row in batch]}
        parsed, meta = api_json(
            model=args.judge_model,
            name="blind_judge",
            schema=JUDGE_SCHEMA,
            system=judge_system(),
            user=json.dumps(payload, ensure_ascii=False),
            budget=budget,
            retries=args.retries,
            timeout=args.timeout,
            mock_payload=mock,
        )
        by_blind = {row["blind_id"]: row for row in batch}
        output = []
        for item in parsed["judgments"]:
            blind_id, repaired = resolve_returned_id(str(item["blind_id"]), by_blind)
            source = by_blind[blind_id]
            output.append({
                **item,
                "blind_id": blind_id,
                "id_repaired": repaired,
                **{key: source[key] for key in ("candidate_id", "prompt_id", "source", "family", "theme", "prompt", "lyrics", "model")},
                "parent_candidate_id": source.get("parent_candidate_id"),
                "judge_model": args.judge_model,
                "response": meta,
            })
        if len(output) != len(batch) or len({row["blind_id"] for row in output}) != len(batch):
            raise RuntimeError(f"Judge returned {len(output)} of {len(batch)} requested rows")
        append_jsonl(out, output)
        existing.update(row["blind_id"] for row in output)
    print(json.dumps({"rows": len(read_jsonl(out)), "output": str(out)}))


def words(line: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(line)]


def syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def rhyme_key(word: str, size: int = 3) -> str:
    token = re.sub(r"[^a-z]", "", word.lower())
    return token[-size:] if token else ""


def structural_metrics(lyrics: str) -> dict[str, Any]:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    line_words = [words(line) for line in lines]
    end_words = [tokens[-1] for tokens in line_words if tokens]
    end_keys = [rhyme_key(word) for word in end_words]
    end_counts = Counter(end_keys)
    end_rhyme_rate = sum(1 for key in end_keys if key and end_counts[key] > 1) / max(1, len(end_keys))
    internal_lines = 0
    multisyllabic_pairs = 0
    for tokens in line_words:
        keys = defaultdict(list)
        for token in set(tokens):
            if len(token) >= 4:
                keys[rhyme_key(token)].append(token)
        pairs = [values for values in keys.values() if len(values) >= 2]
        if pairs:
            internal_lines += 1
        multisyllabic_pairs += sum(1 for values in pairs if sum(syllables(value) >= 2 for value in values) >= 2)
    syllable_totals = [sum(syllables(token) for token in tokens) for tokens in line_words if tokens]
    mean_syllables = sum(syllable_totals) / max(1, len(syllable_totals))
    variance = sum((value - mean_syllables) ** 2 for value in syllable_totals) / max(1, len(syllable_totals))
    repeated_end_ratio = 1 - len(set(end_words)) / max(1, len(end_words))
    final = lines[-1] if lines else ""
    unresolved = bool(re.search(r"(?:\b(?:and|but|because|while|with|to|of|the|a)|[,;:\-])\s*$", final, re.I))
    token_set = [" ".join(sum(line_words, [] )[i : i + 5]) for i in range(max(0, len(sum(line_words, [])) - 4))]
    duplicate_5gram_ratio = 1 - len(set(token_set)) / max(1, len(token_set))
    clean_hits = sorted(set(sum(line_words, [])) & BLOCKED_CLEAN_TERMS)
    return {
        "line_count": len(lines),
        "internal_rhyme_line_count": internal_lines,
        "multisyllabic_rhyme_pair_count": multisyllabic_pairs,
        "end_rhyme_rate": round(end_rhyme_rate, 4),
        "repeated_end_word_ratio": round(repeated_end_ratio, 4),
        "syllable_cv": round(math.sqrt(variance) / mean_syllables, 4) if mean_syllables else 0.0,
        "duplicate_5gram_ratio": round(duplicate_5gram_ratio, 4),
        "unresolved_final_fragment": unresolved,
        "blocked_clean_terms": clean_hits,
    }


def strict_pass(row: dict[str, Any], parent: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    scores = row["scores"]
    metrics = structural_metrics(row["lyrics"])
    failures = []
    if metrics["line_count"] != 12:
        failures.append("line_count")
    for name in POSITIVE_DIMENSIONS:
        if int(scores[name]) < 4:
            failures.append(f"score_{name}")
    if int(scores["family_compliance"]) != 5:
        failures.append("family_compliance")
    if set(row.get("critical_failure_flags") or []) & CRITICAL_FLAGS:
        failures.append("critical_failure")
    if row["family"] == "technical":
        if metrics["internal_rhyme_line_count"] < 4:
            failures.append("internal_rhyme")
        if metrics["multisyllabic_rhyme_pair_count"] < 2:
            failures.append("multisyllabic_rhyme")
        if metrics["end_rhyme_rate"] < 0.33:
            failures.append("end_rhyme")
        if metrics["repeated_end_word_ratio"] > 0.25:
            failures.append("repeated_endings")
        if metrics["syllable_cv"] > 0.30:
            failures.append("syllable_variance")
    else:
        if int(scores["cleanliness"]) != 5:
            failures.append("cleanliness")
        if int(scores["genericness"]) > 2:
            failures.append("genericness")
        if metrics["blocked_clean_terms"]:
            failures.append("blocked_clean_terms")
    if metrics["unresolved_final_fragment"]:
        failures.append("unresolved_final")
    if metrics["duplicate_5gram_ratio"] > 0.15:
        failures.append("duplicate_phrase")
    if parent:
        for name in POSITIVE_DIMENSIONS:
            if int(scores[name]) < int(parent["scores"][name]) - 1 or int(scores[name]) < 4:
                failures.append(f"regressed_{name}")
    return not failures, sorted(set(failures))


def jaccard_ngrams(left: str, right: str, n: int = 5) -> float:
    def grams(text: str) -> set[tuple[str, ...]]:
        tokens = words(text)
        return {tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))}

    a, b = grams(left), grams(right)
    return len(a & b) / max(1, len(a | b))


def cmd_report(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.output_dir / "judgments.jsonl")
    by_candidate = {row["candidate_id"]: row for row in rows}
    for row in rows:
        parent = by_candidate.get(str(row.get("parent_candidate_id") or ""))
        passed, failures = strict_pass(row, parent)
        row["structural_metrics_v1"] = structural_metrics(row["lyrics"])
        row["strict_pass"] = passed
        row["strict_failures"] = failures
    accepted = [row for row in rows if row["strict_pass"] and row["source"] == "independent_critique_revision"]
    diverse: list[dict[str, Any]] = []
    for row in sorted(accepted, key=lambda item: (-sum(int(item["scores"][name]) for name in POSITIVE_DIMENSIONS), item["candidate_id"])):
        if all(jaccard_ngrams(row["lyrics"], other["lyrics"]) < 0.45 for other in diverse):
            diverse.append(row)
        else:
            row["strict_pass"] = False
            row["strict_failures"].append("diversity_similarity")
    accepted = diverse
    groups: dict[str, dict[str, Any]] = {}
    for family in args.families:
        groups[family] = {}
        for source in ("current_generator", "teacher_generate", "independent_critique_revision"):
            subset = [row for row in rows if row["family"] == family and row["source"] == source]
            groups[family][source] = {
                "rows": len(subset),
                "strict_pass": sum(bool(row["strict_pass"]) for row in subset),
                "strict_pass_rate": round(sum(bool(row["strict_pass"]) for row in subset) / max(1, len(subset)), 4),
                "failure_counts": dict(sorted(Counter(failure for row in subset for failure in row["strict_failures"]).items())),
            }
    phase1_pass = all(groups[family]["independent_critique_revision"]["strict_pass_rate"] >= 0.20 for family in groups)
    uplift_pass = all(
        groups[family]["independent_critique_revision"]["strict_pass_rate"]
        - groups[family]["teacher_generate"]["strict_pass_rate"] >= 0.10
        for family in groups
    )
    report = {
        "schema_version": 1,
        "decision": "advance_to_ablation" if phase1_pass and uplift_pass else "stop",
        "phase1_strict_yield_pass": phase1_pass,
        "revision_uplift_pass": uplift_pass,
        "groups": groups,
        "accepted_revision_rows": len(accepted),
        "accepted_themes_by_family": {
            family: len({row["theme"] for row in accepted if row["family"] == family})
            for family in args.families
        },
        "api_totals": dict(zip(("calls", "total_tokens"), ApiBudget(args.output_dir, args.max_api_calls, args.max_total_tokens).totals())),
    }
    write_json(args.output_dir / "report.json", report)
    append_jsonl(args.output_dir / "accepted.jsonl", accepted) if accepted and not (args.output_dir / "accepted.jsonl").exists() else None
    lines = ["# Teacher revise-and-rank Phase 1", "", f"Decision: **{report['decision'].upper()}**", ""]
    for family, values in groups.items():
        lines.extend([f"## {family}", "", "| Source | Rows | Strict pass | Yield |", "|---|---:|---:|---:|"])
        for source, metrics in values.items():
            lines.append(f"| {source} | {metrics['rows']} | {metrics['strict_pass']} | {metrics['strict_pass_rate']:.1%} |")
        lines.extend(["", "Failure counts:"])
        for source, metrics in values.items():
            failures = ", ".join(f"{name}={count}" for name, count in metrics["failure_counts"].items()) or "none"
            lines.append(f"- {source}: {failures}")
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--teacher-model", default=os.getenv("OPENAI_TEACHER_MODEL") or "gpt-5.4")
    parser.add_argument("--critic-model", default=os.getenv("OPENAI_CRITIC_MODEL") or "gpt-5.4-mini")
    parser.add_argument("--judge-model", default=os.getenv("OPENAI_JUDGE_MODEL") or "gpt-4.1-mini")
    parser.add_argument("--candidates-per-prompt", type=int, default=4)
    parser.add_argument("--discovery-themes", type=int, default=8)
    parser.add_argument("--families", nargs="+", choices=("technical", "clean"), default=["technical", "clean"])
    parser.add_argument("--judge-batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-api-calls", type=int, default=80)
    parser.add_argument("--max-total-tokens", type=int, default=500000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--mock", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = common_parser()
    for name, function in (
        ("preflight", cmd_preflight),
        ("generate", cmd_generate),
        ("critique", cmd_critique),
        ("revise", cmd_revise),
        ("judge", cmd_judge),
        ("report", cmd_report),
    ):
        child = sub.add_parser(name, parents=[common])
        child.set_defaults(function=function)
    run = sub.add_parser("run", parents=[common])
    run.set_defaults(function=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_command(args)
    if args.candidates_per_prompt != 4:
        raise ValueError("Phase 1 requires exactly four candidates per prompt")
    if args.command == "run":
        for function in (cmd_generate, cmd_critique, cmd_revise, cmd_judge, cmd_report):
            function(args)
    else:
        args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
