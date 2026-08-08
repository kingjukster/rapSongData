"""Build specialized SFT datasets for Qwen3 rap-adapter probes."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SLUR_RE = re.compile(r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b", re.I)
QUESTION_RE = re.compile(r"\?")
DIALOGUE_RE = re.compile(r"[\"“”]|\b(?:said|says|asked|told|replied|answered)\b", re.I)
DRIFT_RE = re.compile(
    r"\b(?:haha|hahaha|lol|lmao|episode|screen|genius\.com|you might also like|embed|"
    r"kill anybody|burn your house|dead body|gonna kill|going to kill|shooting|stabbing)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return str(row.get("generated_text") or row.get("text") or "")
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return str(row.get("prompt") or "")
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text))


def normalize_sft(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("expected messages-format SFT row")
    metadata = dict(row.get("metadata") or {})
    metadata["specialized_source"] = source
    return {"messages": messages, "metadata": metadata}


def no_slur_requested(text: str) -> bool:
    lower = text.lower()
    return any(token in lower for token in ["no slur", "no-slur", "clean", "radio-safe", "radio safe"])


def good_shape(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not (4 <= len(lines) <= 24):
        return False
    if any(word_count(line) > 34 for line in lines):
        return False
    return word_count(text) >= 28


def is_control_clean(row: dict[str, Any]) -> bool:
    text = assistant_text(row)
    prompt = user_text(row)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if not good_shape(text):
        return False
    if DRIFT_RE.search(text) or DIALOGUE_RE.search(text):
        return False
    if QUESTION_RE.search(text) and line_count(text) > 4:
        return False
    if no_slur_requested(prompt) and SLUR_RE.search(text):
        return False
    if int(meta.get("control_score") or 0) < 4:
        return False
    return True


def is_rap_energy(row: dict[str, Any]) -> bool:
    text = assistant_text(row)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if not good_shape(text):
        return False
    if DRIFT_RE.search(text):
        return False
    if int(meta.get("quality_score") or 0) < 3:
        return False
    if int(meta.get("creativity_score") or 0) < 3:
        return False
    role = str(meta.get("section_role") or "").lower()
    if role and role not in {"verse", "hook", "chorus"}:
        return False
    return True


def repair_record(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(row.get("raw_generated_text") or row.get("rejected") or "").strip()
    fixed = str(row.get("generated_text") or row.get("chosen") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    if not raw or not fixed or raw == fixed:
        return None
    action_names = row.get("postprocess_action_names") or []
    metadata = {
        "specialized_source": "postprocess_repair_pair",
        "prompt_index": row.get("prompt_index"),
        "sample_index": row.get("sample_index"),
        "postprocess_action_names": action_names,
        "raw_decision_label": row.get("raw_decision_label"),
        "postprocessed_decision_label": row.get("postprocessed_decision_label"),
    }
    user = (
        "Repair this rap lyric completion so it stays lyric-only, keeps the requested intent, "
        "removes dangling endings or artifacts, and does not add explanation.\n\n"
        f"Original prompt:\n{prompt}\n\n"
        f"Raw completion:\n{raw}"
    ).strip()
    return {
        "messages": [
            {"role": "system", "content": "Repair rap lyric completions. Return only the corrected lyrics."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": fixed},
        ],
        "metadata": metadata,
    }


def split_rows(rows: list[dict[str, Any]], *, validation_records: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    validation_count = min(validation_records, max(1, len(shuffled) // 10))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    for split, split_rows_ in [("validation", validation), ("train", train)]:
        for row in split_rows_:
            row.setdefault("metadata", {})["split"] = split
    return train, validation


def build(args: argparse.Namespace) -> dict[str, Any]:
    controlled = read_jsonl(args.controlled_train)
    curation_repair = read_jsonl(args.repair_pairs) if args.repair_pairs.exists() else []
    raw_keep = read_jsonl(args.raw_keep) if args.raw_keep.exists() else []

    control_clean = [normalize_sft(row, source="control_clean") for row in controlled if is_control_clean(row)]
    rap_energy = [normalize_sft(row, source="rap_energy_controlled") for row in controlled if is_rap_energy(row)]

    # Add locally mined raw keepers as high-signal positives for the energy run.
    for row in raw_keep:
        prompt = str(row.get("prompt") or "").strip()
        text = str(row.get("raw_generated_text") or row.get("generated_text") or "").strip()
        if prompt and text and good_shape(text):
            rap_energy.append(
                {
                    "messages": [
                        {"role": "system", "content": "Write only original rap lyrics. Keep line breaks. Do not explain."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text},
                    ],
                    "metadata": {
                        "specialized_source": "raw_keep_positive",
                        "prompt_index": row.get("prompt_index"),
                        "sample_index": row.get("sample_index"),
                    },
                }
            )

    repairs = [record for row in curation_repair if (record := repair_record(row)) is not None]
    # Keep enough normal lyric examples in repair SFT so it does not become only edit-mode.
    repair_context = control_clean[: min(len(control_clean), max(500, len(repairs)))]
    repair_sft = repairs + repair_context

    datasets = {
        "control_clean": control_clean,
        "repair_sft": repair_sft,
        "rap_energy_balanced": rap_energy,
    }
    summary: dict[str, Any] = {
        "inputs": {
            "controlled_train": str(args.controlled_train),
            "repair_pairs": str(args.repair_pairs),
            "raw_keep": str(args.raw_keep),
        },
        "outputs": {},
        "datasets": {},
    }
    for name, rows in datasets.items():
        train, validation = split_rows(rows, validation_records=args.validation_records, seed=args.seed)
        train_path = args.out_dir / f"qwen3_{name}_train.jsonl"
        validation_path = args.out_dir / f"qwen3_{name}_validation.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(validation_path, validation)
        source_counts = Counter(str(row.get("metadata", {}).get("specialized_source") or "unknown") for row in rows)
        summary["outputs"][name] = {"train": str(train_path), "validation": str(validation_path)}
        summary["datasets"][name] = {
            "total": len(rows),
            "train": len(train),
            "validation": len(validation),
            "sources": dict(source_counts),
        }
    summary_path = args.out_dir / "qwen3_specialized_sft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-train", type=Path, default=Path("data/sft/rap_full_song_openai_sft_controlled_v2_train.jsonl"))
    parser.add_argument("--repair-pairs", type=Path, default=Path("data/curation/local_postprocess_v1_500/postprocess_repair_pairs.jsonl"))
    parser.add_argument("--raw-keep", type=Path, default=Path("data/curation/local_postprocess_v1_500/raw_keep_positive.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft/qwen3_specialized"))
    parser.add_argument("--validation-records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
