from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import PRIVATE_RESEARCH_POLICY, command_record, iter_jsonl, read_json, utc_now, write_json


SECTION_LABEL_RE = re.compile(
    r"^(?:\[(?:verse|hook|chorus|bridge|intro|outro|refrain|pre[- ]?chorus|post[- ]?chorus)[^\]]*\]"
    r"|(?:verse|hook|chorus|bridge|intro|outro|refrain)(?:\s+\d+)?\s*:?)$",
    re.IGNORECASE,
)
PROMPT_LEAK_RE = re.compile(
    r"<\|(?:task_generate|title|year|year_unknown|section|target_lines|content|lyrics)[^>]*\|>",
    re.IGNORECASE,
)


def prompt_from_row(row: dict[str, Any]) -> str:
    value = str(row.get("sft_text") or "")
    if not value:
        from .corpus import sft_document

        value = sft_document(
            str(row.get("title") or ""), row.get("year"), str(row.get("lyrics") or ""), list(row.get("content_flags") or [])
        )
    marker = "<|lyrics|>"
    return value.split(marker, 1)[0] + marker + "\n"


def reference_from_row(row: dict[str, Any]) -> str:
    value = str(row.get("sft_text") or "")
    if not value:
        from .corpus import sft_document

        value = sft_document(
            str(row.get("title") or ""), row.get("year"), str(row.get("lyrics") or ""), list(row.get("content_flags") or [])
        )
    marker = "<|lyrics|>"
    return value.split(marker, 1)[1].removesuffix("<|eos|>").strip() if marker in value else str(row.get("lyrics") or "")


def target_lines(row: dict[str, Any]) -> int:
    value = str(row.get("sft_text") or "")
    if not value:
        from .corpus import first_section

        return len(first_section(str(row.get("lyrics") or ""))[1])
    match = re.search(r"<\|target_lines\|>(\d+)", value)
    return int(match.group(1)) if match else 0


def legacy_lyric_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("<|")]


def is_section_label(line: str) -> bool:
    return bool(SECTION_LABEL_RE.fullmatch(line.strip()))


def is_prompt_leakage(line: str) -> bool:
    return bool(PROMPT_LEAK_RE.search(line))


def lyric_lines(text: str) -> list[str]:
    return [
        line
        for line in legacy_lyric_lines(text)
        if not is_section_label(line) and not is_prompt_leakage(line)
    ]


def normalized_line(line: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", line.lower()).strip()


def repeated_line_ratio(text: str) -> float:
    lines = [normalized_line(line) for line in lyric_lines(text)]
    lines = [line for line in lines if line]
    if not lines:
        return 1.0
    counts = Counter(lines)
    return sum(count for count in counts.values() if count > 1) / len(lines)


def distinct_n(texts: list[str], n: int) -> float:
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        words = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.lower())
        grams = [tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def generate_samples(
    model_path: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    model.config.use_cache = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    outputs: list[str] = []
    generated_tokens = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row_index, sequence in enumerate(generated):
            prompt_length = int(encoded["attention_mask"][row_index].sum().item())
            continuation = sequence[-(sequence.shape[0] - encoded["input_ids"].shape[1]) :] if sequence.shape[0] > encoded["input_ids"].shape[1] else sequence[0:0]
            generated_tokens += int(continuation.shape[0])
            outputs.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
    elapsed = time.monotonic() - started
    telemetry = {
        "model_path": model_path,
        "wall_seconds": round(elapsed, 3),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / max(elapsed, 1e-9), 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 6) if torch.cuda.is_available() else 0.0,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, telemetry


def summarize_outputs(rows: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    exact = 0
    section_ok = 0
    repetitions: list[float] = []
    for row, output in zip(rows, outputs):
        lines = lyric_lines(output)
        target = target_lines(row)
        if target and len(lines) == target:
            exact += 1
        if lines and not any(line.lower().startswith(("title:", "artist:", "explanation:")) for line in lines):
            section_ok += 1
        repetitions.append(repeated_line_ratio(output))
    count = max(1, len(outputs))
    return {
        "sample_count": len(outputs),
        "exact_line_match_rate": exact / count,
        "section_correctness_rate": section_ok / count,
        "average_repeated_line_ratio": sum(repetitions) / count,
        "distinct_1": distinct_n(outputs, 1),
        "distinct_2": distinct_n(outputs, 2),
        "distinct_3": distinct_n(outputs, 3),
    }


def compute_mauve(reference: list[str], generated: list[str], *, required: bool) -> dict[str, Any]:
    try:
        import mauve
    except ImportError:
        if required:
            raise RuntimeError("Install the metrics extra to require MAUVE evaluation.")
        return {"status": "unavailable", "reason": "mauve-text is not installed"}
    import torch

    result = mauve.compute_mauve(
        p_text=reference,
        q_text=generated,
        device_id=0 if torch.cuda.is_available() else -1,
        max_text_length=256,
        verbose=False,
    )
    return {"status": "complete", "score": float(result.mauve)}


def polynomial_hashes(token_ids: list[int], window: int, base: int = 1_000_003) -> dict[int, int]:
    if len(token_ids) < window:
        return {}
    mask = (1 << 63) - 1
    powers = [pow(base, index, 1 << 63) for index in range(window)]
    hashes: dict[int, int] = {}
    for start in range(len(token_ids) - window + 1):
        value = sum((token_ids[start + offset] + 1) * powers[offset] for offset in range(window)) & mask
        hashes[value] = hashes.get(value, 0) + 1
    return hashes


def scan_extraction(
    tokenizer: Any,
    outputs: list[str],
    train_manifest: dict[str, Any],
    *,
    windows: tuple[int, ...] = (20, 50),
    chunk_tokens: int = 250_000,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for extraction scanning.") from exc

    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in outputs]
    per_sample = {
        window: [set(polynomial_hashes(ids, window)) for ids in encoded]
        for window in windows
    }
    targets = {
        window: sorted(set().union(*per_sample[window])) if per_sample[window] else []
        for window in windows
    }
    matched: dict[int, set[int]] = {window: set() for window in windows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = (1 << 63) - 1
    scanned_tokens = 0
    for shard in train_manifest["shards"]:
        values = np.memmap(shard["path"], mode="r", dtype=np.uint16)
        for start in range(0, len(values), chunk_tokens):
            stop = min(len(values), start + chunk_tokens + max(windows) - 1)
            tensor = torch.as_tensor(np.asarray(values[start:stop], dtype=np.int64), device=device)
            scanned_tokens += min(chunk_tokens, len(values) - start)
            for window in windows:
                if tensor.numel() < window or not targets[window]:
                    continue
                powers = torch.tensor(
                    [pow(1_000_003, index, 1 << 63) for index in range(window)],
                    dtype=torch.int64,
                    device=device,
                )
                hashes = ((tensor.unfold(0, window, 1) + 1) * powers).sum(dim=1).bitwise_and(mask)
                target_tensor = torch.tensor(targets[window], dtype=torch.int64, device=device)
                hits = hashes[torch.isin(hashes, target_tensor)].unique().cpu().tolist()
                matched[window].update(int(value) for value in hits)
    samples: list[dict[str, Any]] = []
    for index in range(len(outputs)):
        row: dict[str, Any] = {"sample_index": index}
        for window in windows:
            sample_hashes = per_sample[window][index]
            hit_count = len(sample_hashes & matched[window])
            row[f"matched_{window}_token_windows"] = hit_count
            row[f"total_{window}_token_windows"] = len(sample_hashes)
            row[f"matched_{window}_token_fraction"] = hit_count / len(sample_hashes) if sample_hashes else 0.0
        samples.append(row)
    any_50 = sum(row["matched_50_token_windows"] > 0 for row in samples)
    any_20 = sum(row["matched_20_token_windows"] > 0 for row in samples)
    return {
        "scanned_training_tokens": scanned_tokens,
        "samples": samples,
        "samples_with_50_token_match": any_50,
        "samples_with_20_token_match": any_20,
        "fraction_with_20_token_match": any_20 / len(samples) if samples else 0.0,
        "scaling_gate_passed": any_50 == 0 and (any_20 / len(samples) if samples else 0.0) <= 0.05,
    }


def build_blind_packet(
    output_dir: Path,
    rows: list[dict[str, Any]],
    generations: dict[str, list[str]],
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    names = list(generations)
    if len(names) < 2:
        return {"status": "not_created", "reason": "at least two models are required"}
    randomizer = random.Random(seed)
    packet_path = output_dir / "blind_review_packet.csv"
    key: list[dict[str, Any]] = []
    with packet_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "prompt_id", "prompt", "output_a", "output_b", "coherence_a", "coherence_b",
            "rhyme_a", "rhyme_b", "structure_a", "structure_b", "originality_a", "originality_b",
            "adherence_a", "adherence_b", "overall_preference", "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows[:count]):
            selected = names[:2]
            if randomizer.random() < 0.5:
                selected = list(reversed(selected))
            writer.writerow(
                {
                    "prompt_id": index,
                    "prompt": prompt_from_row(row),
                    "output_a": generations[selected[0]][index],
                    "output_b": generations[selected[1]][index],
                }
            )
            key.append({"prompt_id": index, "a": selected[0], "b": selected[1]})
    key_path = output_dir / "blind_review_key.json"
    write_json(key_path, {"private_research_only": True, "rows": key})
    return {"status": "created", "packet": str(packet_path), "key": str(key_path), "prompt_count": min(count, len(rows))}


def score_blind_review(packet_path: Path, key_path: Path) -> dict[str, Any]:
    key_rows = {int(row["prompt_id"]): row for row in read_json(key_path)["rows"]}
    scratch_scores: dict[str, list[float]] = {name: [] for name in ("coherence", "structure", "originality")}
    structure_wins_or_ties = 0
    reviewed = 0
    with packet_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = int(row["prompt_id"])
            key = key_rows.get(prompt_id)
            if not key or "sft" not in (key["a"], key["b"]):
                continue
            side = "a" if key["a"] == "sft" else "b"
            other = "b" if side == "a" else "a"
            try:
                scores = {
                    name: float(row[f"{name}_{side}"])
                    for name in ("coherence", "structure", "originality")
                }
                other_structure = float(row[f"structure_{other}"])
            except (TypeError, ValueError):
                continue
            if any(value < 1 or value > 5 for value in [*scores.values(), other_structure]):
                continue
            reviewed += 1
            for name, value in scores.items():
                scratch_scores[name].append(value)
            if scores["structure"] >= other_structure:
                structure_wins_or_ties += 1
    medians = {
        name: float(np.median(values)) if values else None
        for name, values in scratch_scores.items()
    }
    structure_rate = structure_wins_or_ties / reviewed if reviewed else 0.0
    passed = (
        reviewed >= 30
        and all(value is not None and value >= 3.0 for value in medians.values())
        and structure_rate >= 0.35
    )
    return {
        "reviewed_comparisons": reviewed,
        "required_comparisons": 30,
        "scratch_median_scores": medians,
        "scratch_structure_win_or_tie_rate": structure_rate,
        "required_structure_win_or_tie_rate": 0.35,
        "human_gate_passed": passed,
    }


def finalize_review(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    report_path = Path(args.existing_report) if args.existing_report else output_dir / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")
    packet = Path(args.review_csv) if args.review_csv else output_dir / "blind_review_packet.csv"
    key = Path(args.review_key) if args.review_key else output_dir / "blind_review_key.json"
    human = score_blind_review(packet, key)
    report = read_json(report_path)
    report["blind_review"]["scoring"] = human
    extraction_passed = bool(report.get("scaling_gate", {}).get("automated_extraction_passed"))
    report["scaling_gate"].update(
        {
            "human_review_status": "passed" if human["human_gate_passed"] else "failed_or_incomplete",
            "human_review_passed": human["human_gate_passed"],
            "scale_to_50m_allowed": extraction_passed and human["human_gate_passed"],
        }
    )
    write_json(report_path, report)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.review_only:
        return finalize_review(args)
    started = time.monotonic()
    rows = list(iter_jsonl(Path(args.corpus_dir) / "test.jsonl"))[: args.samples]
    if not rows:
        raise RuntimeError("No test rows are available for evaluation.")
    prompts = [prompt_from_row(row) for row in rows]
    references = [reference_from_row(row) for row in rows]
    models = {"base": args.base_model, "sft": args.sft_model, "baseline": args.baseline_model}
    generations: dict[str, list[str]] = {}
    model_reports: dict[str, Any] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exact_command.txt").write_text(
        " ".join(command_record()) + "\n", encoding="utf-8"
    )
    for name, path in models.items():
        if not path:
            continue
        outputs, telemetry = generate_samples(
            str(path), prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, seed=args.seed
        )
        generations[name] = outputs
        write_json(output_dir / f"{name}_generations.json", {"outputs": outputs, "telemetry": telemetry})
        model_reports[name] = {
            "telemetry": telemetry,
            "metrics": summarize_outputs(rows, outputs),
            "mauve": compute_mauve(references, outputs, required=args.require_mauve),
        }
    extraction = None
    if "sft" in generations:
        tokenizer = AutoTokenizer.from_pretrained(args.sft_model, use_fast=True)
        token_manifest = read_json(Path(args.data_dir) / "tokenization_manifest.json")
        extraction = scan_extraction(
            tokenizer,
            generations["sft"],
            token_manifest["splits"]["base"]["train"],
        )
        write_json(output_dir / "extraction_report.json", extraction)
    blind_packet = build_blind_packet(
        output_dir, rows, generations, count=args.blind_prompts, seed=args.seed
    )
    report = {
        **PRIVATE_RESEARCH_POLICY,
        "generated_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": command_record(),
        "sample_count": len(rows),
        "models": model_reports,
        "extraction": extraction,
        "blind_review": blind_packet,
        "scaling_gate": {
            "automated_extraction_passed": extraction.get("scaling_gate_passed") if extraction else None,
            "human_reviews_required": 30,
            "human_review_status": "pending",
            "scale_to_50m_allowed": False,
        },
    }
    write_json(output_dir / "evaluation_report.json", report)
    return report


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--sft-model")
    parser.add_argument("--baseline-model")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--blind-prompts", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--require-mauve", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--review-key", type=Path)
    parser.add_argument("--existing-report", type=Path)
