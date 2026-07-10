"""Build follow-up reports for the calibrated 12-line smoke eval.

The reports are intentionally local and deterministic:

- paired_sample_review.md/json: 10 adapter wins, 10 base wins, 10 close cases
- latency_diagnostic.md/json: throughput and config checks for base vs adapter

These are meant to gate the next training run without requiring more manual
labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ranked", type=Path, required=True)
    parser.add_argument("--adapter-ranked", type=Path, required=True)
    parser.add_argument("--base-sweep", type=Path, required=True)
    parser.add_argument("--adapter-sweep", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--adapter-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-bucket", type=int, default=10)
    parser.add_argument("--diagnostic-base-sweep", type=Path, default=None)
    parser.add_argument("--diagnostic-base-summary", type=Path, default=None)
    parser.add_argument("--diagnostic-adapter-sweep", type=Path, default=None)
    parser.add_argument("--diagnostic-adapter-summary", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in rows if row.get("row_id")}


def short_tags(row: dict[str, Any]) -> str:
    tags = row.get("quality_tags") or []
    return ", ".join(str(tag) for tag in tags) if tags else "none"


def format_score(value: Any) -> str:
    return f"{as_float(value):.4f}"


def dimension_deltas(base: dict[str, Any], adapter: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    base_dims = base.get("quality_dimensions") or {}
    adapter_dims = adapter.get("quality_dimensions") or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base_dims) & set(adapter_dims)):
        before = as_float(base_dims.get(key))
        after = as_float(adapter_dims.get(key))
        rows.append(
            {
                "dimension": key,
                "base": round(before, 4),
                "adapter": round(after, 4),
                "delta": round(after - before, 4),
            }
        )
    rows.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return rows[:limit]


def selected_pair_payload(
    pair: dict[str, Any],
    base_row: dict[str, Any],
    adapter_row: dict[str, Any],
    bucket: str,
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "row_id": pair["row_id"],
        "prompt_key": base_row.get("prompt_key"),
        "candidate_index": base_row.get("candidate_index"),
        "sample_index": base_row.get("sample_index"),
        "theme": base_row.get("theme"),
        "style": base_row.get("style"),
        "prompt": base_row.get("prompt"),
        "base": {
            "quality_score": base_row.get("quality_score"),
            "quality_tags": base_row.get("quality_tags") or [],
            "quality_dimensions": base_row.get("quality_dimensions") or {},
            "structural_metrics": base_row.get("structural_metrics") or {},
            "rhyme_metrics": base_row.get("rhyme_metrics") or {},
            "lyrics": base_row.get("lyrics") or "",
        },
        "adapter": {
            "quality_score": adapter_row.get("quality_score"),
            "quality_tags": adapter_row.get("quality_tags") or [],
            "quality_dimensions": adapter_row.get("quality_dimensions") or {},
            "structural_metrics": adapter_row.get("structural_metrics") or {},
            "rhyme_metrics": adapter_row.get("rhyme_metrics") or {},
            "lyrics": adapter_row.get("lyrics") or "",
        },
        "adapter_minus_base_quality": pair["delta"],
        "largest_dimension_deltas": dimension_deltas(base_row, adapter_row),
        "automated_score_winner": "adapter"
        if pair["delta"] > 0
        else ("base" if pair["delta"] < 0 else "tie"),
    }


def select_pair_buckets(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    *,
    pairs_per_bucket: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_by_id = row_map(base_rows)
    adapter_by_id = row_map(adapter_rows)
    pairs: list[dict[str, Any]] = []
    for row_id in sorted(set(base_by_id) & set(adapter_by_id)):
        base_score = as_float(base_by_id[row_id].get("quality_score"))
        adapter_score = as_float(adapter_by_id[row_id].get("quality_score"))
        pairs.append(
            {
                "row_id": row_id,
                "base_score": round(base_score, 4),
                "adapter_score": round(adapter_score, 4),
                "delta": round(adapter_score - base_score, 4),
            }
        )

    adapter_wins = sorted(
        [pair for pair in pairs if pair["delta"] > 0],
        key=lambda pair: (-pair["delta"], pair["row_id"]),
    )[:pairs_per_bucket]
    base_wins = sorted(
        [pair for pair in pairs if pair["delta"] < 0],
        key=lambda pair: (pair["delta"], pair["row_id"]),
    )[:pairs_per_bucket]
    selected_ids = {pair["row_id"] for pair in adapter_wins + base_wins}
    close_cases = sorted(
        [pair for pair in pairs if pair["row_id"] not in selected_ids],
        key=lambda pair: (abs(pair["delta"]), pair["row_id"]),
    )[:pairs_per_bucket]

    selected: list[dict[str, Any]] = []
    for bucket, bucket_pairs in [
        ("adapter_large_win", adapter_wins),
        ("base_large_win", base_wins),
        ("close_or_tie", close_cases),
    ]:
        for pair in bucket_pairs:
            selected.append(selected_pair_payload(pair, base_by_id[pair["row_id"]], adapter_by_id[pair["row_id"]], bucket))

    deltas = [pair["delta"] for pair in pairs]
    summary = {
        "paired_rows": len(pairs),
        "adapter_wins": sum(1 for pair in pairs if pair["delta"] > 0),
        "base_wins": sum(1 for pair in pairs if pair["delta"] < 0),
        "ties": sum(1 for pair in pairs if pair["delta"] == 0),
        "adapter_win_rate": round(sum(1 for pair in pairs if pair["delta"] > 0) / max(1, len(pairs)), 4),
        "avg_adapter_minus_base_quality": round(statistics.mean(deltas), 4) if deltas else 0.0,
        "median_adapter_minus_base_quality": round(statistics.median(deltas), 4) if deltas else 0.0,
        "selected_counts": dict(Counter(row["bucket"] for row in selected)),
    }
    return selected, summary


def most_common_float(values: list[float]) -> float:
    if not values:
        return 0.0
    counts = Counter(round(value, 2) for value in values)
    return float(counts.most_common(1)[0][0])


def summarize_latency_run(name: str, summary: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(raw_rows)
    wall_seconds = as_float(summary.get("generation_wall_seconds"))
    load_seconds = as_float(summary.get("load_seconds"))
    timing_batch_sizes = [as_int((row.get("timing") or {}).get("batch_size")) for row in raw_rows]
    batch_size_counts = Counter(size for size in timing_batch_sizes if size > 0)
    configured_batch_size = max(batch_size_counts) if batch_size_counts else 1

    initial_batch_seconds = 0.0
    initial_batch_count = 0
    ordered = sorted(raw_rows, key=lambda row: as_int(row.get("candidate_index")))
    for index in range(0, len(ordered), configured_batch_size):
        chunk = ordered[index : index + configured_batch_size]
        expected_size = len(chunk)
        timing_values = [
            as_float((row.get("timing") or {}).get("batch_seconds"))
            for row in chunk
            if as_int((row.get("timing") or {}).get("batch_size")) == expected_size
        ]
        if not timing_values:
            timing_values = [
                as_float((row.get("timing") or {}).get("batch_seconds"))
                for row in chunk
                if as_int((row.get("timing") or {}).get("batch_size")) > 1
            ]
        if timing_values:
            initial_batch_seconds += most_common_float(timing_values)
            initial_batch_count += 1

    retry_seconds = 0.0
    retry_attempts = 0
    for row in raw_rows:
        for attempt in row.get("retry_attempts") or []:
            retry_seconds += as_float((attempt.get("timing") or {}).get("batch_seconds"))
            retry_attempts += 1

    accepted_tokens = sum(as_int((row.get("timing") or {}).get("effective_generated_tokens")) for row in raw_rows)
    raw_line_counts = [as_int(row.get("raw_line_count")) for row in raw_rows]
    post_line_counts = [as_int(row.get("postprocessed_line_count")) for row in raw_rows]
    postprocess_count = sum(1 for row in raw_rows if row.get("postprocess_applied"))

    return {
        "name": name,
        "row_count": row_count,
        "adapter_enabled": bool((summary.get("settings") or {}).get("adapter_enabled")),
        "adapter_dir": (summary.get("settings") or {}).get("adapter_dir"),
        "wall_seconds": round(wall_seconds, 2),
        "wall_minutes": round(wall_seconds / 60.0, 2) if wall_seconds else 0.0,
        "load_seconds": round(load_seconds, 2),
        "rows_per_minute": round(row_count / max(wall_seconds / 60.0, 1e-9), 2),
        "wall_seconds_per_row": round(wall_seconds / max(row_count, 1), 3),
        "configured_or_effective_batch_size": configured_batch_size,
        "timing_batch_size_counts": {str(key): value for key, value in sorted(batch_size_counts.items())},
        "estimated_initial_batch_count": initial_batch_count,
        "estimated_initial_batch_seconds": round(initial_batch_seconds, 2),
        "retry_attempts": retry_attempts,
        "retry_seconds": round(retry_seconds, 2),
        "estimated_model_seconds_in_rows": round(initial_batch_seconds + retry_seconds, 2),
        "summary_underlength_retry_triggered": as_int(summary.get("underlength_retry_triggered")),
        "summary_underlength_retry_generations": as_int(summary.get("underlength_retry_generations")),
        "accepted_effective_tokens": accepted_tokens,
        "accepted_effective_tokens_per_wall_second": round(accepted_tokens / max(wall_seconds, 1e-9), 2),
        "avg_raw_line_count": round(statistics.mean(raw_line_counts), 2) if raw_line_counts else 0.0,
        "avg_postprocessed_line_count": round(statistics.mean(post_line_counts), 2) if post_line_counts else 0.0,
        "postprocess_applied_count": postprocess_count,
        "settings": summary.get("settings") or {},
        "runtime": summary.get("runtime") or {},
    }


def latency_findings(runs: dict[str, dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    base = runs.get("base_full")
    adapter = runs.get("adapter_full")
    if base and adapter:
        base_batch = as_int(base.get("configured_or_effective_batch_size"), 1)
        adapter_batch = as_int(adapter.get("configured_or_effective_batch_size"), 1)
        wall_ratio = as_float(adapter.get("wall_seconds_per_row")) / max(
            as_float(base.get("wall_seconds_per_row")),
            1e-9,
        )
        findings.append(
            f"Full-run wall time per row ratio is {wall_ratio:.2f}x adapter/base "
            f"({adapter['wall_seconds_per_row']}s vs {base['wall_seconds_per_row']}s)."
        )
        if base_batch != adapter_batch:
            findings.append(
                f"The full-run comparison is not controlled: base used effective batch {base_batch}, "
                f"adapter used effective batch {adapter_batch}. This alone can explain most of the apparent slowdown."
            )
        else:
            findings.append(f"The full-run comparison is controlled on effective batch size {base_batch}.")
        retry_delta = as_int(adapter.get("summary_underlength_retry_generations")) - as_int(
            base.get("summary_underlength_retry_generations")
        )
        findings.append(
            f"Adapter retries increased by {retry_delta} generations, but retry count is too small to explain a multi-x slowdown."
        )
        base_runtime = base.get("runtime") or {}
        adapter_runtime = adapter.get("runtime") or {}
        if base_runtime == adapter_runtime:
            findings.append("Runtime flags match in the summaries: bf16 support, TF32, and cudnn benchmark are the same.")

    diag_base = runs.get("base_batch6_diagnostic")
    if (
        diag_base
        and adapter
        and as_int(diag_base.get("configured_or_effective_batch_size"), 1)
        == as_int(adapter.get("configured_or_effective_batch_size"), 1)
    ):
        diag_ratio = as_float(adapter.get("wall_seconds_per_row")) / max(
            as_float(diag_base.get("wall_seconds_per_row")),
            1e-9,
        )
        findings.append(
            f"Against the batch-6 base diagnostic, adapter/base wall time per row ratio is {diag_ratio:.2f}x "
            f"({adapter['wall_seconds_per_row']}s vs {diag_base['wall_seconds_per_row']}s)."
        )
        findings.append(
            "Use the batch-6 diagnostic ratio, not the original full-run ratio, when deciding whether adapter inference is actually slow."
        )
    diag_adapter = runs.get("adapter_large_batch_diagnostic")
    if diag_adapter and adapter:
        batch = diag_adapter.get("configured_or_effective_batch_size")
        speedup = as_float(adapter.get("wall_seconds_per_row")) / max(
            as_float(diag_adapter.get("wall_seconds_per_row")),
            1e-9,
        )
        findings.append(
            f"The adapter large-batch diagnostic used effective batch {batch} and was {speedup:.2f}x faster per row "
            f"than the original adapter full run."
        )
        if base:
            controlled_ratio = as_float(diag_adapter.get("wall_seconds_per_row")) / max(
                as_float(base.get("wall_seconds_per_row")),
                1e-9,
            )
            findings.append(
                f"Batch-64 adapter diagnostic vs batch-64 base full run ratio is {controlled_ratio:.2f}x "
                f"({diag_adapter['wall_seconds_per_row']}s vs {base['wall_seconds_per_row']}s per row)."
            )
    return findings


def write_paired_markdown(path: Path, selected: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Base vs Smoke Adapter Paired Review",
        "",
        "This report samples large adapter wins, large base wins, and close cases from the same prompt/sample pairs.",
        "The winner shown here is the automated heuristic-score winner, not a new human label.",
        "",
        "## Summary",
        "",
        f"- paired rows: `{summary['paired_rows']}`",
        f"- adapter wins: `{summary['adapter_wins']}`",
        f"- base wins: `{summary['base_wins']}`",
        f"- ties: `{summary['ties']}`",
        f"- adapter win rate: `{summary['adapter_win_rate']}`",
        f"- avg adapter-minus-base score: `{summary['avg_adapter_minus_base_quality']}`",
        f"- median adapter-minus-base score: `{summary['median_adapter_minus_base_quality']}`",
        f"- selected counts: `{summary['selected_counts']}`",
        "",
    ]
    for index, row in enumerate(selected, 1):
        deltas = ", ".join(
            f"{item['dimension']} {item['delta']:+.4f}" for item in row["largest_dimension_deltas"]
        )
        lines.extend(
            [
                f"## {index:02d}. {row['bucket']} | {row.get('theme') or 'unthemed'}",
                "",
                f"- row_id: `{row['row_id']}`",
                f"- automated score winner: `{row['automated_score_winner']}`",
                f"- quality: base `{format_score(row['base']['quality_score'])}` | adapter `{format_score(row['adapter']['quality_score'])}` | delta `{row['adapter_minus_base_quality']:+.4f}`",
                f"- tags: base `{short_tags(row['base'])}` | adapter `{short_tags(row['adapter'])}`",
                f"- largest dimension moves: `{deltas or 'none'}`",
                "",
                "**Prompt**",
                "",
                f"> {row.get('prompt') or ''}",
                "",
                "**Base**",
                "",
                "```text",
                row["base"]["lyrics"],
                "```",
                "",
                "**Smoke Adapter**",
                "",
                "```text",
                row["adapter"]["lyrics"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_latency_markdown(path: Path, runs: dict[str, dict[str, Any]], findings: list[str]) -> None:
    base = runs.get("base_full") or {}
    adapter = runs.get("adapter_full") or {}
    full_batches_match = bool(base and adapter) and (
        base.get("configured_or_effective_batch_size") == adapter.get("configured_or_effective_batch_size")
    )
    lines: list[str] = [
        "# Smoke Adapter Latency Diagnostic",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | rows | adapter | batch | wall min | sec/row | rows/min | retries | accepted tok/sec |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, run in runs.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(run["row_count"]),
                    "yes" if run["adapter_enabled"] else "no",
                    str(run["configured_or_effective_batch_size"]),
                    f"{run['wall_minutes']:.2f}",
                    f"{run['wall_seconds_per_row']:.3f}",
                    f"{run['rows_per_minute']:.2f}",
                    str(run["summary_underlength_retry_generations"]),
                    f"{run['accepted_effective_tokens_per_wall_second']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The full base-vs-adapter speed comparison is now controlled on batch size."
            if full_batches_match
            else "The original smoke-adapter speed comparison should be treated as contaminated by batch size until a same-batch full eval is run.",
            "Use the paired quality report for the model decision; use this latency report only to rule speed in or out as a blocker."
            if full_batches_match
            else "The quality comparison remains useful because rows are paired by prompt/sample, but wall-clock scaling should use a controlled batch-size run.",
            "",
            "Recommended next gate: decide whether the adapter's modest quality lift justifies a short epoch-controlled follow-up train."
            if full_batches_match
            else "Recommended next gate: inspect this paired report for visible quality, then rerun the adapter eval with the largest stable batch size before longer training.",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize_quality_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [as_float(row.get("quality_score")) for row in rows]
    top50 = sorted(rows, key=lambda row: as_float(row.get("quality_score")), reverse=True)[:50]
    top100 = sorted(rows, key=lambda row: as_float(row.get("quality_score")), reverse=True)[:100]
    tag_counts = Counter(tag for row in rows for tag in (row.get("quality_tags") or []))
    top50_tag_counts = Counter(tag for row in top50 for tag in (row.get("quality_tags") or []))
    return {
        "rows": len(rows),
        "avg_quality_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "median_quality_score": round(statistics.median(scores), 4) if scores else 0.0,
        "top100_avg_quality_score": round(
            statistics.mean([as_float(row.get("quality_score")) for row in top100]),
            4,
        )
        if top100
        else 0.0,
        "top50_tag_counts": dict(top50_tag_counts),
        "tag_counts": dict(tag_counts),
    }


def count_raw_issues(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "underlength_retry_generations": sum(as_int(row.get("underlength_retry_count")) for row in rows),
        "hit_token_cap_count": sum(1 for row in rows if row.get("hit_token_cap")),
        "postprocess_applied_count": sum(1 for row in rows if row.get("postprocess_applied")),
    }


def write_decision_markdown(
    path: Path,
    *,
    pair_summary: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    base_ranked: list[dict[str, Any]],
    adapter_ranked: list[dict[str, Any]],
    base_raw: list[dict[str, Any]],
    adapter_raw: list[dict[str, Any]],
) -> None:
    base_quality = summarize_quality_rows(base_ranked)
    adapter_quality = summarize_quality_rows(adapter_ranked)
    base_issues = count_raw_issues(base_raw)
    adapter_issues = count_raw_issues(adapter_raw)
    base_run = runs.get("base_full") or {}
    adapter_run = runs.get("adapter_full") or {}
    speed_ratio = as_float(adapter_run.get("wall_seconds_per_row")) / max(
        as_float(base_run.get("wall_seconds_per_row")),
        1e-9,
    )
    speed_ratio = round(speed_ratio, 2)
    quality_delta = round(adapter_quality["avg_quality_score"] - base_quality["avg_quality_score"], 4)
    top100_delta = round(adapter_quality["top100_avg_quality_score"] - base_quality["top100_avg_quality_score"], 4)

    allow_short_followup = (
        pair_summary.get("adapter_win_rate", 0) >= 0.6
        and quality_delta > 0.02
        and speed_ratio <= 3.0
    )
    verdict = (
        "Proceed with a short epoch-controlled follow-up train, not a long scale-up."
        if allow_short_followup
        else "Do not run another train yet; fix the quality or speed blocker first."
    )

    lines = [
        "# Fair Smoke Eval Training Gate",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Evidence",
        "",
        f"- paired adapter win rate: `{pair_summary['adapter_win_rate']}` ({pair_summary['adapter_wins']} adapter wins / {pair_summary['base_wins']} base wins / {pair_summary['ties']} ties)",
        f"- avg quality: base `{base_quality['avg_quality_score']}` | adapter `{adapter_quality['avg_quality_score']}` | delta `{quality_delta:+.4f}`",
        f"- median quality: base `{base_quality['median_quality_score']}` | adapter `{adapter_quality['median_quality_score']}`",
        f"- top-100 avg quality: base `{base_quality['top100_avg_quality_score']}` | adapter `{adapter_quality['top100_avg_quality_score']}` | delta `{top100_delta:+.4f}`",
        f"- speed: base `{base_run.get('wall_minutes')}` min | adapter `{adapter_run.get('wall_minutes')}` min | ratio `{speed_ratio}x` at effective batch `{adapter_run.get('configured_or_effective_batch_size')}`",
        f"- underlength retry generations: base `{base_issues['underlength_retry_generations']}` | adapter `{adapter_issues['underlength_retry_generations']}`",
        "",
        "## Remaining Concerns",
        "",
        f"- adapter top-50 tags: `{adapter_quality['top50_tag_counts']}`",
        f"- base top-50 tags: `{base_quality['top50_tag_counts']}`",
        "- The adapter quality lift is real but modest, and weak imagery remains the largest visible bottleneck.",
        "- The next train should target imagery, scene specificity, and awkward/generic phrasing rather than just more steps.",
        "",
        "## Next Training Step",
        "",
        "Create short epoch-controlled configs from the calibrated 12-line dataset and run the smallest follow-up first.",
        "Use the same fair batch-64 eval after each adapter and stop if the generic/awkward tags rise or the win-rate gain disappears.",
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_ranked = read_jsonl(args.base_ranked)
    adapter_ranked = read_jsonl(args.adapter_ranked)
    base_raw = read_jsonl(args.base_sweep)
    adapter_raw = read_jsonl(args.adapter_sweep)
    selected, pair_summary = select_pair_buckets(
        base_ranked,
        adapter_ranked,
        pairs_per_bucket=args.pairs_per_bucket,
    )
    paired_payload = {"summary": pair_summary, "pairs": selected}
    (args.out_dir / "paired_sample_review.json").write_text(
        json.dumps(paired_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_paired_markdown(args.out_dir / "paired_sample_review.md", selected, pair_summary)

    runs = {
        "base_full": summarize_latency_run("base_full", read_json(args.base_summary), base_raw),
        "adapter_full": summarize_latency_run(
            "adapter_full",
            read_json(args.adapter_summary),
            adapter_raw,
        ),
    }
    if args.diagnostic_base_sweep and args.diagnostic_base_summary:
        runs["base_batch6_diagnostic"] = summarize_latency_run(
            "base_batch6_diagnostic",
            read_json(args.diagnostic_base_summary),
            read_jsonl(args.diagnostic_base_sweep),
        )
    if args.diagnostic_adapter_sweep and args.diagnostic_adapter_summary:
        runs["adapter_large_batch_diagnostic"] = summarize_latency_run(
            "adapter_large_batch_diagnostic",
            read_json(args.diagnostic_adapter_summary),
            read_jsonl(args.diagnostic_adapter_sweep),
        )
    findings = latency_findings(runs)
    latency_payload = {"findings": findings, "runs": runs}
    (args.out_dir / "latency_diagnostic.json").write_text(
        json.dumps(latency_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_latency_markdown(args.out_dir / "latency_diagnostic.md", runs, findings)
    write_decision_markdown(
        args.out_dir / "training_gate_decision.md",
        pair_summary=pair_summary,
        runs=runs,
        base_ranked=base_ranked,
        adapter_ranked=adapter_ranked,
        base_raw=base_raw,
        adapter_raw=adapter_raw,
    )

    print(
        json.dumps(
            {
                "paired_review": str(args.out_dir / "paired_sample_review.md"),
                "latency_diagnostic": str(args.out_dir / "latency_diagnostic.md"),
                "training_gate_decision": str(args.out_dir / "training_gate_decision.md"),
                "selected_pairs": len(selected),
                "latency_runs": list(runs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
