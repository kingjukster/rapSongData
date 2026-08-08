"""Compare saved fixed-prompt generation eval JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_PATTERNS = {
    "dialogue_quote": re.compile(r'"[^"\n]{8,}"'),
    "laughter_adlib": re.compile(r"\b(?:haha|ha ha|uh-huh|ohh+|ooo+|yeah yeah yeah)\b", re.I),
    "unfinished": re.compile(r"(?:\(|,|:|-|and|but|because|cause|when|if|the|a|I|you)$", re.I),
    "prose_marker": re.compile(r"\b(?:said|replied|asks|remarks|speaks|told me)\b", re.I),
}


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_for(row: dict) -> str:
    return str(row.get("generated_text") or row.get("text") or row.get("raw_text") or "")


def analysis_for(row: dict, text: str) -> dict:
    if isinstance(row.get("analysis"), dict):
        return row["analysis"]
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "line_count": len(lines),
        "word_count": len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)),
        "slur_count": len(re.findall(r"\b(?:nigga(?:s)?|nigger(?:s)?|faggot(?:s)?|fa[g]{2}(?:ot)?s?)\b", text, re.I)),
        "repeated_line_ratio": 0.0,
        "exact_line_match": None,
        "hook_line_cap_ok": None,
    }


def timing_for(row: dict) -> dict:
    return row.get("timing") or row.get("generation") or {}


def label_for(path: Path) -> str:
    return path.stem.removeprefix("generation_eval_")


def summarize(path: Path) -> dict:
    rows = read_rows(path)
    texts = [text_for(row) for row in rows]
    analyses = [analysis_for(row, text) for row, text in zip(rows, texts)]
    timings = [timing_for(row) for row in rows]
    hook_total = sum(1 for item in analyses if item.get("hook_line_cap_ok") is not None)
    flags = {
        name: sum(1 for text in texts if pattern.search(text))
        for name, pattern in BAD_PATTERNS.items()
    }
    avg = lambda values: round(sum(values) / max(1, len(values)), 2)
    line_counts = [int(item.get("line_count") or 0) for item in analyses]
    word_counts = [int(item.get("word_count") or 0) for item in analyses]
    slur_counts = [int(item.get("slur_count") or 0) for item in analyses]
    repeated = [float(item.get("repeated_line_ratio") or 0.0) for item in analyses]
    toksec = [float(item.get("tokens_per_second") or 0.0) for item in timings]
    vram = [float(item.get("max_memory_allocated_gb") or 0.0) for item in timings]
    exact_matches = sum(1 for item in analyses if item.get("exact_line_match") is True)
    hook_pass = sum(1 for item in analyses if item.get("hook_line_cap_ok") is True)

    control_score = 100
    control_score -= 10 * flags["dialogue_quote"]
    control_score -= 8 * flags["laughter_adlib"]
    control_score -= 8 * flags["prose_marker"]
    control_score -= 8 * flags["unfinished"]
    if hook_total:
        control_score -= 12 * (hook_total - hook_pass)
    control_score -= 4 * (len(rows) - exact_matches)
    control_score = max(0, control_score)

    fullness_score = 0
    for count in line_counts:
        if 8 <= count <= 18:
            fullness_score += 1
    fullness_score = round(100 * fullness_score / max(1, len(line_counts)), 1)

    return {
        "label": label_for(path),
        "path": str(path),
        "records": len(rows),
        "line_counts": line_counts,
        "avg_lines": avg(line_counts),
        "avg_words": avg(word_counts),
        "slur_counts": slur_counts,
        "slur_prompt_count": sum(1 for count in slur_counts if count > 0),
        "exact_line_match_count": exact_matches,
        "hook_line_cap": f"{hook_pass}/{hook_total}" if hook_total else "n/a",
        "avg_repeated_line_ratio": round(sum(repeated) / max(1, len(repeated)), 4),
        "avg_tokens_per_second": avg(toksec),
        "peak_vram_gb": max(vram or [0.0]),
        "flags": flags,
        "control_score": control_score,
        "fullness_score": fullness_score,
    }


def write_markdown(path: Path, summaries: list[dict], excluded: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed Generation Runs Without Mixed-SFT 500 Outlier",
        "",
        "This comparison excludes the old `mixed_sft_500_keeper` run by request.",
        "",
        "## Excluded",
        "",
    ]
    lines.extend(f"- `{item}`" for item in excluded)
    lines.extend(
        [
            "",
            "## Ranked Snapshot",
            "",
            "| Rank | Run | Control | Fullness | Avg lines | Avg words | Slur prompts | Hook cap | Flags |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    ranked = sorted(summaries, key=lambda item: (item["control_score"], item["fullness_score"]), reverse=True)
    for index, item in enumerate(ranked, start=1):
        flags = ", ".join(f"{key}:{value}" for key, value in item["flags"].items() if value) or "none"
        lines.append(
            f"| {index} | `{item['label']}` | {item['control_score']} | {item['fullness_score']} | "
            f"{item['avg_lines']} | {item['avg_words']} | {item['slur_prompt_count']} | {item['hook_line_cap']} | {flags} |"
        )
    lines.extend(["", "## Details", ""])
    for item in ranked:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Path: `{item['path']}`",
                f"- Records: {item['records']}",
                f"- Line counts: {item['line_counts']}",
                f"- Slur counts: {item['slur_counts']}",
                f"- Exact line matches: {item['exact_line_match_count']}",
                f"- Hook cap: {item['hook_line_cap']}",
                f"- Avg repeated-line ratio: {item['avg_repeated_line_ratio']}",
                f"- Avg generation speed: {item['avg_tokens_per_second']} tok/s",
                f"- Peak VRAM: {item['peak_vram_gb']} GB",
                "",
            ]
        )
    lines.extend(
        [
            "## Read",
            "",
            "After removing the old mixed-SFT checkpoint-500 outlier, the current field is not yet dominated by a clear keeper. "
            "Slur counts are tracked for visibility but are not used as a fail condition or ranking penalty. "
            "The newer OpenAI-labeled full-song runs are fuller and more consistent than the tiny probes, but they still miss exact line-count control and compact hook behavior. "
            "The main remaining failures are hook over-expansion, ad-lib/dialogue residue, and endings that can feel cut off.",
            "",
            "The next useful experiment should target dataset composition and prompt conditioning, not just more steps: add stricter hook examples, line-count-conditioned prompts, and stronger rejection of dialogue/ad-lib clutter.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=20)
    args = parser.parse_args()

    excluded: list[str] = []
    summaries: list[dict] = []
    for path in sorted(args.reports_dir.glob("generation_eval*.jsonl")):
        if any(token in path.name for token in args.exclude):
            excluded.append(path.name)
            continue
        rows = read_rows(path)
        if len(rows) > args.max_records:
            continue
        summaries.append(summarize(path))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summaries, excluded)
    print(json.dumps({"summaries": len(summaries), "excluded": excluded, "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
