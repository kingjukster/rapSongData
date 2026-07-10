"""Generate and score multiple Runpod lyric samples for debugging."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from rap_song_data.integrations.slack import notify_slack, slack_webhook_from_env


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
BAD_ARTIFACT_RE = re.compile(
    r"lyrics taken from|genius\.com|https?://|source:|you might also like|\bembed\b|"
    r"<\|\s*(?:begin|end|bar|verse)|\{\s*(?:intro|verse|chorus|hook|bridge|outro)|"
    r"\*[^*\n]{1,80}\*|<tool_call>|</tool_call>|<\|im_(?:start|end)\|>|<\s*br\s*/?\s*>|[^\x00-\x7F]{2,}",
    re.IGNORECASE,
)
BRACKET_LABEL_RE = re.compile(r"^\s*\[(intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus).*?\]\s*$", re.I)
CURLY_LABEL_RE = re.compile(r"^\s*\{(?:intro|verse|chorus|hook|bridge|outro|pre[- ]?chorus).*?\}\s*$", re.I)
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.I)
SIMILE_RE = re.compile(r"\b(?:like|as)\b", re.I)
PROFANITY_RE = re.compile(
    r"\b(?:fuck(?:in|ing)?|shit|bitch(?:es)?|hoe(?:s)?|nigga(?:s)?|nigger(?:s)?|dick|pussy|ass|damn)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-repo", default="kingjukster77/rap-lyrics-lora-adapter")
    parser.add_argument("--adapter-subfolder", default=None)
    parser.add_argument("--add-structural-special-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--title", default="Night Shift")
    parser.add_argument("--artist", default="Original")
    parser.add_argument("--rap-family", default="Melodic / Emo / Cloud")
    parser.add_argument("--rap-category", default="Emo Rap / Melodic Rap")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--structure", default="12-line verse, no chorus, no bracket labels")
    parser.add_argument("--theme", default="working late, loneliness, ambition, city lights")
    parser.add_argument("--keywords", default="night, shift, city, lights, work, ambition")
    parser.add_argument("--rules", default="Write only original lyrics. Keep line breaks. Do not explain. Stay focused on the requested theme and keywords. Avoid random sexual content, threats, unrelated violent imagery, parenthetical ad-libs, bracket labels, dialogue, URLs, source notes, and copied song text.")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--target-bars", type=int, default=16)
    parser.add_argument("--bar-count-range", default="10-25")
    parser.add_argument("--min-bars", type=int, default=10)
    parser.add_argument("--max-bars", type=int, default=25)
    parser.add_argument("--ideal-min-bars", type=int, default=12)
    parser.add_argument("--ideal-max-bars", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("model/reports/generation_eval.json"))
    parser.add_argument("--runpod-endpoint-id", default="5gbees2jnwrg84")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--poll-timeout-seconds", type=int, default=600)
    parser.add_argument("--remote-batch-size", type=int, default=5)
    parser.add_argument("--slack-webhook-url", default=None)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def line_end_rhyme_key(line: str) -> str:
    tokens = words(line)
    if not tokens:
        return ""
    word = tokens[-1]
    return word[-3:] if len(word) >= 3 else word


def repeated_line_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"[^a-z0-9]+", " ", line.lower()).strip() for line in lines if line.strip()]
    if not normalized:
        return 0.0
    counts = {line: normalized.count(line) for line in set(normalized)}
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(normalized)


def count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_RE.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def line_syllables(line: str) -> int:
    return sum(count_syllables(token) for token in words(line))


def rhyme_tail(token: str, chars: int = 4) -> str:
    clean = re.sub(r"[^a-z0-9]", "", token.lower())
    if len(clean) <= chars:
        return clean
    return clean[-chars:]


def rhyme_signature(line: str, tail_words: int = 2) -> str:
    tokens = words(line)
    if not tokens:
        return ""
    return " ".join(rhyme_tail(token, 3) for token in tokens[-tail_words:])


def multi_syllabic_match_count(lines: list[str]) -> int:
    signatures = [rhyme_signature(line) for line in lines]
    count = 0
    for index, signature in enumerate(signatures):
        if not signature:
            continue
        nearby = signatures[index + 1 : index + 3]
        if signature in nearby:
            count += 1
    return count


def internal_rhyme_density(lines: list[str]) -> float:
    matched_lines = 0
    usable_lines = 0
    for line in lines:
        tokens = [token for token in words(line) if len(token) > 3]
        if len(tokens) < 5:
            continue
        usable_lines += 1
        tails = [rhyme_tail(token, 3) for token in tokens]
        if any(tails.count(tail) >= 2 for tail in set(tails)):
            matched_lines += 1
    return matched_lines / usable_lines if usable_lines else 0.0


def profanity_spam_metrics(text: str, token_list: list[str]) -> dict[str, Any]:
    terms = [match.group(0).lower() for match in PROFANITY_RE.finditer(text)]
    counts = {term: terms.count(term) for term in set(terms)}
    per_100 = len(terms) / max(1, len(token_list)) * 100
    return {
        "profanity_count": len(terms),
        "profanity_per_100_words": round(per_100, 2),
        "max_repeated_profanity": max(counts.values()) if counts else 0,
    }


def score_sample(lyrics: str, theme: str, keywords: str) -> dict[str, Any]:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    token_list = words(lyrics)
    avg_line_len = sum(len(line) for line in lines) / len(lines) if lines else 0.0
    theme_terms = {term for term in words(theme) if len(term) > 3}
    keyword_terms = {term for term in words(keywords) if len(term) > 3}
    lyric_terms = set(token_list)
    rhyme_keys = [line_end_rhyme_key(line) for line in lines if line_end_rhyme_key(line)]
    rhyme_counts = {key: rhyme_keys.count(key) for key in set(rhyme_keys)}
    repeated_rhymes = sum(count for count in rhyme_counts.values() if count > 1)
    syllables = [line_syllables(line) for line in lines]
    avg_syllables = sum(syllables) / len(syllables) if syllables else 0.0
    syllable_std = math.sqrt(sum((value - avg_syllables) ** 2 for value in syllables) / len(syllables)) if syllables else 0.0
    unique_words = {token for token in token_list if len(token) > 2}
    type_token_ratio = len(unique_words) / len(token_list) if token_list else 0.0
    profanity = profanity_spam_metrics(lyrics, token_list)
    return {
        "line_count": len(lines),
        "word_count": len(token_list),
        "avg_line_length": round(avg_line_len, 2),
        "avg_line_syllables": round(avg_syllables, 2),
        "std_dev_line_syllables": round(syllable_std, 2),
        "type_token_ratio": round(type_token_ratio, 4),
        "repeated_line_ratio": round(repeated_line_ratio(lines), 4),
        "bad_artifact_count": len(BAD_ARTIFACT_RE.findall(lyrics)),
        "bracket_label_count": sum(1 for line in lines if BRACKET_LABEL_RE.match(line) or CURLY_LABEL_RE.match(line)),
        "theme_keyword_overlap": sorted(theme_terms & lyric_terms),
        "theme_overlap_count": len(theme_terms & lyric_terms),
        "keyword_overlap": sorted(keyword_terms & lyric_terms),
        "keyword_overlap_count": len(keyword_terms & lyric_terms),
        "rhyme_density_estimate": round(repeated_rhymes / len(rhyme_keys), 4) if rhyme_keys else 0.0,
        "multi_syllabic_match_count": multi_syllabic_match_count(lines),
        "internal_rhyme_density": round(internal_rhyme_density(lines), 4),
        "simile_line_count": sum(1 for line in lines if SIMILE_RE.search(line)),
        **profanity,
    }


def parameter_grid(samples: int) -> list[dict[str, Any]]:
    base = [
        {"temperature": 0.65, "top_p": 0.82, "repetition_penalty": 1.35, "no_repeat_ngram_size": 5},
        {"temperature": 0.75, "top_p": 0.85, "repetition_penalty": 1.35, "no_repeat_ngram_size": 5},
        {"temperature": 0.85, "top_p": 0.88, "repetition_penalty": 1.3, "no_repeat_ngram_size": 6},
        {"temperature": 0.95, "top_p": 0.9, "repetition_penalty": 1.28, "no_repeat_ngram_size": 6},
    ]
    grid = []
    for index in range(samples):
        params = dict(base[index % len(base)])
        params["seed"] = 42 + index
        params["top_k"] = 50
        grid.append(params)
    return grid


def base_generation_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rap_song_data.generation.runpod_flash",
        "--adapter-repo",
        args.adapter_repo,
    ]
    if args.adapter_subfolder:
        command.extend(["--adapter-subfolder", args.adapter_subfolder])
    command.extend([
        "--title",
        args.title,
        "--artist",
        args.artist,
        "--rap-family",
        args.rap_family,
        "--rap-category",
        args.rap_category,
        "--year",
        str(args.year),
        "--structure",
        args.structure,
        "--theme",
        args.theme,
        "--keywords",
        args.keywords,
        "--rules",
        args.rules,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--target-bars",
        str(args.target_bars),
        "--target-lines",
        str(args.target_bars),
        "--bar-count-range",
        args.bar_count_range,
    ])
    return command


def run_generation(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    result = runpod_generate(args, params)
    if result.get("lyrics"):
        return result
    raise RuntimeError(f"Generation returned no lyrics: {result}")


def run_generation_batch(args: argparse.Namespace, params_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = runpod_generate(args, {"batch": params_batch})
    samples = result.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError(f"Batch generation returned no samples: {result}")
    return samples


def runpod_generate(args: argparse.Namespace, overrides: dict[str, Any]) -> dict[str, Any]:
    api_key = runpod_api_key()
    hub_token = hf_token()
    config = {
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "add_structural_special_tokens": args.add_structural_special_tokens,
        "hf_token": hub_token,
        "slack_webhook_url": "",
        "title": args.title,
        "artist": args.artist,
        "rap_family": args.rap_family,
        "rap_category": args.rap_category,
        "year": args.year,
        "structure": args.structure,
        "theme": args.theme,
        "keywords": args.keywords,
        "rules": args.rules,
        "max_new_tokens": args.max_new_tokens,
        "target_bars": args.target_bars,
        "target_lines": args.target_bars,
        "bar_count_range": args.bar_count_range,
        "max_words_per_line": 9,
        **overrides,
    }
    response = requests.post(
        f"https://api.runpod.ai/v2/{args.runpod_endpoint_id}/runsync",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": {"job_config": config}},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("lyrics") or result.get("samples"):
        return result
    if result.get("output"):
        output = result["output"]
        if output.get("lyrics") or output.get("samples"):
            return output
    if result.get("id") and result.get("status") in {"IN_QUEUE", "IN_PROGRESS"}:
        return poll_runpod_job(args, result["id"])
    raise RuntimeError(f"Generation returned no lyrics: {result}")


def env_file_value(name: str) -> str:
    path = Path(".env")
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def runpod_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY") or env_file_value("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("Missing RUNPOD_API_KEY for generation evaluation.")
    return api_key


def hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or env_file_value("HF_TOKEN")
        or env_file_value("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise RuntimeError("Missing HF_TOKEN or HUGGINGFACE_HUB_TOKEN for remote adapter access.")
    return token


def poll_runpod_job(args: argparse.Namespace, job_id: str) -> dict[str, Any]:
    api_key = runpod_api_key()
    url = f"https://api.runpod.ai/v2/{args.runpod_endpoint_id}/status/{job_id}"
    deadline = time.monotonic() + args.poll_timeout_seconds
    while time.monotonic() < deadline:
        response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        if status == "COMPLETED":
            output = data.get("output") or {}
            if output.get("lyrics") or output.get("samples"):
                return output
            raise RuntimeError(f"Completed generation had no lyrics or samples: {data}")
        if status not in {"IN_QUEUE", "IN_PROGRESS"}:
            raise RuntimeError(f"Generation job ended with status {status}: {data}")
        time.sleep(args.poll_seconds)
    raise TimeoutError(f"Timed out waiting for generation job {job_id}")


def line_count_score(line_count: int, min_bars: int = 10, max_bars: int = 25, ideal_min: int = 12, ideal_max: int = 16) -> float:
    if ideal_min <= line_count <= ideal_max:
        return 22.0
    if min_bars <= line_count <= max_bars:
        distance = min(abs(line_count - ideal_min), abs(line_count - ideal_max))
        return max(16.0, 22.0 - distance * 1.5)
    if line_count < min_bars:
        return max(0.0, 16.0 - (min_bars - line_count) * 3.0)
    return max(0.0, 16.0 - (line_count - max_bars) * 2.0)


def sample_score(metrics: dict[str, Any], args: argparse.Namespace | None = None) -> float:
    if args:
        line_score = line_count_score(
            metrics["line_count"],
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            ideal_min=args.ideal_min_bars,
            ideal_max=args.ideal_max_bars,
        )
    else:
        line_score = line_count_score(metrics["line_count"])
    flow_score = max(0, 15 - abs(metrics["std_dev_line_syllables"] - 2.5) * 4)
    richness_score = min(18, metrics["type_token_ratio"] * 24)
    rhyme_score = (
        metrics["multi_syllabic_match_count"] * 10
        + metrics["internal_rhyme_density"] * 12
        + metrics["rhyme_density_estimate"] * 8
    )
    figurative_score = min(6, metrics["simile_line_count"] * 2)
    theme_score = min(20, metrics["theme_overlap_count"] * 4 + metrics["keyword_overlap_count"] * 3)
    artifact_penalty = metrics["bad_artifact_count"] * 25 + metrics["bracket_label_count"] * 10
    repeat_penalty = metrics["repeated_line_ratio"] * 30
    length_penalty = max(0, metrics["avg_line_length"] - 95) * 0.25
    profanity_spam_penalty = max(0, metrics["profanity_count"] - 4) * 2
    profanity_spam_penalty += max(0, metrics["max_repeated_profanity"] - 2) * 4
    profanity_spam_penalty += max(0, metrics["profanity_per_100_words"] - 8) * 1.5
    return round(
        line_score
        + flow_score
        + richness_score
        + rhyme_score
        + figurative_score
        + theme_score
        - artifact_penalty
        - repeat_penalty
        - length_penalty
        - profanity_spam_penalty,
        2,
    )


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if "metrics" in item]
    failures = [item for item in results if "error" in item]
    scores = [item["score"] for item in completed]
    line_counts = [item["metrics"]["line_count"] for item in completed]
    def avg_metric(name: str) -> float:
        return round(sum(item["metrics"][name] for item in completed) / len(completed), 2) if completed else 0

    return {
        "completed": len(completed),
        "failed": len(failures),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "best_score": max(scores) if scores else 0,
        "best_sample": max(completed, key=lambda item: item["score"])["sample"] if completed else None,
        "avg_lines": round(sum(line_counts) / len(line_counts), 2) if line_counts else 0,
        "avg_syllable_std": avg_metric("std_dev_line_syllables"),
        "avg_type_token_ratio": avg_metric("type_token_ratio"),
        "avg_internal_rhyme_density": avg_metric("internal_rhyme_density"),
        "avg_multi_syllabic_matches": avg_metric("multi_syllabic_match_count"),
        "avg_profanity_count": avg_metric("profanity_count"),
        "avg_profanity_per_100_words": avg_metric("profanity_per_100_words"),
        "artifact_total": sum(item["metrics"]["bad_artifact_count"] for item in completed),
        "bracket_label_total": sum(item["metrics"]["bracket_label_count"] for item in completed),
        "avg_theme_overlap": round(sum(item["metrics"]["theme_overlap_count"] for item in completed) / len(completed), 2) if completed else 0,
        "avg_keyword_overlap": round(sum(item["metrics"]["keyword_overlap_count"] for item in completed) / len(completed), 2) if completed else 0,
    }


def write_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(results)
    payload = {
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "summary": summary,
        "samples": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    webhook_url = args.slack_webhook_url if args.slack_webhook_url is not None else slack_webhook_from_env()
    adapter_label = args.adapter_subfolder or "final"
    notify_slack(
        webhook_url,
        "Rap generation sweep started",
        {"samples": args.samples, "output": str(args.output), "adapter": adapter_label},
    )
    grid = parameter_grid(args.samples)
    batch_size = max(1, args.remote_batch_size)
    for batch_start in range(0, len(grid), batch_size):
        params_batch = grid[batch_start : batch_start + batch_size]
        try:
            generated_batch = run_generation_batch(args, params_batch) if batch_size > 1 else [
                run_generation(args, params_batch[0])
            ]
        except Exception as exc:
            for offset, params in enumerate(params_batch):
                sample_index = batch_start + offset + 1
                error = str(exc)
                results.append({"sample": sample_index, "params": params, "error": error})
                print(json.dumps({"sample": sample_index, "params": params, "error": error}, ensure_ascii=False))
            write_report(args, results)
            continue

        for offset, params in enumerate(params_batch):
            sample_index = batch_start + offset + 1
            try:
                result = generated_batch[offset]
                lyrics = result["lyrics"]
                metrics = score_sample(lyrics, args.theme, args.keywords)
                score = sample_score(metrics, args)
                results.append({"sample": sample_index, "params": params, "score": score, "metrics": metrics, "lyrics": lyrics})
                print(json.dumps({"sample": sample_index, "params": params, "score": score, "metrics": metrics}, ensure_ascii=False))
            except Exception as exc:
                error = str(exc)
                results.append({"sample": sample_index, "params": params, "error": error})
                print(json.dumps({"sample": sample_index, "params": params, "error": error}, ensure_ascii=False))
        summary = write_report(args, results)
        notify_slack(
            webhook_url,
            "Rap generation sweep progress",
            {
                **summary,
                "output": str(args.output),
                "adapter": adapter_label,
                "processed": len(results),
                "target": args.samples,
            },
        )

    summary = write_report(args, results)
    notify_slack(webhook_url, "Rap generation sweep completed", {**summary, "output": str(args.output), "adapter": adapter_label})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
