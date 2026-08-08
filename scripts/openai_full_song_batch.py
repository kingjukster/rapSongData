"""Create, submit, monitor, and materialize OpenAI Batch jobs for full-song splitting."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

from split_full_songs_with_openai import (
    compact_song,
    iter_parquet_rows,
    parse_response_json,
    request_payload,
    response_text,
    song_ok,
    text_from_range,
)


FILES_URL = "https://api.openai.com/v1/files"
BATCHES_URL = "https://api.openai.com/v1/batches"
DEFAULT_MODEL = "gpt-5.4-mini"


def request_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def json_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def load_manifest_keys(path: Path, *, offset: int, limit: int | None) -> dict[str, int]:
    keys: dict[str, int] = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if skipped < offset:
                skipped += 1
                continue
            row = json.loads(line)
            key = row.get("song_key")
            if isinstance(key, str) and key and key not in keys:
                keys[key] = offset + len(keys) + 1
            if limit is not None and len(keys) >= limit:
                break
    return keys


def iter_selected_songs(
    *,
    input_path: Path,
    manifest_keys: dict[str, int],
    read_batch_size: int,
    max_chars: int,
) -> Iterable[dict[str, Any]]:
    columns = ["title", "tag", "artist", "artist_clean", "year", "views", "id", "language_cld3", "language_ft", "language", "lyrics"]
    remaining = set(manifest_keys)
    for row in iter_parquet_rows(input_path, batch_size=read_batch_size, columns=columns):
        if not remaining:
            break
        lyrics = str(row.get("lyrics") or "")
        key = str(row.get("id") or "")
        if not key:
            continue
        if key not in remaining:
            continue
        if not song_ok(row):
            remaining.remove(key)
            continue
        song = compact_song(row, max_chars=max_chars)
        song["manifest_rank"] = manifest_keys[key]
        remaining.remove(key)
        yield song


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def cmd_build_requests(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        args.limit = None
    manifest_keys = load_manifest_keys(args.manifest, offset=args.manifest_offset, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 1
    shard_requests = 0
    shard_bytes = 0
    total_requests = 0
    shard_path = args.output_dir / f"{args.name}_shard_{shard_index:04d}.jsonl"
    shard_handle = shard_path.open("w", encoding="utf-8")
    shards: list[dict[str, Any]] = []
    max_bytes = int(args.max_file_mb * 1024 * 1024)
    source_rows: list[dict[str, Any]] = []

    def close_shard() -> None:
        nonlocal shard_handle, shard_path, shard_requests, shard_bytes
        shard_handle.close()
        if shard_requests > 0:
            shards.append({"path": str(shard_path), "requests": shard_requests, "bytes": shard_bytes})

    for song in iter_selected_songs(
        input_path=args.input,
        manifest_keys=manifest_keys,
        read_batch_size=args.read_batch_size,
        max_chars=args.max_chars,
    ):
        clean_song = {key: value for key, value in song.items() if not key.startswith("_") and key != "numbered_lyrics"}
        custom_id = f"rawfullsong:{song['song_key']}"
        batch_row = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": request_payload([song], model=args.model, max_output_tokens=args.max_output_tokens),
        }
        encoded = json.dumps(batch_row, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded_bytes = len(encoded.encode("utf-8"))
        if shard_requests > 0 and (
            shard_requests >= args.max_requests_per_file or shard_bytes + encoded_bytes > max_bytes
        ):
            close_shard()
            shard_index += 1
            shard_path = args.output_dir / f"{args.name}_shard_{shard_index:04d}.jsonl"
            shard_handle = shard_path.open("w", encoding="utf-8")
            shard_requests = 0
            shard_bytes = 0
        shard_handle.write(encoded)
        shard_requests += 1
        shard_bytes += encoded_bytes
        total_requests += 1
        source_rows.append({"custom_id": custom_id, **clean_song})
        if args.progress_every and total_requests % args.progress_every == 0:
            print(json.dumps({"requests": total_requests, "current_shard": shard_index}, ensure_ascii=False), flush=True)

    close_shard()
    source_map_path = args.output_dir / f"{args.name}_source_map.jsonl"
    write_jsonl(source_map_path, source_rows)
    summary = {
        "name": args.name,
        "manifest": str(args.manifest),
        "input": str(args.input),
        "manifest_offset": args.manifest_offset,
        "limit": args.limit,
        "selected_manifest_keys": len(manifest_keys),
        "requests": total_requests,
        "source_map": str(source_map_path),
        "shards": shards,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "max_chars": args.max_chars,
        "endpoint": "/v1/responses",
    }
    summary_path = args.output_dir / f"{args.name}_batch_request_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def upload_file(path: Path, *, api_key: str) -> dict[str, Any]:
    with path.open("rb") as file_handle:
        response = requests.post(
            FILES_URL,
            headers=request_headers(api_key),
            files={"file": (path.name, file_handle, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def create_batch(*, file_id: str, api_key: str, metadata: dict[str, str]) -> dict[str, Any]:
    response = requests.post(
        BATCHES_URL,
        headers=json_headers(api_key),
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": metadata,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def cmd_submit(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    job_log_path = args.job_log or (Path(summary["shards"][0]["path"]).parent / f"{summary['name']}_batch_jobs.jsonl")
    existing_paths: set[str] = set()
    if job_log_path.exists() and args.resume:
        with job_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_paths.add(str(json.loads(line).get("request_path")))
    submitted = 0
    with job_log_path.open("a", encoding="utf-8") as handle:
        for shard in summary["shards"][args.start_shard_index :]:
            if args.max_shards is not None and submitted >= args.max_shards:
                break
            shard_path = Path(shard["path"])
            if str(shard_path) in existing_paths:
                continue
            upload = upload_file(shard_path, api_key=api_key)
            batch = create_batch(
                file_id=upload["id"],
                api_key=api_key,
                metadata={
                    "name": str(summary["name"]),
                    "request_path": str(shard_path),
                    "manifest_offset": str(summary.get("manifest_offset")),
                    "limit": str(summary.get("limit")),
                },
            )
            row = {
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_path": str(shard_path),
                "requests": shard["requests"],
                "bytes": shard["bytes"],
                "file": upload,
                "batch": batch,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            submitted += 1
            print(json.dumps({"submitted": submitted, "request_path": str(shard_path), "batch_id": batch.get("id")}, ensure_ascii=False), flush=True)
    print(json.dumps({"job_log": str(job_log_path), "submitted": submitted}, indent=2))


def get_batch(batch_id: str, *, api_key: str) -> dict[str, Any]:
    response = requests.get(f"{BATCHES_URL}/{batch_id}", headers=request_headers(api_key), timeout=120)
    response.raise_for_status()
    return response.json()


def cmd_status(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    rows = []
    with args.job_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            batch = get_batch(row["batch"]["id"], api_key=api_key)
            rows.append(
                {
                    "request_path": row.get("request_path"),
                    "requests": row.get("requests"),
                    "batch_id": batch.get("id"),
                    "status": batch.get("status"),
                    "request_counts": batch.get("request_counts"),
                    "output_file_id": batch.get("output_file_id"),
                    "error_file_id": batch.get("error_file_id"),
                    "created_at": batch.get("created_at"),
                    "completed_at": batch.get("completed_at"),
                }
            )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def download_file(file_id: str, *, api_key: str, output_path: Path) -> None:
    response = requests.get(f"{FILES_URL}/{file_id}/content", headers=request_headers(api_key), timeout=300)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def cmd_download(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    with args.job_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            job = json.loads(line)
            batch = get_batch(job["batch"]["id"], api_key=api_key)
            if batch.get("status") != "completed":
                continue
            for kind in ["output", "error"]:
                file_id = batch.get(f"{kind}_file_id")
                if not file_id:
                    continue
                output_path = args.output_dir / f"{batch['id']}_{kind}.jsonl"
                if output_path.exists() and not args.overwrite:
                    continue
                download_file(file_id, api_key=api_key, output_path=output_path)
                downloaded.append(str(output_path))
    print(json.dumps({"downloaded": downloaded}, indent=2, ensure_ascii=False))


def iter_batch_outputs(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_completed_bodies(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    bodies: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row in iter_batch_outputs(paths):
        custom_id = str(row.get("custom_id") or "")
        response = row.get("response") or {}
        body = response.get("body") if isinstance(response, dict) else None
        if isinstance(body, dict) and int(response.get("status_code") or 0) == 200:
            bodies[custom_id] = body
        else:
            errors.append(row)
    return bodies, errors


def cmd_materialize(args: argparse.Namespace) -> None:
    output_paths = sorted(args.batch_output_dir.glob("*_output.jsonl"))
    bodies, errors = load_completed_bodies(output_paths)
    keys = {custom_id.split(":", 1)[1] for custom_id in bodies if ":" in custom_id}
    source_by_key: dict[str, dict[str, Any]] = {}
    columns = ["title", "tag", "artist", "artist_clean", "year", "views", "id", "language_cld3", "language_ft", "language", "lyrics"]
    for row in iter_parquet_rows(args.input, batch_size=args.read_batch_size, columns=columns):
        key = str(row.get("id") or "")
        if key in keys:
            source_by_key[key] = compact_song(row, max_chars=args.max_chars)
            if len(source_by_key) >= len(keys):
                break

    args.output_sections.parent.mkdir(parents=True, exist_ok=True)
    args.output_songs.parent.mkdir(parents=True, exist_ok=True)
    written_sections = 0
    written_songs = 0
    materialize_errors = list(errors)
    with args.output_sections.open("w", encoding="utf-8") as section_handle, args.output_songs.open("w", encoding="utf-8") as song_handle:
        for custom_id, body in bodies.items():
            key = custom_id.split(":", 1)[1] if ":" in custom_id else custom_id
            source = source_by_key.get(key)
            if not source:
                materialize_errors.append({"custom_id": custom_id, "error": "source_not_found"})
                continue
            try:
                parsed = parse_response_json(body)
            except Exception as exc:  # noqa: BLE001
                materialize_errors.append({"custom_id": custom_id, "error": str(exc), "response_text": response_text(body)[:8000]})
                continue
            clean_source = {field: value for field, value in source.items() if not field.startswith("_") and field != "numbered_lyrics"}
            for song_result in parsed.get("songs") or []:
                song_key = str(song_result.get("song_key") or key)
                song_row = {**clean_source, **song_result, "custom_id": custom_id, "response_id": body.get("id")}
                song_handle.write(json.dumps(song_row, ensure_ascii=False) + "\n")
                written_songs += 1
                for section_index, section in enumerate(song_result.get("sections") or []):
                    section_text = text_from_range(source, int(section.get("start_line") or 1), int(section.get("end_line") or 1))
                    if not section_text:
                        continue
                    record = {
                        **clean_source,
                        **section,
                        "text": section_text,
                        "song_key": song_key,
                        "section_index": section_index,
                        "record_id": f"song:{song_key}:openai_batch_section:{section_index}:{custom_id.replace(':', '_')}",
                        "custom_id": custom_id,
                        "response_id": body.get("id"),
                    }
                    section_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written_sections += 1
    if materialize_errors:
        write_jsonl(args.output_errors, materialize_errors)
    summary = {
        "batch_output_dir": str(args.batch_output_dir),
        "response_count": len(bodies),
        "source_found": len(source_by_key),
        "song_reviews": written_songs,
        "sections": written_sections,
        "errors": len(materialize_errors),
        "outputs": {
            "sections": str(args.output_sections),
            "songs": str(args.output_songs),
            "errors": str(args.output_errors),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-requests")
    build.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    build.add_argument("--manifest", type=Path, default=Path("data/manifests/full_song_openai_ranked_manifest_top250k.jsonl"))
    build.add_argument("--output-dir", type=Path, default=Path("data/openai_batch/full_song_top250k"))
    build.add_argument("--name", default="full_song_top250k")
    build.add_argument("--manifest-offset", type=int, default=0)
    build.add_argument("--limit", type=int, default=1000)
    build.add_argument("--model", default=os.environ.get("OPENAI_FULL_SONG_MODEL", DEFAULT_MODEL))
    build.add_argument("--max-chars", type=int, default=10000)
    build.add_argument("--max-output-tokens", type=int, default=4000)
    build.add_argument("--max-file-mb", type=float, default=180.0)
    build.add_argument("--max-requests-per-file", type=int, default=50000)
    build.add_argument("--read-batch-size", type=int, default=8192)
    build.add_argument("--progress-every", type=int, default=1000)
    build.set_defaults(func=cmd_build_requests)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--summary", type=Path, required=True)
    submit.add_argument("--job-log", type=Path, default=None)
    submit.add_argument("--max-shards", type=int, default=1)
    submit.add_argument("--start-shard-index", type=int, default=0)
    submit.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    submit.set_defaults(func=cmd_submit)

    status = subparsers.add_parser("status")
    status.add_argument("--job-log", type=Path, required=True)
    status.set_defaults(func=cmd_status)

    download = subparsers.add_parser("download")
    download.add_argument("--job-log", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    download.set_defaults(func=cmd_download)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--input", type=Path, default=Path("data/rap_songs_with_lyrics.parquet"))
    materialize.add_argument("--batch-output-dir", type=Path, required=True)
    materialize.add_argument("--output-sections", type=Path, required=True)
    materialize.add_argument("--output-songs", type=Path, required=True)
    materialize.add_argument("--output-errors", type=Path, required=True)
    materialize.add_argument("--summary-output", type=Path, required=True)
    materialize.add_argument("--read-batch-size", type=int, default=8192)
    materialize.add_argument("--max-chars", type=int, default=10000)
    materialize.set_defaults(func=cmd_materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
