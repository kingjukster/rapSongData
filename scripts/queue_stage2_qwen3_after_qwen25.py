"""Queue Stage 2 Qwen2.5 runs after the active Stage 2 Qwen2.5 run exits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
QWEN25_DIR = REPO / "model/artifacts/stage2-qwen2.5-7b-cleaned-chunks-512-60m"
WAIT_PID_FILE = QWEN25_DIR / "stage2_active_pid.txt"
QUEUE_LOG = QWEN25_DIR / "stage2_queue.log"
QUEUE_PID_FILE = QWEN25_DIR / "stage2_queue_pid.txt"
COMMAND = [sys.executable, "-u", "scripts/run_stage2_qwen25_60m.py"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--not-before",
        default=None,
        help="Local ISO timestamp before which the queued run must not start, e.g. 2026-06-16T01:30:00.",
    )
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(event: str, **payload: object) -> None:
    QWEN25_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "time": now(), **payload}) + "\n")


def pid_alive(pid: str) -> bool:
    if not pid:
        return False
    try:
        result = subprocess.run(
            ["tasklist.exe", "/FI", f"PID eq {pid}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        log("pid_check_error", pid=pid, error=str(exc))
        return True
    return "python.exe" in result.stdout.lower() and pid in result.stdout


def read_wait_pid() -> str:
    try:
        return WAIT_PID_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def parse_not_before(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid --not-before timestamp: {value}") from exc


def main() -> int:
    args = parse_args()
    not_before = parse_not_before(args.not_before)
    QWEN25_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    log(
        "queue_started",
        command=COMMAND,
        waiting_on_pid_file=str(WAIT_PID_FILE.relative_to(REPO)),
        not_before=not_before.isoformat(timespec="seconds") if not_before else None,
    )

    last_heartbeat = 0.0
    while True:
        wait_pid = read_wait_pid()
        alive = pid_alive(wait_pid)
        time_ready = not_before is None or dt.datetime.now() >= not_before
        if not alive and time_ready:
            log("wait_complete", waited_pid=wait_pid)
            break
        if time.monotonic() - last_heartbeat > 300:
            log(
                "waiting",
                waited_pid=wait_pid,
                waiting_for_qwen25=alive,
                waiting_for_time=not time_ready,
                not_before=not_before.isoformat(timespec="seconds") if not_before else None,
            )
            last_heartbeat = time.monotonic()
        time.sleep(60)

    started = dt.datetime.now()
    log("qwen2.5_start", command=COMMAND)
    with QUEUE_LOG.open("a", encoding="utf-8", buffering=1) as handle:
        proc = subprocess.run(COMMAND, cwd=REPO, stdout=handle, stderr=handle, text=True)
    ended = dt.datetime.now()
    log(
        "qwen2.5_end",
        returncode=proc.returncode,
        wall_seconds=round((ended - started).total_seconds(), 2),
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
