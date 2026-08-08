"""Combine JSONL files without changing row contents."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output.open("w", encoding="utf-8") as dst:
        for path in args.input:
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        dst.write(line)
                        rows += 1
    print({"output": str(args.output), "rows_written": rows, "inputs": [str(p) for p in args.input]})


if __name__ == "__main__":
    main()
