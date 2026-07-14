from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .common import (
    PRIVATE_RESEARCH_POLICY,
    SPECIAL_TOKENS,
    command_record,
    hash_file,
    iter_jsonl,
    path_manifest,
    read_json,
    utc_now,
    write_json,
)


def text_iterator(path: Path, field: str, *, limit: int | None = None) -> Iterator[str]:
    for index, row in enumerate(iter_jsonl(path)):
        value = str(row.get(field) or "").strip()
        if not value:
            from .corpus import base_document, sft_document

            if field == "base_text":
                value = base_document(str(row.get("title") or ""), row.get("year"), str(row.get("lyrics") or ""))
            elif field == "sft_text":
                value = sft_document(
                    str(row.get("title") or ""),
                    row.get("year"),
                    str(row.get("lyrics") or ""),
                    list(row.get("content_flags") or []),
                )
        if value:
            yield value
        if limit is not None and index + 1 >= limit:
            return


def train_bpe(train_path: Path, output_dir: Path, *, vocab_size: int, limit: int | None = None) -> Any:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError("The cuda extra is required to train and package the tokenizer.") from exc

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(train_path, "base_text", limit=limit), trainer=trainer)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
        model_max_length=512,
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(output_dir)
    return fast


class BinaryShardWriter:
    def __init__(self, output_dir: Path, *, sequence_length: int, shard_tokens: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.shard_tokens = max(sequence_length, shard_tokens // sequence_length * sequence_length)
        self.buffer: list[int] = []
        self.shards: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_written_tokens = 0

    def add(self, token_ids: Iterable[int]) -> None:
        values = list(token_ids)
        if values and max(values) >= 65_536:
            raise ValueError("Tokenizer vocabulary exceeds uint16 shard capacity.")
        self.total_input_tokens += len(values)
        self.buffer.extend(values)
        while len(self.buffer) >= self.shard_tokens:
            self._write(self.buffer[: self.shard_tokens])
            del self.buffer[: self.shard_tokens]

    def _write(self, values: list[int]) -> None:
        index = len(self.shards)
        path = self.output_dir / f"shard-{index:05d}.bin"
        temporary = path.with_suffix(".bin.tmp")
        np.asarray(values, dtype=np.uint16).tofile(temporary)
        os.replace(temporary, path)
        self.total_written_tokens += len(values)
        self.shards.append(
            {
                "path": str(path),
                "tokens": len(values),
                "blocks": len(values) // self.sequence_length,
                "sha256": hash_file(path),
            }
        )

    def finish(self) -> dict[str, Any]:
        usable = len(self.buffer) // self.sequence_length * self.sequence_length
        if usable:
            self._write(self.buffer[:usable])
        discarded = len(self.buffer) - usable
        self.buffer.clear()
        return {
            "input_tokens": self.total_input_tokens,
            "written_tokens": self.total_written_tokens,
            "discarded_tail_tokens": discarded,
            "blocks": self.total_written_tokens // self.sequence_length,
            "shards": self.shards,
        }


def batched(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def tokenize_split(
    tokenizer: Any,
    source: Path,
    output_dir: Path,
    *,
    field: str,
    sequence_length: int,
    shard_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    writer = BinaryShardWriter(output_dir, sequence_length=sequence_length, shard_tokens=shard_tokens)
    documents = 0
    for texts in batched(text_iterator(source, field, limit=limit), 256):
        encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        for token_ids in encoded:
            writer.add(token_ids)
            documents += 1
    summary = writer.finish()
    summary.update({"source": str(source), "field": field, "documents": documents})
    return summary


def train_and_tokenize(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    corpus_manifest = read_json(corpus_dir / "corpus_manifest.json")
    if corpus_manifest.get("source_license") != "unknown" or not corpus_manifest.get("private_research_only"):
        raise ValueError("Scratch corpus policy fields are missing or unexpected.")
    manifest_path = output_dir / "tokenization_manifest.json"
    if manifest_path.exists() and not args.force:
        existing = read_json(manifest_path)
        if existing.get("corpus_manifest_sha256") == hash_file(corpus_dir / "corpus_manifest.json"):
            return existing

    tokenizer = train_bpe(
        corpus_dir / "train.jsonl",
        tokenizer_dir,
        vocab_size=args.vocab_size,
        limit=args.limit,
    )
    if len(tokenizer) > 65_535:
        raise ValueError("Tokenizer is too large for uint16 shards.")
    modes = {"base": "base_text", "sft": "sft_text"}
    splits: dict[str, Any] = {}
    for mode, field in modes.items():
        splits[mode] = {}
        for split in ("train", "validation", "test"):
            splits[mode][split] = tokenize_split(
                tokenizer,
                corpus_dir / f"{split}.jsonl",
                output_dir / "tokenized" / mode / split,
                field=field,
                sequence_length=args.sequence_length,
                shard_tokens=args.shard_tokens,
                limit=args.limit,
            )
    train_tokens = int(splits["base"]["train"]["input_tokens"])
    manifest = {
        **PRIVATE_RESEARCH_POLICY,
        "schema_version": 1,
        "status": "complete",
        "started_at": started_at,
        "ended_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "command": command_record(),
        "corpus_dir": str(corpus_dir),
        "corpus_manifest_sha256": hash_file(corpus_dir / "corpus_manifest.json"),
        "tokenizer": {
            "path": str(tokenizer_dir),
            "vocab_size": len(tokenizer),
            "requested_vocab_size": args.vocab_size,
            "tokenizer_json": path_manifest(tokenizer_dir / "tokenizer.json"),
        },
        "settings": {
            "sequence_length": args.sequence_length,
            "shard_tokens": args.shard_tokens,
            "limit": args.limit,
        },
        "splits": splits,
        "acceptance": {
            "minimum_unique_training_tokens": args.min_train_tokens,
            "unique_training_tokens": train_tokens,
            "token_gate_passed": train_tokens >= args.min_train_tokens,
            "retained_song_gate_passed": bool(
                corpus_manifest.get("acceptance", {}).get("retained_songs_passed")
            ),
            "full_pilot_allowed": train_tokens >= args.min_train_tokens
            and bool(corpus_manifest.get("acceptance", {}).get("retained_songs_passed")),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def add_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scratch/v1"))
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--shard-tokens", type=int, default=8_388_608)
    parser.add_argument("--min-train-tokens", type=int, default=300_000_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
