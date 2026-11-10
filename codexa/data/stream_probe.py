"""Explicitly bounded development probe of a pinned published stream."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, BinaryIO

from codexa.data.streaming import StreamingData, open_provider
from codexa.tokenizer.bpe import CodexaTokenizer


class ByteBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def wrap(self, raw: BinaryIO) -> BudgetReader:
        return BudgetReader(raw, self)


class BudgetReader:
    def __init__(self, raw: BinaryIO, budget: ByteBudget):
        self.raw, self.budget = raw, budget

    def read(self, size: int = -1) -> bytes:
        remaining = self.budget.limit - self.budget.used
        if remaining <= 0:
            raise RuntimeError("Development stream byte budget exhausted")
        data = self.raw.read(remaining if size < 0 else min(size, remaining))
        self.budget.used += len(data)
        return data

    def close(self) -> None:
        self.raw.close()


def validate_config(config: dict[str, Any]) -> list[dict[str, str]]:
    if set(config) != {"dataset", "revision", "shards"}:
        raise ValueError("Stream config requires dataset, revision and shards")
    repo, revision, entries = config["dataset"], config["revision"], config["shards"]
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid dataset repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Revision must be a full commit SHA")
    if not isinstance(entries, list) or not entries:
        raise ValueError("At least one pinned shard is required")
    shards = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "etag"}:
            raise ValueError("Each shard needs a path and ETag")
        path, etag = entry["path"], entry["etag"]
        if (
            not isinstance(path, str)
            or not re.fullmatch(r"[A-Za-z0-9_./-]+\.jsonl?\.gz", path)
            or path.startswith("/")
            or any(p in {"", ".", ".."} for p in path.split("/"))
        ):
            raise ValueError("Invalid gzip JSON shard path")
        if not isinstance(etag, str) or not re.fullmatch(r"[0-9a-f]{64}", etag):
            raise ValueError("Shard ETag must be a full object hash")
        shards.append(
            {
                "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}",
                "etag": etag,
            }
        )
    if len({s["url"] for s in shards}) != len(shards):
        raise ValueError("Repeated shard")
    return shards


def probe(
    config: dict[str, Any], tokenizer: str, windows: int, sequence: int, max_bytes: int
) -> dict[str, Any]:
    if (
        not 2 <= windows <= 2048
        or not 1 <= sequence <= 8192
        or not 1 <= max_bytes <= 64 * 1024 * 1024
    ):
        raise ValueError("Probe exceeds bounded development limits")
    shards = validate_config(config)
    budget = ByteBudget(max_bytes)
    stream = StreamingData(
        shards,
        CodexaTokenizer.load(tokenizer),
        sequence,
        42,
        "train",
        lambda row: True,
        lambda text: False,
        opener=lambda shard: budget.wrap(open_provider(shard)),  # type: ignore[arg-type,return-value]
        policy_identity="development-only-no-protection-v1",
    )
    start = time.perf_counter()
    try:
        stream.next(1)
        ready = time.perf_counter()
        stream.next(windows - 1)
        end = time.perf_counter()
        state = stream.state_dict()
        return {
            "dataset": config["dataset"],
            "revision": config["revision"],
            "status": "development-only-not-approved",
            "windows": windows,
            "target_tokens": windows * sequence,
            "compressed_bytes": budget.used,
            "first_window_seconds": ready - start,
            "subsequent_seconds": end - ready,
            "subsequent_target_tokens_per_second": (windows - 1) * sequence / (end - ready),
            "records_examined_in_current_shard": state["record"],
            "document_sha256": state["seen"],
            "stream_identity": state["identity"],
        }
    finally:
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            probe(
                json.loads(args.config.read_text()),
                args.tokenizer,
                args.windows,
                args.sequence,
                args.max_bytes,
            ),
            indent=2,
        )
    )
