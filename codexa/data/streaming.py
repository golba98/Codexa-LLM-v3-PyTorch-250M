"""Deterministic, resumable JSONL-gzip streams; no prepared token corpus."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import random
import time
import zlib
from collections.abc import Callable
from typing import Any, BinaryIO, cast
from urllib.request import Request, urlopen

import torch
from torch import Tensor

from codexa.tokenizer.bpe import BOS, EOS, CodexaTokenizer

RecordFilter = Callable[[dict[str, Any]], bool]
Opener = Callable[[dict[str, str]], BinaryIO]


class CompleteResponse:
    """Detect a premature EOF even at a valid concatenated-gzip member boundary."""

    def __init__(self, response: BinaryIO, length: int):
        self.response, self.remaining = response, length

    def read(self, size: int = -1) -> bytes:
        try:
            data = self.response.read(size)
        except http.client.IncompleteRead as error:
            if not error.partial:
                raise
            data = error.partial
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError("Provider exceeded its declared object length")
        if not data and size != 0 and self.remaining:
            raise OSError("Provider ended before its declared object length")
        return data

    def close(self) -> None:
        self.response.close()


def open_provider(shard: dict[str, str]) -> BinaryIO:
    """Open a pinned Hugging Face file and verify its recorded object identity."""
    url = shard["url"]
    if not url.startswith("https://huggingface.co/datasets/") or "/resolve/" not in url:
        raise ValueError("Expected an immutable Hugging Face dataset URL")
    revision = url.split("/resolve/", 1)[1].split("/", 1)[0]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Provider revision must be a full commit SHA")
    response = urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=30)
    if response.status != 200 or not str(response.headers.get("Content-Length", "")).isdigit():
        response.close()
        raise ValueError("A complete pinned provider response is required")
    if response.headers.get("ETag", "").strip('"') != shard["etag"].strip('"'):
        response.close()
        raise ValueError("Provider object identity changed")
    return cast(BinaryIO, CompleteResponse(response, int(response.headers["Content-Length"])))


class StreamingData:
    """Single-pass stream with serializable cursor, buffer and exact-content deduplication.

    The caller supplies deterministic source and protection checks. A missing protection
    check is rejected; passing this class alone does not constitute source approval.
    """

    def __init__(
        self,
        shards: list[dict[str, str]],
        tokenizer: CodexaTokenizer,
        sequence: int,
        seed: int,
        split: str,
        admit: RecordFilter,
        protected: Callable[[str], bool],
        *,
        opener: Opener = open_provider,
        retries: int = 3,
        policy_identity: str = "",
    ) -> None:
        if not shards or sequence < 1 or split not in {"train", "val", "test"} or retries < 0:
            raise ValueError("Invalid stream configuration")
        if not callable(admit) or not callable(protected):
            raise TypeError("Source admission and protected-text checks are required")
        self.shards = [dict(s) for s in shards]
        if any(set(s) != {"url", "etag"} or not all(s.values()) for s in self.shards):
            raise ValueError("Every shard requires a URL and object identity")
        random.Random(seed).shuffle(self.shards)
        self.tokenizer, self.sequence, self.split = tokenizer, sequence, split
        self.admit, self.protected, self.opener, self.retries = admit, protected, opener, retries
        self.tokenizer_identity = hashlib.sha256(tokenizer.backend.to_str().encode()).hexdigest()
        if not policy_identity:
            raise ValueError("A pinned admission and protection policy identity is required")
        self.identity = hashlib.sha256(
            json.dumps(
                {
                    "shards": self.shards,
                    "sequence": sequence,
                    "split": split,
                    "tokenizer": self.tokenizer_identity,
                    "policy": policy_identity,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.shard = self.record = self.windows = 0
        self.launch_check: Callable[[], None] | None = None
        self.required_config_identity: str | None = None
        self.buffer: list[int] = []
        self.seen: set[str] = set()
        self._added: list[str] = []
        self._raw: BinaryIO | None = None
        self._gzip: gzip.GzipFile | None = None

    def close(self) -> None:
        if self._gzip is not None:
            self._gzip.close()
        if self._raw is not None:
            self._raw.close()
        self._gzip = None
        self._raw = None

    def _connect(self) -> None:
        self._raw = self.opener(self.shards[self.shard])
        self._gzip = gzip.GzipFile(fileobj=self._raw)
        for _ in range(self.record):
            if not self._gzip.readline():
                raise ValueError("Pinned shard is shorter than its saved cursor")

    def _record(self) -> dict[str, Any]:
        attempts = 0
        while self.shard < len(self.shards):
            try:
                if self._gzip is None:
                    self._connect()
                assert self._gzip is not None
                line = self._gzip.readline()
                if not line:
                    self.close()
                    self.shard += 1
                    self.record = 0
                    attempts = 0
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise TypeError("Invalid provider record")
                self.record += 1
                return row
            except (OSError, EOFError, TimeoutError, http.client.IncompleteRead, zlib.error):
                self.close()
                if attempts >= self.retries:
                    raise
                time.sleep(min(2**attempts, 8))
                attempts += 1
        raise EOFError("Streaming source exhausted")

    @staticmethod
    def document_split(text_hash: str) -> str:
        slot = int(text_hash[:16], 16) % 1000
        return "val" if slot < 10 else "test" if slot < 20 else "train"

    def _fill(self) -> None:
        while len(self.buffer) < self.sequence + 1:
            row = self._record()
            text = row["text"]
            metadata_check = getattr(self.admit, "metadata_eligible", None)
            if metadata_check and not metadata_check(row):
                continue
            context = getattr(self.admit, "position", None)
            if context:
                context(self.shard, self.record)
            if not text.strip() or self.protected(text) or not self.admit(row):
                continue
            identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
            group = getattr(self.admit, "group", None)
            split_identity = hashlib.sha256(group(row).encode()).hexdigest() if group else identity
            if self.document_split(split_identity) != self.split or identity in self.seen:
                continue
            tokens = self.tokenizer.encode(text)
            if any(token < 4 for token in tokens) or self.tokenizer.decode(tokens) != text:
                raise ValueError(
                    "Provider text violates frozen tokenizer roundtrip or special-ID invariants"
                )
            self.buffer.extend([BOS, *tokens, EOS])
            self.seen.add(identity)
            self._added.append(identity)

    def next(self, count: int) -> list[list[int]]:
        if count < 1:
            raise ValueError("Batch size must be positive")
        rows = []
        previous = (self.shard, self.record, self.windows, list(self.buffer))
        self._added = []
        begin = getattr(self.admit, "begin", None)
        if begin:
            begin()
        try:
            for _ in range(count):
                self._fill()
                rows.append(self.buffer[: self.sequence + 1])
                del self.buffer[: self.sequence]
                self.windows += 1
        except Exception:
            self.close()
            self.shard, self.record, self.windows, self.buffer = previous
            self.seen.difference_update(self._added)
            self._added = []
            rollback = getattr(self.admit, "rollback", None)
            if rollback:
                rollback()
            raise
        commit = getattr(self.admit, "commit", None)
        if commit:
            commit()
        self._added = []
        return rows

    def batch(self, rows: list[list[int]], device: torch.device) -> tuple[Tensor, Tensor]:
        values = torch.tensor(rows, dtype=torch.long, device=device)
        return values[:, :-1], values[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "shard": self.shard,
            "record": self.record,
            "windows": self.windows,
            "buffer": list(self.buffer),
            "seen": sorted(self.seen),
            "admission": self.admit.state_dict() if hasattr(self.admit, "state_dict") else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("identity") != self.identity:
            raise ValueError("Streaming identity changed")
        shard, record = state["shard"], state["record"]
        if (
            not isinstance(shard, int)
            or not 0 <= shard <= len(self.shards)
            or not isinstance(record, int)
            or record < 0
        ):
            raise ValueError("Invalid streaming cursor")
        if (
            type(state.get("windows")) is not int
            or state["windows"] < 0
            or not isinstance(state.get("buffer"), list)
            or any(
                type(t) is not int or not 0 <= t < self.tokenizer.backend.get_vocab_size()
                for t in state["buffer"]
            )
            or not isinstance(state.get("seen"), list)
            or any(
                not isinstance(h, str)
                or len(h) != 64
                or any(c not in "0123456789abcdef" for c in h)
                for h in state["seen"]
            )
        ):
            raise ValueError("Invalid streaming checkpoint state")
        self.close()
        self.shard, self.record, self.windows = shard, record, state["windows"]
        self.buffer, self.seen = list(state["buffer"]), set(state["seen"])
        if hasattr(self.admit, "load_state_dict"):
            self.admit.load_state_dict(state["admission"])
