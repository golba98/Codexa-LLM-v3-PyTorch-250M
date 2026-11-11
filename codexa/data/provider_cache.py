"""Bounded read-through cache of immutable compressed provider blocks.

The existing stream and duplicate checker read the same bytes and keep their
native cursors. Cache blocks are disposable and never enter a checkpoint.
"""

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO, cast
from urllib.request import Request, urlopen

from codexa.training.checkpoint import check_volume

from .streaming import CompleteResponse, open_provider


class CachedProvider:
    def __init__(
        self,
        root: Path,
        volume_uuid: str | None = None,
        *,
        maximum_bytes: int = 2 * 1024**3,
        block_bytes: int = 1024**2,
        reserve_bytes: int = 40 * 1024**3,
    ):
        if block_bytes <= 0 or maximum_bytes < block_bytes:
            raise ValueError("Invalid provider cache limits")
        self.root, self.volume_uuid = root, volume_uuid
        self.maximum_bytes, self.block_bytes = maximum_bytes, block_bytes
        self.reserve_bytes = reserve_bytes
        self.memory: OrderedDict[str, bytes] = OrderedDict()
        self.network_bytes = self.cache_hits = self.network_requests = 0

    def initialize(self) -> None:
        check_volume(self.root, self.volume_uuid)
        if self.root.is_symlink():
            raise ValueError("Provider cache cannot redirect through a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.root / "owner.json"
        if owner.exists():
            if owner.is_symlink() or json.loads(owner.read_text()) != {
                "kind": "codexa-provider-block-cache-1",
                "block_bytes": self.block_bytes,
            }:
                raise ValueError("Unrecognized provider cache directory")
        else:
            if any(self.root.iterdir()):
                raise ValueError("Use an empty directory for the provider cache")
            owner.write_text(
                json.dumps(
                    {"kind": "codexa-provider-block-cache-1", "block_bytes": self.block_bytes}
                )
            )

    def __call__(self, shard: dict[str, str]) -> BinaryIO:
        self.initialize()
        key = hashlib.sha256(json.dumps(shard, sort_keys=True).encode()).hexdigest()
        path = self.root / (key + ".json")
        if path.is_symlink():
            raise ValueError("Provider descriptor cannot be a symlink")
        if path.exists():
            descriptor = json.loads(path.read_text())
            if (
                descriptor.get("shard") != shard
                or type(descriptor.get("length")) is not int
                or descriptor["length"] <= 0
            ):
                raise ValueError("Invalid cached provider descriptor")
            length = descriptor["length"]
        else:
            # Reuse the production opener's immutable URL and ETag checks.
            raw = open_provider(shard)
            try:
                length = cast(CompleteResponse, raw).remaining
                if type(length) is not int or length <= 0:
                    raise ValueError("Provider length is unavailable")
            finally:
                raw.close()
            self.publish(path, json.dumps({"shard": shard, "length": length}).encode())
        return cast(BinaryIO, CachedReader(self, shard, key, length))

    def publish(self, path: Path, data: bytes) -> None:
        temporary = self.root / (".block-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def make_room(self, size: int) -> bool:
        blocks = sorted(
            (
                p
                for p in self.root.glob("*.block")
                if re.fullmatch(r"[0-9a-f]{64}-[0-9a-f]{16}\.block", p.name)
            ),
            key=lambda p: p.stat().st_mtime_ns,
        )
        used = sum(p.stat().st_size for p in blocks)
        for path in blocks:
            if (
                used + size <= self.maximum_bytes
                and shutil.disk_usage(self.root).free >= self.reserve_bytes + size
            ):
                return True
            used -= path.stat().st_size
            path.unlink()
            self.memory.pop(path.name, None)
        return (
            size <= self.maximum_bytes
            and shutil.disk_usage(self.root).free >= self.reserve_bytes + size
        )

    def block(self, shard: dict[str, str], key: str, length: int, index: int) -> bytes:
        name = f"{key}-{index:016x}.block"
        if name in self.memory:
            self.memory.move_to_end(name)
            self.cache_hits += 1
            return self.memory[name]
        path = self.root / name
        start = index * self.block_bytes
        end = min(length, start + self.block_bytes) - 1
        if path.is_symlink():
            raise ValueError("Provider cache block cannot be a symlink")
        data = None
        if path.exists():
            encoded = path.read_bytes()
            header, separator, body = encoded.partition(b"\n")
            if (
                separator
                and header.decode("ascii", errors="replace") == hashlib.sha256(body).hexdigest()
                and len(body) == end - start + 1
            ):
                data = body
                self.cache_hits += 1
                os.utime(path, None)
            else:
                path.unlink()  # Discard only this run's corrupt disposable cache block.
        if data is None:
            request = Request(
                shard["url"],
                headers={"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"},
            )
            with urlopen(request, timeout=30) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{length}"
                    or response.headers.get("ETag", "").strip('"') != shard["etag"].strip('"')
                    or response.headers.get("Content-Length") != str(end - start + 1)
                ):
                    raise ValueError("Pinned provider range identity or length changed")
                data = response.read(end - start + 2)
                self.network_bytes += len(data)
                self.network_requests += 1
                if len(data) != end - start + 1:
                    raise OSError("Incomplete provider range")
            encoded = hashlib.sha256(data).hexdigest().encode() + b"\n" + data
            if self.make_room(len(encoded)):
                self.publish(path, encoded)
        self.memory[name] = data
        self.memory.move_to_end(name)
        while sum(len(b) for b in self.memory.values()) > 32 * 1024**2 and len(self.memory) > 1:
            self.memory.popitem(last=False)
        return data


class CachedReader:
    def __init__(self, provider: CachedProvider, shard: dict[str, str], key: str, length: int):
        self.provider, self.shard, self.key, self.length = provider, shard, key, length
        self.position = 0
        self.closed = False
        self.current_index = -1
        self.current = b""

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("Read from closed provider stream")
        remaining = (
            min(size, self.length - self.position) if size >= 0 else self.length - self.position
        )
        output = []
        while remaining:
            index, offset = divmod(self.position, self.provider.block_bytes)
            if index != self.current_index:
                self.current = self.provider.block(self.shard, self.key, self.length, index)
                self.current_index = index
            take = min(remaining, len(self.current) - offset)
            output.append(self.current[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(output)

    def close(self) -> None:
        self.closed = True
        self.current = b""
