"""Verified, run-owned rolling native checkpoints with transactional retention."""

import hashlib
import json
import math
import os
import shutil
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .checkpoint import check_volume, save_atomic


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_directory(root: Path) -> None:
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def equal_state(expected: Any, actual: Any) -> None:
    """Check complete read-back, including RNG, sampler buffers and optimizer tensors."""
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or not torch.equal(expected.detach().cpu(), actual)
            or (actual.is_floating_point() and not bool(torch.isfinite(actual).all()))
        ):
            raise ValueError("Checkpoint tensor read-back mismatch or nonfinite state")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("Checkpoint mapping read-back mismatch")
        for key in expected:
            equal_state(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError("Checkpoint sequence read-back mismatch")
        for a, b in zip(expected, actual):
            equal_state(a, b)
    elif expected != actual or (isinstance(actual, float) and not math.isfinite(actual)):
        raise ValueError("Checkpoint scalar read-back mismatch")


def payload_bytes(value: Any) -> int:
    """Conservative payload estimate; includes serialized admission ledger and tensor aliases."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(payload_bytes(k) + payload_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(payload_bytes(v) for v in value)
    if isinstance(value, (bytes, str)):
        return len(value) * (4 if isinstance(value, str) else 1)
    return 64


class RollingCheckpoints:
    def __init__(
        self, root: Path, keep: int = 2, volume_uuid: str | None = None, reserve: int = 40 * 1024**3
    ):
        if keep < 2:
            raise ValueError("At least two recovery checkpoints required")
        self.root, self.keep, self.volume_uuid, self.reserve = root, keep, volume_uuid, reserve
        self.manifest = root / "recovery.json"
        self.record: dict[str, Any] = (
            json.loads(self.manifest.read_text())
            if self.manifest.exists()
            else {"version": 1, "owner": uuid.uuid4().hex, "retained": []}
        )
        if self.record.get("version") != 1 or not self.record.get("owner"):
            raise ValueError("Invalid rolling checkpoint manifest")

    def path(self, entry: dict[str, Any]) -> Path:
        name = entry["file"]
        if Path(name).name != name or not (
            name.startswith("recovery-" + self.record["owner"] + "-") or name == "final.pt"
        ):
            raise ValueError("Checkpoint does not belong to this run")
        result = self.root / name
        if result.is_symlink() or result.parent.resolve() != self.root.resolve():
            raise ValueError("Checkpoint redirection refused")
        return result

    def verify(self, entry: dict[str, Any]) -> Path:
        path = self.path(entry)
        if path.stat().st_size != entry["size_bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Recovery checkpoint integrity mismatch")
        return path

    def latest(self) -> Path:
        for index, entry in enumerate(self.record["retained"]):
            try:
                path = self.verify(entry)
            except (OSError, ValueError):
                continue
            if index:
                warnings.warn(
                    "Latest checkpoint failed integrity; using previous recovery point",
                    stacklevel=2,
                )
            return path
        raise ValueError("No verified recovery checkpoint remains")

    def check_space(self, state: dict[str, Any]) -> None:
        check_volume(self.root, self.volume_uuid)
        required = int(payload_bytes(state) * 1.25) + 64 * 1024**2
        previous = max((e["size_bytes"] for e in self.record["retained"]), default=0)
        required = max(required, int(previous * 1.25))
        if shutil.disk_usage(self.root).free < self.reserve + required:
            raise OSError("Insufficient checkpoint transaction space above free-space reserve")

    def status(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.root)
        latest = self.record["retained"][0] if self.record["retained"] else None
        checkpoint = (
            {
                "identifier": latest["file"],
                "step": latest["step"],
                "timestamp": latest.get("verified_at", ""),
                "size_bytes": latest["size_bytes"],
                "verification_status": "readback_verified_at_publication",
            }
            if latest
            else {}
        )
        return checkpoint | {
            "retained": self.record["retained"],
            "disk_free_bytes": disk.free,
            "disk_used_bytes": disk.used,
            "recovery_bytes": sum(e["size_bytes"] for e in self.record["retained"]),
        }

    def save(self, state: dict[str, Any], final: bool = False) -> dict[str, Any]:
        old = list(self.record["retained"])
        if old and old[0]["step"] == state["step"] and old[0]["tokens"] == state["tokens"]:
            try:
                self.verify(old[0])
                return old[0]
            except (OSError, ValueError):
                pass
        self.check_space(state)
        pending = self.root / (".pending-" + uuid.uuid4().hex + ".pt")
        name = (
            "final.pt"
            if final
            else f"recovery-{self.record['owner']}-{state['step']:08d}-{uuid.uuid4().hex[:12]}.pt"
        )
        destination = self.root / name
        if destination.is_symlink() or (
            destination.exists() and not (final and any(e["file"] == name for e in old))
        ):
            raise FileExistsError("Checkpoint generation already exists")
        created = False
        try:
            save_atomic(state, pending)
            restored = torch.load(pending, map_location="cpu", weights_only=True, mmap=True)
            equal_state(state, restored)
            del restored
            entry = {
                "file": name,
                "sha256": digest(pending),
                "size_bytes": pending.stat().st_size,
                "step": state["step"],
                "tokens": state["tokens"],
                "verified_at": datetime.now(UTC).isoformat(),
            }
            valid = []
            for candidate in old:
                try:
                    self.verify(candidate)
                    valid.append(candidate)
                except (OSError, ValueError):
                    pass
            retained = [entry, *valid[: self.keep - 1]]
            os.replace(pending, destination)
            created = True
            sync_directory(self.root)
            updated = self.record | {"retained": retained, "latest": entry}
            updated["previous"] = retained[1] if len(retained) > 1 else None
            if final:
                updated["final"] = entry
            write_json(self.manifest, updated)
            self.record = updated
            receipts = self.root / "checkpoint-history"
            receipts.mkdir(exist_ok=True)
            write_json(receipts / (name + ".json"), entry)
            for obsolete in old:
                if obsolete["file"] not in {
                    e["file"] for e in retained
                } and obsolete != self.record.get("final"):
                    self.path(obsolete).unlink(missing_ok=True)
            sync_directory(self.root)
            return entry
        finally:
            pending.unlink(missing_ok=True)
            if created and not final:
                try:
                    published = (
                        json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
                    )
                    if name not in {e["file"] for e in published.get("retained", [])}:
                        destination.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass  # Uncertain publication state must never trigger deletion.

    def save_best(self, state: dict[str, Any]) -> None:
        """Private weights artifact, explicitly distinct from native resumable checkpoints."""
        self.check_space(state)
        pending = self.root / (".best-" + uuid.uuid4().hex + ".pt")
        try:
            save_atomic(state, pending)
            restored = torch.load(pending, map_location="cpu", weights_only=True, mmap=True)
            equal_state(state, restored)
            del restored
            os.replace(pending, self.root / "best-weights.pt")
            sync_directory(self.root)
        finally:
            pending.unlink(missing_ok=True)
