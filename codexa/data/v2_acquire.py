"""Pinned, bounded HTTP acquisition. No Hugging Face cache or remote code execution."""

import fcntl
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.v2_storage import Storage


def validate_spec(spec: dict[str, Any]) -> str:
    for key in ("name", "repo", "revision", "path", "sha256", "bytes", "license", "category"):
        if not spec.get(key):
            raise ValueError("Missing source field: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]) or not re.fullmatch(
        r"[0-9a-f]{64}", spec["sha256"]
    ):
        raise ValueError("Immutable revision and SHA256 required")
    if not isinstance(spec["bytes"], int) or spec["bytes"] <= 0:
        raise ValueError("Positive pinned byte count required")
    name = spec["name"] + "--" + spec["path"].replace("/", "--")
    if Path(name).name != name or ".." in name or Path(spec["path"]).is_absolute():
        raise ValueError("Unsafe acquisition path")
    return name


def acquire(storage: Storage, spec: dict[str, Any]) -> Path:
    storage.check()
    validate_spec(spec)
    key = hashlib.sha256((spec["name"] + spec["path"]).encode()).hexdigest()
    with (storage.root / "tmp" / (key + ".download.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _acquire(storage, spec)


def _acquire(storage: Storage, spec: dict[str, Any]) -> Path:
    name = validate_spec(spec)
    path = storage.root / "raw" / name
    receipt = path.with_name(path.name + ".receipt.json")
    if path.is_symlink():
        raise ValueError("Acquisition files must not redirect through symlinks")
    if path.exists():
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Existing acquisition failed integrity")
        if not receipt.exists():
            atomic(receipt, spec | {"verified_sha256": spec["sha256"]})
        return path
    raw_bytes = Storage(storage.root / "raw").usage()
    partial = path.with_name(path.name + ".partial")
    if path.is_symlink() or partial.is_symlink():
        raise ValueError("Acquisition files must not redirect through symlinks")
    offset = partial.stat().st_size if partial.exists() else 0
    if raw_bytes + spec["bytes"] - offset > storage.raw_maximum:
        raise OSError("Raw acquisition ceiling exceeded")
    storage.check(spec["bytes"] - offset + 64 * 1024**2)
    url = spec.get("url") or (
        f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    )
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec["bytes"]:
            raise ValueError("Oversized partial acquisition retained for investigation")
        if offset == spec["bytes"]:
            break
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                ):
                    raise ValueError("Server did not honor safe range recovery; partial retained")
                h = hashlib.sha256()
                if offset:
                    with partial.open("rb") as previous:
                        for block in iter(lambda: previous.read(8 * 1024**2), b""):
                            h.update(block)
                with partial.open("ab") as f:
                    since_check = 0
                    while block := response.read(8 * 1024**2):
                        if offset + len(block) > spec["bytes"]:
                            raise ValueError("Download exceeds pinned size")
                        if since_check >= 64 * 1024**2 or since_check == 0:
                            storage.check(min(spec["bytes"] - offset, 128 * 1024**2))
                            since_check = 0
                        f.write(block)
                        h.update(block)
                        offset += len(block)
                        since_check += len(block)
                    f.flush()
                    os.fsync(f.fileno())
                if offset != spec["bytes"]:
                    raise OSError("Incomplete transfer")
                if h.hexdigest() != spec["sha256"]:
                    raise ValueError("Source SHA256 mismatch; partial retained")
                break
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(min(2**attempt, 8))
    if digest(partial) != spec["sha256"]:
        raise ValueError("Completed partial failed SHA256")
    storage.check(1024**2)
    partial.rename(path)
    atomic(
        receipt, spec | {"verified_sha256": spec["sha256"], "verified_bytes": path.stat().st_size}
    )
    print(json.dumps({"acquired": name, "bytes": path.stat().st_size}), flush=True)
    return path


def acquire_xet(storage: Storage, spec: dict[str, Any]) -> Path:
    """Optional pinned hf-xet transport with an isolated, monitored child process."""
    import subprocess
    import sys

    storage.check()
    validate_spec(spec)
    key = hashlib.sha256((spec["name"] + spec["path"]).encode()).hexdigest()
    with (storage.root / "tmp" / (key + ".download.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        name = spec["name"] + "--" + spec["path"].replace("/", "--")
        path = storage.root / "raw" / name
        if path.is_symlink():
            raise ValueError("Accelerated acquisition files must not be symlinks")
        if path.exists():
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError("Existing accelerated acquisition failed integrity")
            receipt = path.with_name(path.name + ".receipt.json")
            if not receipt.exists():
                atomic(
                    receipt,
                    spec
                    | {
                        "verified_sha256": spec["sha256"],
                        "verified_bytes": path.stat().st_size,
                        "transport": "recovered verified accelerated payload",
                    },
                )
            return path
        if Storage(storage.root / "raw").usage() + spec["bytes"] > storage.raw_maximum:
            raise OSError("Raw acquisition ceiling exceeded")
        storage.check(spec["bytes"] * 2 + 64 * 1024**2)
        destination = storage.root / "raw" / "_hub" / spec["name"]
        if not (destination / spec["path"]).resolve().is_relative_to(storage.root.resolve()):
            raise ValueError("Accelerated acquisition path escapes the preparation volume")
        destination.mkdir(parents=True, exist_ok=True)
        log = storage.root / "logs" / ("xet-" + spec["sha256"][:12] + ".log")
        code = (
            "import json,sys; from huggingface_hub import hf_hub_download; "
            "s=json.loads(sys.argv[1]); "
            "p=hf_hub_download(s['repo'],s['path'],repo_type='dataset',revision=s['revision'],local_dir=sys.argv[2],token=False); "
            "print(p)"
        )
        env = os.environ | {
            "HF_XET_CACHE": str(storage.root / "cache/hf/xet"),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        with log.open("a") as out:
            child = subprocess.Popen(
                [sys.executable, "-B", "-c", code, json.dumps(spec), str(destination)],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
            try:
                started = time.monotonic()
                while child.poll() is None:
                    storage.check(64 * 1024**2)
                    if time.monotonic() - started > 7200:
                        raise TimeoutError("Bounded accelerated acquisition timed out")
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError("Accelerated acquisition failed; inspect " + str(log))
            except BaseException:
                child.terminate()
                child.wait(timeout=30)
                raise
        source = destination / spec["path"]
        if (
            source.is_symlink()
            or source.stat().st_size != spec["bytes"]
            or digest(source) != spec["sha256"]
        ):
            raise ValueError("Accelerated source failed pinned size/SHA256")
        os.link(source, path)
        atomic(
            path.with_name(path.name + ".receipt.json"),
            spec
            | {
                "verified_sha256": spec["sha256"],
                "verified_bytes": source.stat().st_size,
                "transport": "hf-xet; retained payload hard-linked, not copied",
            },
        )
        print(json.dumps({"acquired_xet": name, "bytes": source.stat().st_size}), flush=True)
        return path
