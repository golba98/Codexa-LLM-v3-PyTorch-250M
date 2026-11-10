import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    n = np.random.get_state(legacy=True)
    assert isinstance(n, tuple), "Native checkpoints require the legacy NumPy RNG tuple"
    return {
        "python": random.getstate(),
        "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def check_volume(path: Path, expected_uuid: str | None) -> None:
    if expected_uuid is None:
        return
    import shutil
    import subprocess

    ancestor = path.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    actual = subprocess.check_output(
        ["findmnt", "-T", str(ancestor), "-n", "-o", "UUID"], text=True
    ).strip()
    if actual != expected_uuid or shutil.disk_usage(ancestor).free < 20 * 1024**3:
        raise OSError("Checkpoint volume unavailable, wrong identity, or below 20 GiB reserve")


def save_atomic(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)
