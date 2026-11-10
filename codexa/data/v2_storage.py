"""Fail-closed storage policy for explicitly requested dataset-v2 preparation."""

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codexa.data.production import atomic

GIB = 1024**3
VOLUME_UUID = "279352d0-0011-4733-8348-6815a4308196"


@dataclass(frozen=True)
class Storage:
    root: Path
    uuid: str = VOLUME_UUID
    reserve: int = 40 * GIB
    maximum: int = 84 * GIB
    raw_maximum: int = 30 * GIB

    def check(self, allocation: int = 0) -> None:
        ancestor = self.root.resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        actual = subprocess.check_output(
            ["findmnt", "-T", str(ancestor), "-n", "-o", "UUID"], text=True
        ).strip()
        if actual != self.uuid:
            raise OSError("Wrong or unavailable dataset-v2 volume")
        if self.root.exists() and any(p.is_symlink() for p in self.root.iterdir()):
            raise OSError("Dataset-v2 storage may not redirect through symlinks")
        if shutil.disk_usage(ancestor).free < self.reserve + allocation:
            raise OSError("Dataset-v2 allocation would violate free-space reserve")
        if self.root.exists() and self.usage() + allocation > self.maximum:
            raise OSError("Dataset-v2 storage ceiling exceeded")

    def usage(self) -> int:
        # Count allocated bytes once per inode, including partial files and caches.
        seen: set[tuple[int, int]] = set()
        total = 0
        for directory, _, files in os.walk(self.root):
            for name in files:
                p = Path(directory) / name
                if p.is_symlink():
                    continue
                try:
                    s = p.stat()
                except FileNotFoundError:
                    continue  # A concurrent atomic commit renamed its temporary file.
                key = (s.st_dev, s.st_ino)
                if key not in seen:
                    seen.add(key)
                    total += s.st_blocks * 512
        return total

    def initialize(self) -> None:
        self.check()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "clean", "tokens", "indexes", "reports", "logs", "tmp", "cache"):
            (self.root / name).mkdir(exist_ok=True)
        atomic(
            self.root / "storage-policy.json",
            {
                "uuid": self.uuid,
                "reserve_bytes": self.reserve,
                "maximum_bytes": self.maximum,
                "raw_maximum_bytes": self.raw_maximum,
            },
        )

    @classmethod
    def load(cls, root: str | Path) -> "Storage":
        root = Path(root)
        p = root / "storage-policy.json"
        if not p.exists():
            raise ValueError("Initialize an explicit storage policy first")
        policy = json.loads(p.read_text())
        return cls(
            root,
            policy["uuid"],
            policy["reserve_bytes"],
            policy["maximum_bytes"],
            policy.get("raw_maximum_bytes", 30 * GIB),
        )


def environment(root: Path) -> dict[str, str]:
    result = {
        "CODEXA_NO_TRAINING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(root / "cache/python"),
        "PYTEST_ADDOPTS": "-o cache_dir=" + shlex.quote(str(root / "cache/pytest")),
    }
    for key in ("TMPDIR", "TMP", "TEMP", "SQLITE_TMPDIR"):
        result[key] = str(root / "tmp")
    for key, folder in {
        "XDG_CACHE_HOME": "xdg",
        "HF_HOME": "hf",
        "HF_DATASETS_CACHE": "hf/datasets",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "TORCH_HOME": "torch",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "PYTHONPYCACHEPREFIX": "python",
        "RUFF_CACHE_DIR": "ruff",
        "MYPY_CACHE_DIR": "mypy",
        "MPLCONFIGDIR": "matplotlib",
        "PIP_CACHE_DIR": "pip",
        "UV_CACHE_DIR": "uv",
    }.items():
        result[key] = str(root / "cache" / folder)
    return result
