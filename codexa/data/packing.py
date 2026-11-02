import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from codexa.tokenizer.bpe import BOS, EOS, CodexaTokenizer


def fingerprint(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_documents(path: str | Path) -> list[str]:
    root = Path(path)
    paths = sorted(root.rglob("*.txt")) + sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    docs: list[str] = []
    for p in paths:
        if p.suffix == ".jsonl":
            docs.extend(
                json.loads(line)["text"] for line in p.read_text().splitlines() if line.strip()
            )
        else:
            docs.append(p.read_text(encoding="utf-8"))
    if not docs or any(not isinstance(d, str) or not d.strip() for d in docs):
        raise ValueError("Corpus must contain nonempty text documents")
    return docs


def split_documents(docs: list[str]) -> tuple[list[str], list[str]]:
    groups = {hashlib.sha256(d.encode()).hexdigest(): d for d in docs}
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two distinct documents")
    n = max(1, len(keys) // 10)
    return [groups[k] for k in keys[n:]], [groups[k] for k in keys[:n]]


def prepare(corpus: str, tokenizer_path: str, output: str) -> None:
    train, val = split_documents(read_documents(corpus))
    tok = CodexaTokenizer.load(tokenizer_path)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"tokenizer_hash": fingerprint(tokenizer_path), "splits": {}}
    for name, docs in (("train", train), ("val", val)):
        path = target / f"{name}.bin"
        count = 0
        with path.open("wb") as f:
            for doc in docs:
                ids = [BOS, *tok.encode(doc), EOS]
                np.asarray(ids, dtype="<u4").tofile(f)
                count += len(ids)
        meta["splits"][name] = {
            "tokens": count,
            "sha256": fingerprint(path),
            "documents": len(docs),
        }
    (target / "manifest.json").write_text(json.dumps(meta, indent=2))


class PackedData:
    def __init__(self, path: str | Path, sequence: int):
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        self.sequence = sequence
        self.size = (len(self.tokens) - 1) // sequence
        if self.size < 1:
            raise ValueError("Split is too small for one full sequence")

    def batch(self, indices: list[int], device: torch.device) -> tuple[Tensor, Tensor]:
        rows = np.stack(
            [
                self.tokens[i * self.sequence : i * self.sequence + self.sequence + 1]
                for i in indices
            ]
        )
        t = torch.from_numpy(rows.astype(np.int64)).to(device)
        return t[:, :-1], t[:, 1:]


class Sampler:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position = 0

    def next(self, count: int) -> list[int]:
        result = []
        for _ in range(count):
            if self.position == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
            result.append(int(self.order[self.position]))
            self.position += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"order": self.order, "position": self.position, "rng": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.order, self.position = state["order"], state["position"]
        self.generator.set_state(state["rng"])
