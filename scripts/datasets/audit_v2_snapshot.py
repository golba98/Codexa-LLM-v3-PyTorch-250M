"""Read-only token-ID and boundary audit for an immutable dataset-v2 snapshot."""

import argparse
import json
from pathlib import Path

import numpy as np

from codexa.data.production import atomic, digest, verify
from codexa.data.v2_storage import Storage


def audit(snapshot: Path, root: Path) -> dict[str, object]:
    storage = Storage.load(root)
    storage.check()
    if not snapshot.resolve().is_relative_to((root / "snapshots").resolve()):
        raise ValueError("Snapshot must be under the verified preparation root")
    manifest = verify(snapshot)
    if not manifest.get("preparation_complete") or manifest.get("dtype") != "<u2":
        raise ValueError("Expected a completed uint16 snapshot")
    audit_path = snapshot / "audit.json"
    details = json.loads(audit_path.read_text())
    if details["manifest_sha256"] != digest(snapshot / "manifest.json"):
        raise ValueError("Snapshot audit/manifest identity mismatch")
    vocab_size = manifest["vocab_size"]
    counts = np.zeros(vocab_size, dtype=np.uint64)
    splits: dict[str, dict[str, int]] = {}
    for split, entries in manifest["splits"].items():
        positions = documents = 0
        for entry in entries:
            tokens = np.memmap(snapshot / entry["file"], dtype="<u2", mode="r")
            offsets = np.memmap(snapshot / entry["index"], dtype="<u8", mode="r")
            if len(tokens) != entry["tokens"] or len(offsets) != entry["documents"]:
                raise ValueError("Token/index dimensions differ from manifest")
            if len(offsets) and (int(offsets[0]) != 0 or np.any(offsets[1:] <= offsets[:-1])):
                raise ValueError("Invalid document offsets")
            if len(tokens) and int(tokens.max()) >= vocab_size:
                raise ValueError("Token ID exceeds frozen vocabulary")
            counts += np.bincount(tokens, minlength=vocab_size).astype(np.uint64)
            ends = np.concatenate((offsets[1:], np.asarray([len(tokens)], dtype="<u8")))
            if len(offsets) and (
                np.any(ends - offsets < 2)
                or np.any(tokens[offsets] != 2)
                or np.any(tokens[ends - 1] != 3)
            ):
                raise ValueError("BOS/EOS document boundary mismatch")
            positions += len(tokens)
            documents += len(offsets)
        splits[split] = {"positions": positions, "documents": documents}
    if int(counts[0]) or int(counts[1]):
        raise ValueError("Frozen tokenizer produced PAD or UNK IDs")
    total_documents = sum(x["documents"] for x in splits.values())
    if int(counts[2]) != total_documents or int(counts[3]) != total_documents:
        raise ValueError("Unexpected BOS/EOS usage")
    source_chars = details["source_training_characters"]
    source_tokens = details["source_training_tokens"]
    result: dict[str, object] = {
        "snapshot": str(snapshot),
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "split_positions_and_documents": splits,
        "vocabulary_size": vocab_size,
        "used_ids": int(np.count_nonzero(counts)),
        "pad_count": int(counts[0]),
        "unk_count": int(counts[1]),
        "bos_count": int(counts[2]),
        "eos_count": int(counts[3]),
        "source_characters_per_token": {
            name: source_chars[name] / tokens for name, tokens in source_tokens.items() if tokens
        },
        "top_token_ids": [
            {"id": int(i), "count": int(counts[i])} for i in np.argsort(counts)[-20:][::-1]
        ],
        "scope": "Exact uint16 ID/boundary audit; source character ratios are descriptive fragmentation proxies, not human text quality.",
        "training_started": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot, args.root)
    atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
