"""Create an immutable, audited token snapshot without deleting staging artifacts."""

import argparse
import json
import os
import sqlite3
import zlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data.production import atomic, digest, sync_directory, verify
from codexa.data.v2_index import Protection
from codexa.data.v2_policy import admission_error
from codexa.data.v2_storage import Storage
from codexa.evaluation.protection import QuestionProtection
from codexa.tokenizer.bpe import CodexaTokenizer


def token_documents(root: Path, entries: list[dict[str, Any]]) -> Iterator[np.ndarray]:
    for entry in entries:
        tokens = np.memmap(root / entry["file"], dtype="<u2", mode="r")
        offsets = np.memmap(root / entry["index"], dtype="<u8", mode="r")
        for i, start in enumerate(offsets):
            end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(tokens)
            yield tokens[int(start) : end]


def finalize(root: Path, tokenizer: Path, name: str) -> dict[str, Any]:
    import fcntl

    storage = Storage.load(root)
    storage.check(256 * 1024**2)
    if Path(name).name != name:
        raise ValueError("Snapshot name must be a directory basename")
    output = root / "snapshots" / name
    if output.exists():
        raise FileExistsError("Historical snapshots are immutable: " + str(output))
    with (root / "tmp/index-writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify(root / "tokens")
        if not manifest.get("preparation_complete"):
            raise ValueError("Finish the current preparation batch before snapshotting")
        if digest(tokenizer) != manifest["tokenizer_hash"]:
            raise ValueError("Frozen tokenizer mismatch")
        tok = CodexaTokenizer.load(tokenizer)
        protection_path = (
            root
            / "indexes"
            / (
                "protected-v2.npz"
                if (root / "indexes/protected-v2.npz").exists()
                else "protected.npz"
            )
        )
        protected = Protection.load(protection_path)
        questions = QuestionProtection(root / "benchmarks/v3")
        admission = json.loads((root / "source-admission.json").read_text())
        registry = json.loads((root / "source-registry.json").read_text())["sources"]
        source_files = {r["sha256"]: r for r in registry}
        db = sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True)
        streams = {
            s: token_documents(root / "tokens", entries)
            for s, entries in manifest["splits"].items()
        }
        output.mkdir(parents=True)
        entries: dict[str, list[dict[str, Any]]] = {s: [] for s in streams}
        buffers = {s: bytearray() for s in streams}
        offsets: dict[str, list[int]] = {s: [] for s in streams}
        counts: Counter[str] = Counter()
        source_tokens: Counter[str] = Counter()
        source_chars: Counter[str] = Counter()
        source_documents: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        seen: Counter[str] = Counter()
        ids_path = output / "documents.jsonl"

        def commit(split: str) -> None:
            if not buffers[split]:
                return
            storage.check(len(buffers[split]) + 64 * 1024**2)
            path = output / f"{split}-{len(entries[split]):05d}.bin"
            with path.open("xb") as f:
                f.write(buffers[split])
                f.flush()
                os.fsync(f.fileno())
            ix = path.with_suffix(".bin.idx")
            with ix.open("xb") as f:
                f.write(np.asarray(offsets[split], dtype="<u8").tobytes())
                f.flush()
                os.fsync(f.fileno())
            entries[split].append(
                {
                    "file": path.name,
                    "index": ix.name,
                    "tokens": len(buffers[split]) // 2,
                    "documents": len(offsets[split]),
                    "sha256": digest(path),
                    "index_sha256": digest(ix),
                }
            )
            buffers[split] = bytearray()
            offsets[split] = []
            sync_directory(output)

        with ids_path.open("x") as listing:
            for ident, source, split, payload, metadata in db.execute(
                "SELECT hash,source,split,payload,metadata FROM docs WHERE role='new' ORDER BY id"
            ):
                seen[split] += 1
                token_ids = next(streams[split])
                text = zlib.decompress(payload).decode()
                # Full reconstruction ties the compressed document, split and token offsets together.
                if (
                    tok.decode(token_ids[1:-1].tolist()) != text
                    or token_ids[0] != 2
                    or token_ids[-1] != 3
                ):
                    raise ValueError("Document/token correspondence changed")
                meta = json.loads(metadata)
                approval = admission["sources"][source]
                source_file = source_files[meta["source_sha256"]]
                reason = None
                if approval.get("status") != "admitted":
                    reason = "source_not_admitted"
                elif approval.get("path_prefixes") and not any(
                    source_file["path"].startswith(p) for p in approval["path_prefixes"]
                ):
                    reason = "source_path_not_admitted"
                else:
                    reason = admission_error(text, meta, approval)
                if not reason and protected.matches(text):
                    reason = "protected_exact_span"
                if not reason and questions.matches(text):
                    reason = "protected_question_paraphrase"
                if reason:
                    rejected[source + ":" + reason] += 1
                    continue
                offsets[split].append(len(buffers[split]) // 2)
                buffers[split].extend(token_ids.tobytes())
                counts[split] += len(token_ids)
                if split == "train":
                    source_tokens[source] += len(token_ids)
                    source_chars[source] += len(text)
                    source_documents[source] += 1
                listing.write(
                    json.dumps(
                        {
                            "id": ident.hex(),
                            "source": source,
                            "split": split,
                            "tokens": len(token_ids),
                        }
                    )
                    + "\n"
                )
                if len(buffers[split]) >= 128 * 1024**2:
                    commit(split)
                    print(
                        json.dumps({"snapshot": name, "tokens": counts, "rejected": rejected}),
                        flush=True,
                    )
            listing.flush()
            os.fsync(listing.fileno())
        for split, stream in streams.items():
            if next(stream, None) is not None:
                raise ValueError("Unaccounted tokenized document")
            commit(split)
        db.close()
        result = {
            "version": 2,
            "stage": "tokenized",
            "dtype": "<u2",
            "vocab_size": manifest["vocab_size"],
            "tokenizer_hash": manifest["tokenizer_hash"],
            "dataset_name": "Codexa-v3-dataset-v2-" + name,
            "boundary": manifest["boundary"],
            "identity": manifest["identity"]
            | {
                "working_manifest": digest(root / "tokens/manifest.json"),
                "admission": digest(root / "source-admission.json"),
                "question_screen": digest(Path(__file__).parents[1] / "evaluation/protection.py"),
                "finalizer": digest(Path(__file__)),
                "cleaner_code": digest(Path(__file__).with_name("v2_quality.py")),
                "admission_code": digest(Path(__file__).with_name("v2_policy.py")),
                "preparation_code": digest(Path(__file__).with_name("v2.py")),
                "index_code": digest(Path(__file__).with_name("v2_index.py")),
                "protection": digest(protection_path),
            },
            "splits": entries,
            "preparation_complete": True,
            "source_tokens": dict(source_tokens),
            "documents_sha256": digest(ids_path),
        }
        atomic(output / "manifest.json", result)
        verify(output)
        report = {
            "snapshot": str(output),
            "manifest_sha256": digest(output / "manifest.json"),
            "split_tokens": dict(counts),
            "source_training_tokens": dict(source_tokens),
            "source_training_characters": dict(source_chars),
            "source_training_documents": dict(source_documents),
            "rejected_documents": dict(rejected),
            "all_accepted_documents_reconstructed": True,
            "all_accepted_documents_screened_against_final_policy": True,
            "semantic_contamination_guaranteed_absent": False,
            "limitations": "Exact spans and conservative lexical paraphrases; no exhaustive semantic contamination proof. Approximate deduplication cannot prove semantic uniqueness.",
            "training_started": False,
        }
        atomic(output / "audit.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--name", required=True)
    a = parser.parse_args()
    print(json.dumps(finalize(a.root, a.tokenizer, a.name), indent=2))


if __name__ == "__main__":
    main()
