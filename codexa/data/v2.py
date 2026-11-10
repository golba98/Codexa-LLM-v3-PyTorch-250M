"""Opt-in, bounded dataset-v2 preparation; never imports or invokes a trainer."""

import argparse
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data.production import atomic, digest, sync_directory, verify
from codexa.data.v2_acquire import acquire
from codexa.data.v2_index import Index, Protection
from codexa.data.v2_policy import admission_error, admission_text
from codexa.data.v2_quality import VERSION, canonical, clean, features, source_error, source_text
from codexa.data.v2_storage import Storage

TOKENIZER: Any = None


def worker_init(tokenizer: str) -> None:
    global TOKENIZER
    from codexa.tokenizer.bpe import CodexaTokenizer

    TOKENIZER = CodexaTokenizer.load(tokenizer)


def records(
    path: Path, spec: dict[str, Any], offset: int = 0
) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.name.endswith(".txt"):
        text: list[str] = []
        size = 0
        ordinal = 0
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "<|endoftext|>":
                    ordinal += 1
                    if ordinal > offset:
                        yield (
                            ordinal,
                            {"text": "".join(text)} if size <= 1024**2 else {"error": "oversize"},
                        )
                    text, size = [], 0
                else:
                    size += len(line)
                    if size <= 1024**2:
                        text.append(line)
            if text or size:
                ordinal += 1
                if ordinal > offset:
                    yield (
                        ordinal,
                        {"text": "".join(text)} if size <= 1024**2 else {"error": "oversize"},
                    )
        return
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    allowed = {
        "text",
        "url",
        "id",
        "title",
        "int_score",
        "score",
        "language_score",
        "audience",
        "format",
        "seed_data",
    }
    columns = [x for x in file.schema_arrow.names if x in allowed]
    ordinal = 0
    for group in range(file.num_row_groups):
        n = file.metadata.row_group(group).num_rows
        if ordinal + n <= offset:
            ordinal += n
            continue
        for batch in file.iter_batches(batch_size=32, columns=columns, row_groups=[group]):
            for row in batch.to_pylist():
                ordinal += 1
                if ordinal > offset:
                    yield ordinal, row


def process(payload: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    ordinal, row, spec = payload
    reason = source_error(row, spec)
    text, errors, stats = (
        clean(
            admission_text(source_text(row, spec["name"]), spec.get("_approval", {})),
            spec["category"],
        )
        if not reason
        else ("", [reason], {})
    )
    result: dict[str, Any] = {"ordinal": ordinal, "errors": errors, "stats": stats}
    if errors:
        return result
    approval_error = admission_error(text, row, spec.get("_approval", {}))
    if approval_error:
        result["errors"] = [approval_error]
        return result
    ids = [2, *TOKENIZER.encode(text), 3]
    if 0 in ids or 1 in ids or TOKENIZER.decode(ids[1:-1]) != text:
        result["errors"] = ["tokenizer_fidelity"]
        return result
    shingles, signature, anchors = features(text)
    result.update(
        text=text,
        ident=hashlib.sha256(canonical(text).encode()).digest(),
        tokens=np.asarray(ids, dtype="<u2").tobytes(),
        token_count=len(ids),
        original_characters=len(row["text"]),
        cleaned_characters=len(text),
        shingles=shingles,
        signature=signature,
        anchors=anchors,
        metadata={
            k: row[k] for k in ("url", "id", "title", "audience", "format", "seed_data") if k in row
        },
    )
    return result


def protection(storage: Storage, historical: Path) -> Protection:
    path = storage.root / "indexes/protected-v2.npz"
    fixtures = storage.root / "benchmarks/v3"
    fixture_hash = digest(fixtures / "manifest.json")
    marker = path.with_suffix(".json")
    if path.exists():
        if json.loads(marker.read_text())["fixture_hash"] != fixture_hash:
            raise ValueError("Protected evaluation identity changed")
        return Protection.load(path)
    texts: list[tuple[str, bool | int]] = []
    db = sqlite3.connect(f"file:{historical}?mode=ro&immutable=1", uri=True)
    texts.extend((r[0], False) for r in db.execute("SELECT text FROM docs WHERE split!='train'"))
    db.close()
    for split in ("dev", "test"):
        for line in (fixtures / f"{split}.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["kind"] in ("fact", "vocabulary", "math"):
                texts.append((item.get("question", ""), True))
            else:
                texts.append((item["prompt"] or item["answer"], True))
                if item["kind"] == "grammar":
                    texts.extend((s, True) for s in item["candidates"])
    for receipt in (storage.root / "raw").glob("*valid*.receipt.json"):
        spec = json.loads(receipt.read_text())
        src = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        texts.extend((row["text"], 64) for _, row in records(src, spec))
    p = Protection.build(texts)
    p.save(path)
    atomic(
        marker,
        {
            "fixture_hash": fixture_hash,
            "sha256": digest(path),
            "span_counts": {str(k): len(v) for k, v in p.items.items()},
            "policy": "13-token historical holdout/item spans; 64-token official story spans plus evaluation document/near/passage index; question paraphrases screened separately; ordinary answer words allowed",
        },
    )
    return p


def reference_evaluation(index: Index, storage: Storage, workers: int) -> None:
    """Index official validation plots without treating generic openings as leakage."""
    from codexa.data.v2_index import _reference_features

    for receipt in sorted((storage.root / "raw").glob("*valid*.receipt.json")):
        spec = json.loads(receipt.read_text())
        source = receipt.with_name(receipt.name.removesuffix(".receipt.json"))
        key = "evaluation_reference:" + spec["sha256"]
        if index.value(key + ":complete", False):
            continue
        if digest(source) != spec["sha256"]:
            raise ValueError("Official validation source changed")
        iterator = records(source, spec, index.value(key + ":cursor", 0))
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            while batch := list(islice(iterator, 512)):
                payload = [
                    (hashlib.sha256(canonical(row["text"]).encode()).hexdigest(), row["text"])
                    for _, row in batch
                ]
                for (ident, text), (_, signature, anchors) in zip(
                    payload, pool.map(_reference_features, payload, chunksize=16)
                ):
                    binary_id = bytes.fromhex(ident)
                    if not index.db.execute(
                        "SELECT 1 FROM docs WHERE hash=?", (binary_id,)
                    ).fetchone():
                        index.insert(
                            binary_id,
                            spec["name"] + "-official-valid",
                            "reference",
                            "protected",
                            text,
                            signature,
                            anchors,
                            {"source_sha256": spec["sha256"]},
                            role="evaluation",
                        )
                index.set(key + ":cursor", batch[-1][0])
                index.db.commit()
                storage.check(64 * 1024**2)
        index.set(key + ":complete", True)
        index.db.commit()


def prepare(
    storage: Storage,
    historical: Path,
    tokenizer: Path,
    target: int,
    workers: int = 4,
    expected_tokenizer_sha: str = "60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771",
    vocab_size: int = 16384,
) -> dict[str, Any]:
    index = Index(storage, historical)
    try:
        from codexa.data.v2_registry import recover

        recover(index)
        return _prepare(
            storage,
            historical,
            tokenizer,
            target,
            index,
            workers,
            expected_tokenizer_sha,
            vocab_size,
        )
    finally:
        index.close()


def _prepare(
    storage: Storage,
    historical: Path,
    tokenizer: Path,
    target: int,
    index: Index,
    workers: int = 4,
    expected_tokenizer_sha: str = "60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771",
    vocab_size: int = 16384,
) -> dict[str, Any]:
    expected = expected_tokenizer_sha
    if digest(tokenizer) != expected:
        raise ValueError("Frozen tokenizer identity mismatch")
    storage.check(256 * 1024**2)
    if not index.value("historical_complete", False):
        raise ValueError("Build the historical reference index before new preparation")
    reference_evaluation(index, storage, workers)
    protected = protection(storage, historical)
    from codexa.evaluation.protection import QuestionProtection

    questions = QuestionProtection(storage.root / "benchmarks/v3")
    config_hash = digest(storage.root / "source-registry.json")
    identity = {
        "source_registry": config_hash,
        "tokenizer": expected,
        "cleaner": VERSION,
        "vocab_size": vocab_size,
        "excluded_dataset_fingerprints": [digest(historical.parent.parent / "tokens/manifest.json")]
        if (historical.parent.parent / "tokens/manifest.json").exists()
        else [],
        "historical": digest(historical.parent / "manifest.json"),
        "evaluation": digest(storage.root / "benchmarks/v3/manifest.json"),
    }
    previous = index.value("preparation_identity")
    if previous and previous != identity:
        raise ValueError("Resume inputs changed")
    index.set("preparation_identity", identity)
    index.db.commit()
    entries = index.value("token_entries", {"train": [], "val": [], "test": []})
    totals = index.value("token_totals", {"train": 0, "val": 0, "test": 0})
    source_tokens = index.value("source_tokens", {})
    source_documents = index.value("source_documents", {})
    accepted = index.value("accepted_documents", 0)
    cursor_state = index.value("source_cursors", {})
    registry = json.loads((storage.root / "source-registry.json").read_text())["sources"]
    admission = json.loads((storage.root / "source-admission.json").read_text())
    if admission["cleaner"] != VERSION:
        raise ValueError("Source review uses a different cleaner")
    admission_sha = digest(storage.root / "source-admission.json")
    revisions = index.value("admission_revisions", [])
    if admission_sha not in revisions:
        revisions.append(admission_sha)
        atomic(storage.root / "reports" / ("admission-" + admission_sha + ".json"), admission)
        index.set("admission_revisions", revisions)
        index.db.commit()
    receipts = list((storage.root / "raw").glob("*.receipt.json"))
    specs = [json.loads(p.read_text()) for p in receipts if "valid" not in p.name]
    rank = {str(x["sha256"]): i for i, x in enumerate(registry)}
    registered = {x["sha256"]: x for x in registry}
    for spec in specs:
        pinned = registered[spec["sha256"]]
        if any(spec.get(k) != pinned.get(k) for k in ("name", "repo", "path", "revision", "bytes")):
            raise ValueError("Receipt no longer matches the pinned registry")
    specs = [registered[x["sha256"]] for x in specs]
    specs.sort(key=lambda x: rank[x["sha256"]])
    started = time.monotonic()
    examples_path = storage.root / "reports/cleaning-examples.jsonl"
    example_counts = index.value("example_counts", {})
    shard_count = sum(len(v) for v in entries.values())
    buffers: dict[str, bytearray] = {s: bytearray() for s in entries}
    offsets: dict[str, list[int]] = {s: [] for s in entries}
    pending = 0

    def commit() -> None:
        nonlocal pending, shard_count
        storage.check(256 * 1024**2)
        for split, payload in buffers.items():
            if not payload:
                continue
            name = f"{split}-{shard_count:06d}.bin"
            path = storage.root / "tokens" / name
            ix = path.with_suffix(".bin.idx")
            expected_hash = hashlib.sha256(payload).hexdigest()
            if path.exists() and digest(path) != expected_hash:
                raise ValueError("Uncommitted shard differs; preserved for investigation")
            if not path.exists():
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as out:
                    out.write(payload)
                    out.flush()
                    os.fsync(out.fileno())
                temporary.rename(path)
            ix_data = np.asarray(offsets[split], dtype="<u8").tobytes()
            if ix.exists() and ix.read_bytes() != ix_data:
                raise ValueError("Uncommitted offset index differs")
            if not ix.exists():
                with ix.open("wb") as out:
                    out.write(ix_data)
                    out.flush()
                    os.fsync(out.fileno())
            entries[split].append(
                {
                    "file": name,
                    "tokens": len(payload) // 2,
                    "documents": len(offsets[split]),
                    "sha256": expected_hash,
                    "index": ix.name,
                    "index_sha256": digest(ix),
                }
            )
            shard_count += 1
            buffers[split] = bytearray()
            offsets[split] = []
        sync_directory(storage.root / "tokens")
        for key, value in [
            ("token_entries", entries),
            ("token_totals", totals),
            ("source_tokens", source_tokens),
            ("source_documents", source_documents),
            ("accepted_documents", accepted),
            ("source_cursors", cursor_state),
            ("example_counts", example_counts),
        ]:
            index.set(key, value)
        index.db.commit()
        manifest = {
            "version": 2,
            "stage": "tokenized",
            "dtype": "<u2",
            "vocab_size": vocab_size,
            "tokenizer_hash": expected,
            "dataset_name": "Codexa-v3-dataset-v2",
            "identity": identity,
            "boundary": "BOS document EOS; causal attention across boundaries",
            "splits": entries,
            "preparation_complete": False,
            "source_tokens": source_tokens,
            "admission_revisions": revisions,
            "registry_revisions": index.value("registry_revisions", []),
            "protection_sha256": digest(storage.root / "indexes/protected-v2.npz"),
        }
        atomic(storage.root / "tokens/manifest.json", manifest)
        atomic(
            storage.root / "reports/preparation-progress.json",
            {
                "tokens": totals,
                "sources": source_tokens,
                "documents": accepted,
                "elapsed_seconds_this_run": time.monotonic() - started,
            },
        )
        pending = 0
        print(
            json.dumps(
                {"accepted_documents": accepted, "tokens": totals, "sources": source_tokens}
            ),
            flush=True,
        )

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(str(tokenizer),),
            mp_context=get_context("spawn"),
        ) as pool:
            for spec in specs:
                approval = admission["sources"].get(spec["name"], {})
                if approval.get("status") != "admitted" or (
                    approval.get("path_prefixes")
                    and not any(spec["path"].startswith(p) for p in approval["path_prefixes"])
                ):
                    continue
                path = (
                    storage.root / "raw" / (spec["name"] + "--" + spec["path"].replace("/", "--"))
                )
                if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                    raise ValueError("Source integrity changed")
                source_key = spec["name"] + ":" + spec["sha256"]
                quota = int(target * approval.get("target_share", spec["target_share"]))
                if source_tokens.get(spec["name"], 0) >= quota:
                    continue
                offset = cursor_state.get(source_key, 0)
                if offset == -1:
                    continue
                iterator = records(path, spec, offset)
                while totals["train"] < target and source_tokens.get(spec["name"], 0) < quota:
                    batch = []
                    for _ in range(128):
                        pair = next(iterator, None)
                        if pair is None:
                            break
                        batch.append((pair[0], pair[1], spec | {"_approval": approval}))
                    if not batch:
                        break
                    for result in pool.map(process, batch, chunksize=8):
                        cursor_state[source_key] = result["ordinal"]
                        errors = result["errors"]
                        if not errors:
                            gate = admission_error(result["text"], result["metadata"], approval)
                            if gate:
                                errors = [gate]
                        if not errors and protected.matches(result["text"]):
                            errors = ["protected_evaluation_span"]
                        if not errors and questions.matches(result["text"]):
                            errors = ["protected_question_paraphrase"]
                        if not errors:
                            reason = index.duplicate(
                                result["ident"],
                                result["text"],
                                result["shingles"],
                                result["signature"],
                                result["anchors"],
                            )
                            if reason:
                                errors = [reason]
                        if errors:
                            index.reject(spec["name"], errors[0])
                            continue
                        text = result["text"]
                        ident = result["ident"]
                        group = result["metadata"].get("url") or ident.hex()
                        bucket = (
                            int(hashlib.sha256(("42:" + group).encode()).hexdigest()[:16], 16)
                            / 2**64
                        )
                        split = "test" if bucket < 0.005 else "val" if bucket < 0.01 else "train"
                        index.insert(
                            ident,
                            spec["name"],
                            spec["category"],
                            split,
                            text,
                            result["signature"],
                            result["anchors"],
                            result["metadata"]
                            | {
                                "revision": spec["revision"],
                                "license": spec["license"],
                                "source_sha256": spec["sha256"],
                                "ordinal": result["ordinal"],
                                "admission_sha256": admission_sha,
                                "quality": result["stats"],
                                "characters": result["cleaned_characters"],
                            },
                            result["token_count"],
                        )
                        offsets[split].append(len(buffers[split]) // 2)
                        buffers[split].extend(result["tokens"])
                        totals[split] += result["token_count"]
                        if split == "train":
                            source_tokens[spec["name"]] = (
                                source_tokens.get(spec["name"], 0) + result["token_count"]
                            )
                        source_documents[spec["name"]] = source_documents.get(spec["name"], 0) + 1
                        accepted += 1
                        pending += result["token_count"]
                        if example_counts.get(spec["name"], 0) < 100:
                            with examples_path.open("a") as out:
                                out.write(
                                    json.dumps(
                                        {
                                            "source": spec["name"],
                                            "id": ident.hex(),
                                            "text": text[:6000],
                                            "ordinal": result["ordinal"],
                                            "source_sha256": spec["sha256"],
                                            "original_characters": result["original_characters"],
                                            "stats": result["stats"],
                                            "scope": "first accepted examples; not a random quality estimate",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            example_counts[spec["name"]] = example_counts.get(spec["name"], 0) + 1
                    if pending >= 4_000_000:
                        commit()
                if totals["train"] >= target:
                    commit()
                    break
                if source_tokens.get(spec["name"], 0) < quota:
                    cursor_state[source_key] = -1
                commit()
        commit()
        manifest = verify(storage.root / "tokens")
        manifest["preparation_complete"] = True
        manifest["target_training_tokens"] = target
        manifest["target_reached"] = totals["train"] >= target
        atomic(storage.root / "tokens/manifest.json", manifest)
        report = {
            "training_tokens": totals["train"],
            "split_tokens": totals,
            "source_tokens": source_tokens,
            "source_documents": source_documents,
            "accepted_documents": accepted,
            "target_reached": totals["train"] >= target,
            "new_training_token_positions": True,
            "historical_complete": True,
            "rejections": [
                {"source": s, "reason": r, "documents": n}
                for s, r, n in index.db.execute(
                    "SELECT source,reason,n FROM errors ORDER BY source,reason"
                )
            ],
            "tokenizer_sha256": expected,
            "tokenizer_roundtrip_checked_for_every_accepted_document": True,
            "original_checkpoint_modified": False,
            "training_started": False,
            "definitions": "Unique retained document positions after exact/approximate document and verified long-passage deduplication; not proof of semantic novelty.",
        }
        atomic(storage.root / "reports/dataset-statistics.json", report)
        return report
    finally:
        index.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["acquire", "reference", "prepare", "verify", "status"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--historical", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--target-tokens", type=int, default=2_000_000_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sha256")
    a = p.parse_args(argv)
    storage = Storage.load(a.root)
    storage.check()
    if a.action == "acquire":
        specs = json.loads((a.root / "source-registry.json").read_text())["sources"]
        for spec in specs:
            if spec["sha256"] == a.sha256:
                print(acquire(storage, spec))
                return
        p.error("Select a pinned --sha256 from the registry")
    elif a.action == "reference":
        if a.historical is None:
            p.error("--historical required")
        index = Index(storage, a.historical)
        try:
            index.reference(a.workers)
        finally:
            index.close()
    elif a.action == "prepare":
        if a.historical is None or a.tokenizer is None:
            p.error("--historical and --tokenizer required")
        print(
            json.dumps(
                prepare(storage, a.historical, a.tokenizer, a.target_tokens, a.workers), indent=2
            )
        )
    elif a.action == "verify":
        print(json.dumps(verify(a.root / "tokens"), indent=2))
    else:
        print(
            json.dumps(
                {
                    "root": str(a.root),
                    "allocated_bytes": storage.usage(),
                    "progress": json.loads(
                        (a.root / "reports/preparation-progress.json").read_text()
                    )
                    if (a.root / "reports/preparation-progress.json").exists()
                    else None,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
