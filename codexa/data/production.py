"""Disk-backed document curation and immutable token shards. No eager PyTorch import."""

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def records(
    path: Path, format: str, max_bytes: int = 1048576, start_record: int = 0, start_byte: int = 0
) -> Iterator[tuple[int, dict]]:
    if format == "parquet":
        import pyarrow.parquet as pq

        file = pq.ParquetFile(path)
        skipped = 0
        for group in range(file.num_row_groups):
            group_count = file.metadata.row_group(group).num_rows
            if skipped + group_count <= start_record:
                skipped += group_count
                continue
            if file.metadata.row_group(group).total_byte_size > 128 * 1048576:
                raise ValueError("Parquet row group exceeds 128 MiB; repartition source")
            for batch in file.iter_batches(batch_size=32, row_groups=[group]):
                for row in batch.to_pylist():
                    skipped += 1
                    if skipped > start_record:
                        yield -1, row
        return
    if format not in ("text", "jsonl"):
        raise ValueError("Expected text, jsonl or parquet")
    with path.open("rb") as f:
        f.seek(start_byte)
        if format == "text":
            # Each bounded paragraph is a record; source file is its split group.
            paragraph = bytearray()
            oversized = False
            for line in iter(lambda: f.readline(max_bytes + 1), b""):
                if not line.strip():
                    if paragraph or oversized:
                        try:
                            yield (
                                f.tell(),
                                {"text": paragraph.decode("utf-8")}
                                if not oversized
                                else {"error": "oversize"},
                            )
                        except UnicodeError:
                            yield f.tell(), {"error": "encoding"}
                    paragraph.clear()
                    oversized = False
                elif len(paragraph) + len(line) > max_bytes:
                    oversized = True
                    paragraph.clear()
                elif not oversized:
                    paragraph.extend(line)
            if paragraph or oversized:
                try:
                    yield (
                        f.tell(),
                        {"text": paragraph.decode("utf-8")}
                        if not oversized
                        else {"error": "oversize"},
                    )
                except UnicodeError:
                    yield f.tell(), {"error": "encoding"}
        else:
            while line := f.readline(max_bytes + 1):
                if len(line) > max_bytes:
                    while line and not line.endswith(b"\n"):
                        line = f.readline(max_bytes + 1)
                    yield f.tell(), {"error": "oversize"}
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                    yield f.tell(), row if isinstance(row, dict) else {"error": "record"}
                except (ValueError, UnicodeError):
                    yield f.tell(), {"error": "encoding_or_json"}


def spans(text):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    for i in range(max(0, len(tokens) - 12)):
        yield hashlib.blake2b(" ".join(tokens[i : i + 13]).encode(), digest_size=8).hexdigest()


def stage_huggingface(iterable, output, provenance, max_bytes=100 * 1048576):
    """Materialize a caller-supplied HF IterableDataset with a bounded payload budget.

    The caller pins revision and controls remote transport; this bounds written text,
    not upstream HTTP buffering. Normal curation then operates on this local snapshot.
    """
    for field in ("name", "snapshot", "license", "attribution"):
        if not provenance.get(field):
            raise ValueError("Missing provenance: " + field)
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = count = 0
    with target.open("xb") as f:
        for row in iterable:
            data = json.dumps(row, ensure_ascii=False).encode() + b"\n"
            if total + len(data) > max_bytes:
                break
            f.write(data)
            total += len(data)
            count += 1
    atomic(
        target.with_suffix(".provenance.json"),
        {
            **provenance,
            "bytes": total,
            "documents": count,
            "sha256": digest(target),
            "budget": max_bytes,
        },
    )
    return target


def shingles(text):
    words = re.findall(r"\w+|[^\w\s]", text)
    if len(words) < 128:
        return set()
    return {
        hashlib.blake2b(" ".join(words[i : i + 5]).encode(), digest_size=8).hexdigest()
        for i in range(len(words) - 4)
    }


def curate(
    config: dict,
    output: str,
    resume=False,
    progress=None,
    *,
    commit_every: int = 100,
    cache_mib: int = 8,
    wal: bool = False,
    protected_cache_entries: int = 0,
) -> Path:
    if commit_every <= 0 or cache_mib <= 0 or protected_cache_entries < 0:
        raise ValueError("Positive curation transaction/cache limits required")
    config = {
        "seed": 42,
        "fractions": [0.99, 0.005, 0.005],
        "near_duplicates": True,
        "near_threshold": 0.85,
        "protect_spans": True,
        "min_chars": 32,
        "max_document_bytes": 1048576,
        "repetition_min_chars": 512,
        "minimum_unique_words": 8,
        **config,
    }
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError(root)
    if resume and (not root.exists() or not (root / "documents.sqlite").exists()):
        raise ValueError("Resume requires an existing curation database")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    db = None
    with (root / ".writer.lock").open("a") as writer_lock:
        fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sources = []
            for source in config["sources"]:
                for field in ("name", "license", "attribution", "snapshot", "paths"):
                    if not source.get(field):
                        raise ValueError("Missing provenance: " + field)
                for name in source["paths"]:
                    p = Path(name)
                    sources.append((source, p, digest(p)))
            identity = hashlib.sha256(
                json.dumps(
                    ["codexa-data-m3.1", config, [(str(p), h) for _, p, h in sources]],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            db = sqlite3.connect(root / "documents.sqlite")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(f"PRAGMA cache_size={-cache_mib * 1024}")
            db.executescript(
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY,source TEXT,split TEXT,text TEXT,signature TEXT); CREATE TABLE IF NOT EXISTS bands(band TEXT,id TEXT); CREATE INDEX IF NOT EXISTS bands_key ON bands(band); CREATE TABLE IF NOT EXISTS errors(source TEXT,ordinal INT,reason TEXT,PRIMARY KEY(source,ordinal));"
            )
            db.execute("CREATE TABLE IF NOT EXISTS split_groups(id TEXT PRIMARY KEY,split TEXT)")
            prior = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if prior and prior[0] != identity:
                raise ValueError("Preparation input/configuration changed")
            if (root / "manifest.json").exists():
                manifest = json.loads((root / "manifest.json").read_text())
                if digest(root / "documents.sqlite") != manifest["database_sha256"]:
                    raise ValueError("Curated database changed")
                db.close()
                return root
            if wal:
                db.execute("PRAGMA journal_mode=WAL")
            # Holdout removal looks up a document's bands by id, not by band hash.
            # Add this only to unfinished databases; completed checksums must stay immutable.
            db.execute("CREATE INDEX IF NOT EXISTS bands_id ON bands(id)")
            db.execute("INSERT OR IGNORE INTO meta VALUES('identity',?)", (identity,))
            db.commit()
            # Declared holdouts are visited before training, so matching train documents cannot win.
            sources.sort(key=lambda s: (s[0].get("split", "train") == "train", str(s[1])))
            max_bytes = config.get("max_document_bytes", 1048576)
            if not 0 < max_bytes <= 128 * 1048576:
                raise ValueError("Invalid document size bound")
            if not 0 < config.get("near_threshold", 0.85) <= 1:
                raise ValueError("Invalid similarity threshold")
            seed = str(config.get("seed", 42))
            fractions = config.get("fractions", [0.99, 0.005, 0.005])
            if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9 or min(fractions) <= 0:
                raise ValueError("Invalid fractions")
            processed = sum(
                int(r[0]) for r in db.execute("SELECT value FROM meta WHERE key LIKE 'cursor:%'")
            )
            for source, path, _ in sources:
                key = "cursor:" + str(path)
                start = int(
                    (db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone() or ["0"])[
                        0
                    ]
                )
                byte_start = int(
                    (
                        db.execute(
                            "SELECT value FROM meta WHERE key=?", ("byte:" + str(path),)
                        ).fetchone()
                        or ["0"]
                    )[0]
                )
                for ordinal, (byte_position, row) in enumerate(
                    records(path, source.get("format", "jsonl"), max_bytes, start, byte_start),
                    start + 1,
                ):
                    text = row.get(source.get("text_field", "text"))
                    reason = row.get("error")
                    try:
                        encoded = text.encode("utf-8") if isinstance(text, str) else b""
                    except UnicodeError:
                        text = None
                        encoded = b""
                        reason = "encoding"
                    if not isinstance(text, str):
                        reason = reason or "missing_text"
                    elif not text.strip():
                        reason = "empty"
                    elif len(encoded) > max_bytes:
                        reason = "oversize"
                    elif len(text.strip()) < source.get("min_chars", config.get("min_chars", 32)):
                        reason = "short"
                    elif config.get("reject_reserved_tokens", False) and any(
                        token in text for token in ("<pad>", "<unk>", "<bos>", "<eos>")
                    ):
                        reason = "reserved_token"
                    elif "\x00" in text or "\ufffd" in text:
                        reason = "corruption"
                    elif len(text) > config.get("repetition_min_chars", 512) and len(
                        set(text.split())
                    ) < config.get("minimum_unique_words", 8):
                        reason = "repetition"
                    if reason:
                        db.execute(
                            "INSERT OR REPLACE INTO errors VALUES(?,?,?)",
                            (str(path), ordinal, str(reason)),
                        )
                    else:
                        assert isinstance(text, str)
                        canonical = " ".join(text.split())
                        docid = hashlib.sha256(canonical.encode()).hexdigest()
                        split = source.get("split") or row.get("split")
                        if split == "validation":
                            split = "val"
                        if split is None:
                            group = row.get(
                                source.get("group_field", "group_id"),
                                str(path) if source.get("format") == "text" else docid,
                            )
                            fraction = (
                                int(
                                    hashlib.sha256((seed + str(group)).encode()).hexdigest()[:16],
                                    16,
                                )
                                / 2**64
                            )
                            split = (
                                "train"
                                if fraction < fractions[0]
                                else "val"
                                if fraction < sum(fractions[:2])
                                else "test"
                            )
                        if split not in ("train", "val", "test"):
                            raise ValueError("Invalid split")
                        parent = row.get(
                            source.get("group_field", "group_id"),
                            str(path) if source.get("format") == "text" else None,
                        )
                        group_conflict = False
                        if parent is not None:
                            known = db.execute(
                                "SELECT split FROM split_groups WHERE id=?", (str(parent),)
                            ).fetchone()
                            group_conflict = bool(known and known[0] != split)
                            db.execute(
                                "INSERT OR IGNORE INTO split_groups VALUES(?,?)",
                                (str(parent), split),
                            )
                        duplicate = db.execute(
                            "SELECT split FROM docs WHERE id=?", (docid,)
                        ).fetchone()
                        duplicate = duplicate or group_conflict
                        sig = shingles(text) if config.get("near_duplicates", True) else set()
                        # 64 deterministic minhash values, 16 bands; verify actual Jaccard before removal.
                        values = np.fromiter((int(v, 16) for v in sig), dtype=np.uint64)
                        mins = []
                        for i in range(64) if sig else []:
                            permutation_key = hashlib.blake2b(
                                str(i).encode(), digest_size=16
                            ).digest()
                            av = np.uint64(int.from_bytes(permutation_key[:8], "little") | 1)
                            bv = np.uint64(int.from_bytes(permutation_key[8:], "little"))
                            mins.append(f"{int(np.min(values * av + bv)):016x}")
                        bands = (
                            [str(i) + ":" + "".join(mins[i * 4 : i * 4 + 4]) for i in range(16)]
                            if mins
                            else []
                        )
                        if not duplicate:
                            query = (
                                "SELECT DISTINCT id FROM bands WHERE band IN ("
                                + ",".join("?" for _ in bands)
                                + ")"
                            )
                            for (candidate,) in db.execute(query, bands) if bands else []:
                                other = db.execute(
                                    "SELECT signature FROM docs WHERE id=?", (candidate,)
                                ).fetchone()
                                other_set = set(json.loads(other[0]))
                                if len(sig & other_set) / len(sig | other_set) >= config.get(
                                    "near_threshold", 0.85
                                ):
                                    duplicate = True
                                    break
                        if duplicate:
                            db.execute(
                                "INSERT OR REPLACE INTO errors VALUES(?,?,?)",
                                (
                                    str(path),
                                    ordinal,
                                    "group_overlap" if group_conflict else "duplicate",
                                ),
                            )
                        else:
                            db.execute(
                                "INSERT INTO docs VALUES(?,?,?,?,?)",
                                (docid, source["name"], split, text, json.dumps(sorted(sig))),
                            )
                            db.executemany(
                                "INSERT INTO bands VALUES(?,?)", [(b, docid) for b in bands]
                            )
                    db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, str(ordinal)))
                    db.execute(
                        "INSERT OR REPLACE INTO meta VALUES(?,?)",
                        ("byte:" + str(path), str(max(0, byte_position))),
                    )
                    processed += 1
                    if ordinal % commit_every == 0:
                        db.commit()
                        if progress:
                            progress(processed)
                db.commit()
            if config.get("protect_spans", True):
                db.execute("CREATE TABLE IF NOT EXISTS protected(gram TEXT PRIMARY KEY)")
                for (text,) in db.execute("SELECT text FROM docs WHERE split!='train'"):
                    db.executemany(
                        "INSERT OR IGNORE INTO protected VALUES(?)", ((v,) for v in spans(text))
                    )
                protected_cache: set[str] | None = None
                if protected_cache_entries:
                    protected_cache = set()
                    for (gram,) in db.execute("SELECT gram FROM protected"):
                        protected_cache.add(gram)
                        if len(protected_cache) > protected_cache_entries:
                            protected_cache = None
                            break
                for docid, text in db.execute("SELECT id,text FROM docs WHERE split='train'"):
                    if any(
                        v in protected_cache
                        if protected_cache is not None
                        else db.execute("SELECT 1 FROM protected WHERE gram=?", (v,)).fetchone()
                        for v in spans(text)
                    ):
                        db.execute("DELETE FROM docs WHERE id=?", (docid,))
                        db.execute("DELETE FROM bands WHERE id=?", (docid,))
                        db.execute(
                            "INSERT OR REPLACE INTO errors VALUES(?,?,?)",
                            ("holdout-protection", int(docid[:15], 16), "holdout_overlap"),
                        )
                db.commit()
            if progress:
                progress(processed)
            counts = dict(db.execute("SELECT split,count(*) FROM docs GROUP BY split"))
            errors = dict(db.execute("SELECT reason,count(*) FROM errors GROUP BY reason"))
            if wal:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.execute("PRAGMA journal_mode=DELETE")
            db.close()
            if any(not counts.get(s) for s in ("train", "val", "test")):
                raise ValueError("Each split needs documents; adjust source groups/fractions")
            manifest = {
                "version": 2,
                "stage": "curated",
                "pipeline": "codexa-data-m3.1",
                "identity": identity,
                "config": config,
                "counts": counts,
                "errors": errors,
                "database_sha256": digest(root / "documents.sqlite"),
                "source_fingerprints": [{"path": str(p), "sha256": h} for _, p, h in sources],
            }
            atomic(root / "manifest.json", manifest)
            return root
        finally:
            if db is not None:
                db.close()


def documents(root, split="train", offset=0):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "documents.sqlite") != manifest["database_sha256"]:
        raise ValueError("Curated database changed")
    db = sqlite3.connect(
        f"file:{root.resolve()}/documents.sqlite?mode=ro", uri=True, check_same_thread=False
    )
    try:
        for (text,) in db.execute(
            "SELECT text FROM docs WHERE split=? ORDER BY id LIMIT -1 OFFSET ?", (split, offset)
        ):
            yield text
    finally:
        db.close()


def tokenize(dataset, tokenizer, output, resume=False, shard_tokens=67108864):
    from codexa.tokenizer.bpe import CodexaTokenizer

    root = Path(output)
    source = Path(dataset)
    identity = {"curated": digest(source / "manifest.json"), "tokenizer": digest(tokenizer)}
    if root.exists() and not resume:
        raise FileExistsError(root)
    if resume and not (root / "progress.json").exists():
        raise ValueError("Resume requires existing tokenization progress")
    root.mkdir(parents=True, exist_ok=True)
    statepath = root / "progress.json"
    state: Any = (
        json.loads(statepath.read_text())
        if statepath.exists()
        else {"identity": identity, "splits": {}}
    )
    if state["identity"] != identity:
        raise ValueError("Tokenization inputs changed")
    if not statepath.exists():
        atomic(statepath, state)
    freeze = Path(tokenizer).parent / "identity.json"
    if freeze.exists() and json.loads(freeze.read_text())["sha256"] != digest(tokenizer):
        raise ValueError("Frozen tokenizer modified")
    tok = CodexaTokenizer.load(tokenizer)
    if tok.backend.get_vocab_size() > 65536:
        raise ValueError("uint16 vocabulary overflow")
    for split in ("train", "val", "test"):
        result = state["splits"].setdefault(split, [])
        for entry in result:
            if digest(root / entry["file"]) != entry["sha256"]:
                raise ValueError("Committed shard changed")
        committed = sum(r["documents"] for r in result)
        stream = None
        offsets = None
        count = docs = 0

        def commit(split=split, result=result):
            nonlocal stream, count, docs, offsets
            if stream is None:
                return
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            assert offsets is not None
            offsets.flush()
            os.fsync(offsets.fileno())
            offsets.close()
            name = f"{split}-{len(result):05}.bin"
            (root / f"{split}.partial").replace(root / name)
            (root / f"{split}.offsets.partial").replace(root / (name + ".idx"))
            result.append(
                {
                    "file": name,
                    "tokens": count,
                    "documents": docs,
                    "sha256": digest(root / name),
                    "index": name + ".idx",
                    "index_sha256": digest(root / (name + ".idx")),
                }
            )
            atomic(statepath, state)
            stream = None
            offsets = None
            count = docs = 0

        for text in documents(source, split, committed):
            ids = [2, *tok.encode(text), 3]
            if max(ids) >= 65536:
                raise ValueError("Token ID overflow")
            if stream is not None and count + len(ids) > shard_tokens:
                commit()
            if stream is None:
                tmp = root / f"{split}.partial"
                stream = tmp.open("wb")
                offsets = (root / f"{split}.offsets.partial").open("wb")
            assert offsets is not None
            offsets.write(count.to_bytes(8, "little"))
            stream.write(np.asarray(ids, dtype="<u2").tobytes())
            count += len(ids)
            docs += 1
        commit()
    manifest = {
        "version": 2,
        "stage": "tokenized",
        "identity": identity,
        "tokenizer_hash": identity["tokenizer"],
        "vocab_size": tok.backend.get_vocab_size(),
        "dtype": "<u2",
        "boundary": "BOS document EOS; causal attention across boundaries",
        "dataset_name": json.loads((source / "manifest.json").read_text())["config"].get(
            "name", source.name
        ),
        "provenance": json.loads((source / "manifest.json").read_text()),
        "splits": state["splits"],
    }
    atomic(root / "manifest.json", manifest)
    verify(root)
    return root


def verify(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("version") != 2
        or m.get("dtype") != "<u2"
        or not 260 <= m.get("vocab_size", 0) <= 65536
    ):
        raise ValueError("Invalid dataset schema")
    if m.get("documents_sha256"):
        listing = root / "documents.jsonl"
        if listing.is_symlink() or digest(listing) != m["documents_sha256"]:
            raise ValueError("Document provenance index mismatch")
    for entries in m["splits"].values():
        for entry in entries:
            name = entry["file"]
            if Path(name).name != name:
                raise ValueError("Unsafe shard path")
            if entry.get("index"):
                ix = root / entry["index"]
                if (
                    Path(entry["index"]).name != entry["index"]
                    or ix.is_symlink()
                    or ix.stat().st_size != entry["documents"] * 8
                    or digest(ix) != entry["index_sha256"]
                ):
                    raise ValueError("Document index mismatch")
            p = root / name
            if (
                p.is_symlink()
                or p.stat().st_size != entry["tokens"] * 2
                or digest(p) != entry["sha256"]
            ):
                raise ValueError("Shard integrity mismatch")
            tokens = np.memmap(p, dtype="<u2", mode="r")
            if len(tokens) == 0 or int(tokens.max()) >= m["vocab_size"]:
                raise ValueError("Token outside declared vocabulary")
            if entry.get("index"):
                offsets = np.memmap(root / entry["index"], dtype="<u8", mode="r")
                if (
                    len(offsets) == 0
                    or offsets[0] != 0
                    or np.any(offsets[1:] <= offsets[:-1])
                    or offsets[-1] >= len(tokens)
                    or np.any(tokens[offsets] != 2)
                    or np.any(tokens[offsets[1:] - 1] != 3)
                    or tokens[-1] != 3
                ):
                    raise ValueError("Invalid document boundaries or offsets")
    return m


class ShardedData:
    def __init__(self, root, split, sequence):
        self.root = Path(root)
        self.sequence = sequence
        self.manifest = verify(root)
        self.entries = self.manifest["splits"][split]
        self.ends = np.cumsum([r["tokens"] for r in self.entries])
        self.size = (int(self.ends[-1]) - 1) // sequence
        if self.size < 1:
            raise ValueError("Split too small")
        self.maps = {}

    def batch(self, indices, device):
        import torch

        rows = np.empty((len(indices), self.sequence + 1), dtype=np.int64)
        for row, index in zip(rows, indices):
            if not 0 <= index < self.size:
                raise IndexError(index)
            position = index * self.sequence
            written = 0
            while written < len(row):
                shard = int(np.searchsorted(self.ends, position, side="right"))
                start = int(self.ends[shard - 1]) if shard else 0
                if shard not in self.maps:
                    if len(self.maps) >= 4:
                        self.maps.pop(next(iter(self.maps)))
                    self.maps[shard] = np.memmap(
                        self.root / self.entries[shard]["file"], dtype="<u2", mode="r"
                    )
                array = self.maps[shard]
                offset = position - start
                n = min(len(row) - written, len(array) - offset)
                row[written : written + n] = array[offset : offset + n]
                written += n
                position += n
        t = torch.from_numpy(rows).to(device)
        return t[:, :-1], t[:, 1:]


class PermutationSampler:
    """Six-round balanced Feistel permutation, cycle-walked to the dataset domain."""

    def __init__(self, size, seed):
        self.size, self.seed, self.epoch, self.position = size, seed, 0, 0
        self.bits = max(1, ((size - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.bits) - 1

    def permute(self, x):
        while True:
            left, right = x >> self.bits, x & self.mask
            for round in range(6):
                key = f"{self.seed}:{self.epoch}:{round}:{right}".encode()
                value = (
                    int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
                    & self.mask
                )
                left, right = right, left ^ value
            x = (left << self.bits) | right
            if x < self.size:
                return x

    def next(self, count):
        result = []
        for _ in range(count):
            if self.position == self.size:
                self.epoch += 1
                self.position = 0
            result.append(self.permute(self.position))
            self.position += 1
        return result

    def state_dict(self):
        return {
            "algorithm": "feistel-v1",
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state):
        if not 0 <= state["position"] <= self.size or state["epoch"] < 0:
            raise ValueError("Invalid sampler cursor")
        if (
            state["algorithm"] != "feistel-v1"
            or state["size"] != self.size
            or state["seed"] != self.seed
        ):
            raise ValueError("Sampler mismatch")
        self.epoch, self.position = state["epoch"], state["position"]
