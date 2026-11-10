"""Disk-backed, compact deduplication shared by all dataset-v2 sources."""

import fcntl
import json
import sqlite3
import zlib
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from codexa.data.production import atomic, digest
from codexa.data.v2_quality import WORD, bands, canonical, features, rolling, shared_passage
from codexa.data.v2_storage import Storage


def _reference_features(row: tuple[str, str]) -> tuple[str, bytes, list[tuple[bytes, int]]]:
    ident, text = row
    _, signature, anchors = features(text)
    return ident, signature, anchors


class Index:
    def __init__(self, storage: Storage, historical: Path):
        storage.check(64 * 1024**2)
        self.lock = (storage.root / "tmp/index-writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another reference/preparation writer owns this root") from None
        self.closed = False
        self.shingle_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.shingle_bytes = 0
        self.storage = storage
        self.historical = historical
        self.old = sqlite3.connect(f"file:{historical}?mode=ro&immutable=1", uri=True)
        self.db = sqlite3.connect(storage.root / "indexes" / "documents.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA cache_size=-262144")
        self.db.execute("PRAGMA wal_autocheckpoint=20000")
        self.db.execute("PRAGMA max_page_count=4194304")  # 16 GiB at the default 4 KiB/page.
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS docs("
            "id INTEGER PRIMARY KEY,hash BLOB UNIQUE,role TEXT,source TEXT,category TEXT,"
            "split TEXT,payload BLOB,signature BLOB,metadata TEXT,tokens INTEGER);"
            "CREATE TABLE IF NOT EXISTS bands(band BLOB,doc INTEGER,PRIMARY KEY(band,doc)) WITHOUT ROWID;"
            "CREATE TABLE IF NOT EXISTS anchors(hash BLOB,doc INTEGER,pos INTEGER,"
            "PRIMARY KEY(hash,doc,pos)) WITHOUT ROWID;"
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE IF NOT EXISTS errors(source TEXT,reason TEXT,n INTEGER,"
            "PRIMARY KEY(source,reason)) WITHOUT ROWID;"
        )

    def value(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value)))

    def reject(self, source: str, reason: str) -> None:
        self.db.execute(
            "INSERT INTO errors VALUES(?,?,1) ON CONFLICT(source,reason) DO UPDATE SET n=n+1",
            (source, reason),
        )

    def text(self, ident: int) -> str:
        row = self.db.execute("SELECT hash,role,payload FROM docs WHERE id=?", (ident,)).fetchone()
        if row[1] == "history":
            old = self.old.execute("SELECT text FROM docs WHERE id=?", (row[0].hex(),)).fetchone()
            if not old:
                raise ValueError("Historical reference disappeared")
            return old[0]
        return zlib.decompress(row[2]).decode()

    def insert(
        self,
        ident: bytes,
        source: str,
        category: str,
        split: str,
        text: str,
        signature: bytes,
        anchors: list[tuple[bytes, int]],
        metadata: dict[str, Any],
        tokens: int = 0,
        role: str = "new",
    ) -> int:
        cursor = self.db.execute(
            "INSERT INTO docs(hash,role,source,category,split,payload,signature,metadata,tokens)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                ident,
                role,
                source,
                category,
                split,
                zlib.compress(text.encode(), 6) if role != "history" else b"",
                signature,
                json.dumps(metadata, sort_keys=True),
                tokens,
            ),
        )
        number = int(cursor.lastrowid or 0)
        self.db.executemany(
            "INSERT OR IGNORE INTO bands VALUES(?,?)", ((b, number) for b in bands(signature))
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO anchors VALUES(?,?,?)", ((h, number, p) for h, p in anchors)
        )
        return number

    def reference(self, workers: int = 4) -> None:
        if self.value("historical_complete", False):
            return
        import time

        start = time.monotonic()
        last = self.value("historical_cursor", "")
        cursor = self.old.execute("SELECT id,text FROM docs WHERE id>? ORDER BY id", (last,))
        count = self.value("historical_documents", 0)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            while rows := cursor.fetchmany(512):
                self.storage.check(256 * 1024**2)
                for ident, signature, anchors in pool.map(_reference_features, rows, chunksize=16):
                    self.insert(
                        bytes.fromhex(ident),
                        "historical",
                        "reference",
                        "protected",
                        "",
                        signature,
                        anchors,
                        {},
                        role="history",
                    )
                    last = ident
                    count += 1
                self.set("historical_cursor", last)
                self.set("historical_documents", count)
                self.db.commit()
                if count % 8192 == 0:
                    print(
                        json.dumps(
                            {"reference_documents": count, "seconds": time.monotonic() - start}
                        ),
                        flush=True,
                    )
        self.set("historical_complete", True)
        self.db.commit()
        atomic(
            self.storage.root / "reports" / "historical-index.json",
            {
                "documents": count,
                "input": str(self.historical),
                "complete": True,
                "source_manifest_sha256": digest(self.historical.parent / "manifest.json"),
            },
        )

    def duplicate(
        self,
        ident: bytes,
        text: str,
        shingles: np.ndarray,
        signature: bytes,
        anchors: list[tuple[bytes, int]],
    ) -> str | None:
        if self.db.execute("SELECT 1 FROM docs WHERE hash=?", (ident,)).fetchone():
            return "exact_document"
        candidates: set[int] = set()
        for band in bands(signature):
            for (number,) in self.db.execute(
                "SELECT doc FROM bands WHERE band=? LIMIT 258", (band,)
            ):
                candidates.add(number)
                if len(candidates) > 256:
                    return "near_candidate_overflow"
        for number in sorted(candidates):
            other = self.shingles(number)
            union = (
                len(shingles)
                + len(other)
                - len(np.intersect1d(shingles, other, assume_unique=True))
            )
            if union and (len(shingles) + len(other) - union) / union >= 0.85:
                return "near_document"
        comparisons = 0
        for anchor, pos in anchors:
            matches = self.db.execute(
                "SELECT doc,pos FROM anchors WHERE hash=? LIMIT 258", (anchor,)
            ).fetchall()
            if len(matches) > 256:
                # Ignore non-discriminative common anchors; verify other anchors and MinHash.
                continue
            for number, otherpos in matches:
                comparisons += 1
                if comparisons > 4096:
                    return "passage_comparison_overflow"
                if shared_passage(text, self.text(number), pos, otherpos):
                    return "shared_128_token_passage"
        return None

    def shingles(self, number: int) -> np.ndarray:
        """Exact verification needs only five-grams; cache immutable rows within 128 MiB."""
        if number in self.shingle_cache:
            self.shingle_cache.move_to_end(number)
            return self.shingle_cache[number]
        result = np.unique(rolling(WORD.findall(self.text(number).lower()), 5))
        self.shingle_cache[number] = result
        self.shingle_bytes += result.nbytes
        while self.shingle_bytes > 128 * 1024**2 or len(self.shingle_cache) > 4096:
            _, discarded = self.shingle_cache.popitem(last=False)
            self.shingle_bytes -= discarded.nbytes
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.shingle_cache.clear()
        # Only explicit commit points publish cursors and corresponding token shards.
        try:
            self.db.rollback()
        finally:
            try:
                self.db.close()
            finally:
                self.old.close()
                self.lock.close()


class Protection:
    """Exact protected spans; variable-length benchmark questions, never bare answers."""

    def __init__(self, items: dict[int, np.ndarray]):
        self.items = {k: np.unique(v) for k, v in items.items() if len(v)}

    @classmethod
    def build(cls, texts: list[tuple[str, bool | int]]) -> "Protection":
        collected: dict[int, list[np.ndarray]] = {}
        for text, question in texts:
            words = WORD.findall(canonical(text).lower())
            size = (
                (min(13, len(words)) if question else 13)
                if isinstance(question, bool)
                else question
            )
            if size >= 4:
                collected.setdefault(size, []).append(rolling(words, size))
        return cls({k: np.unique(np.concatenate(v)) for k, v in collected.items()})

    def matches(self, text: str) -> bool:
        words = WORD.findall(canonical(text).lower())
        for size, protected in self.items.items():
            grams = rolling(words, size)
            places = np.searchsorted(protected, grams)
            valid = places < len(protected)
            if np.any(protected[places[valid]] == grams[valid]):
                return True
        return False

    def save(self, path: Path) -> None:
        arrays: Any = {str(k): v for k, v in self.items.items()}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Protection":
        with np.load(path) as z:
            return cls({int(k): z[k] for k in z.files})
