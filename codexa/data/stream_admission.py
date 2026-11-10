"""Wikimedia metadata admission and native duplicate checks without a text archive."""

import gzip
import http.client
import json
import re
import sqlite3
import time
import zlib
from collections import OrderedDict
from typing import Any

from codexa.data.streaming import open_provider
from codexa.data.v2_index import Index
from codexa.data.v2_quality import canonical, features

# Source categories are frozen before review; no individual-record repair or scoring.
GENERAL_ENGLISH_SITES = frozenset(
    f"{name}.stackexchange.com"
    for name in (
        "academia",
        "aviation",
        "bicycles",
        "biology",
        "cooking",
        "diy",
        "earthscience",
        "economics",
        "ell",
        "english",
        "gardening",
        "history",
        "interpersonal",
        "law",
        "linguistics",
        "literature",
        "money",
        "movies",
        "music",
        "outdoors",
        "parenting",
        "pets",
        "philosophy",
        "politics",
        "scifi",
        "travel",
        "workplace",
        "worldbuilding",
        "writers",
    )
)
STACKEXCHANGE_LICENSES = frozenset(
    "Creative Commons - Attribution Share-Alike - "
    f"https://creativecommons.org/licenses/by-sa/{version}/"
    for version in ("2.5", "3.0", "4.0")
)


def stackexchange_discussion(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    text = row.get("text")
    if not isinstance(metadata, dict):
        return False
    site = metadata.get("site")
    licenses = metadata.get("all_licenses")
    ident = row.get("id")
    return (
        site in GENERAL_ENGLISH_SITES
        and isinstance(ident, str)
        and re.fullmatch(r"[0-9]+", ident) is not None
        and metadata.get("url") == f"https://{site}/questions/{ident}"
        and metadata.get("license") in STACKEXCHANGE_LICENSES
        and isinstance(licenses, list)
        and bool(licenses)
        and all(license in STACKEXCHANGE_LICENSES for license in licenses)
        and isinstance(text, str)
        and len(text.split()) >= 100
    )


def wikipedia_navigation(row: dict[str, Any]) -> bool:
    """Exclude explicit disambiguation using preserved title and introductory text."""
    metadata = row.get("metadata", {})
    title = str(metadata.get("title", "")).casefold() if isinstance(metadata, dict) else ""
    if title.endswith("(disambiguation)"):
        return True
    text = row.get("text")
    if not isinstance(text, str):
        return False
    prefix = text[:2048].casefold()
    return bool(
        re.search(r"\bmay (?:also )?refer to\s*:", prefix)
        and re.search(r"^\s*\*\s+", prefix, re.MULTILINE)
    )


def wikimedia_article(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    text = row.get("text")
    return (
        isinstance(metadata, dict)
        and metadata.get("wiki") == "wikipedia.com"
        and isinstance(row.get("id"), str)
        and re.fullmatch(r"0-[0-9]+", row["id"]) is not None
        and str(metadata.get("namespace")) == "0"
        and metadata.get("license")
        == "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"
        and isinstance(metadata.get("url"), str)
        and metadata["url"].startswith("https://wikipedia.com/wiki/")
        and isinstance(text, str)
        and len(text.split()) >= 100
        and not wikipedia_navigation(row)
    )


class OnlineDuplicates(Index):
    """Reuse native thresholds and verification; checkpoint locations, never document text."""

    def __init__(self, shards: list[dict[str, str]], opener=open_provider):
        self.shards, self.opener = shards, opener
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(
            "CREATE TABLE docs(id INTEGER PRIMARY KEY, hash BLOB UNIQUE, shard INTEGER, record INTEGER, attribution TEXT); CREATE TABLE bands(band BLOB,doc INTEGER); CREATE INDEX band_lookup ON bands(band); CREATE TABLE anchors(hash BLOB,doc INTEGER,pos INTEGER); CREATE INDEX anchor_lookup ON anchors(hash);"
        )
        self.shingle_cache = OrderedDict()
        self.shingle_bytes = 0
        self.cache: OrderedDict[int, str] = OrderedDict()
        self.cache_bytes = 0

    def text(self, ident: int) -> str:
        if ident not in self.cache:
            shard, record = self.db.execute(
                "SELECT shard,record FROM docs WHERE id=?", (ident,)
            ).fetchone()
            for attempt in range(4):
                raw = None
                try:
                    raw = self.opener(self.shards[shard])
                    with gzip.GzipFile(fileobj=raw) as stream:
                        for _ in range(record):
                            line = stream.readline()
                            if not line:
                                raise ValueError("Duplicate reference record missing")
                        text = json.loads(line)["text"]
                    break
                except (OSError, EOFError, http.client.IncompleteRead, zlib.error):
                    if attempt == 3:
                        raise
                    time.sleep(min(2**attempt, 8))
                finally:
                    if raw is not None:
                        raw.close()
            self._cache(ident, text)
        return self.cache[ident]

    def _cache(self, ident: int, text: str) -> None:
        self.cache[ident] = text
        self.cache_bytes += len(text.encode())
        while self.cache_bytes > 32 * 1024**2 and len(self.cache) > 1:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= len(old.encode())

    def add(
        self, text: str, shard: int, record: int, attribution: dict[str, Any] | None = None
    ) -> str | None:
        import hashlib

        from codexa.data.v2_quality import bands

        ident = hashlib.sha256(canonical(text).encode()).digest()
        shingles, signature, anchors = features(text)
        reason = self.duplicate(ident, text, shingles, signature, anchors)
        if reason:
            return reason
        cursor = self.db.execute(
            "INSERT INTO docs(hash,shard,record,attribution) VALUES(?,?,?,?)",
            (ident, shard, record, json.dumps(attribution or {}, sort_keys=True)),
        )
        number = int(cursor.lastrowid or 0)
        self.db.executemany(
            "INSERT INTO bands VALUES(?,?)", ((b, number) for b in bands(signature))
        )
        self.db.executemany(
            "INSERT INTO anchors VALUES(?,?,?)", ((h, number, p) for h, p in anchors)
        )
        self._cache(number, text)
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"database": self.db.serialize()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.db.deserialize(state["database"])
        self._clear_cache()

    def _clear_cache(self) -> None:
        self.cache.clear()
        self.cache_bytes = 0
        self.shingle_cache.clear()
        self.shingle_bytes = 0

    def begin(self) -> None:
        self.db.execute("SAVEPOINT batch")

    def commit(self) -> None:
        self.db.execute("RELEASE batch")

    def rollback(self) -> None:
        self.db.execute("ROLLBACK TO batch")
        self.db.execute("RELEASE batch")
        self._clear_cache()
