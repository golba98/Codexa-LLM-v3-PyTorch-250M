"""Bounded language-corpus comparison and preparation; no training entry point.

Network range samples are development evidence, never verified complete source files.
Source admission requires an explicit frozen development/rights decision before prepare.
"""

import argparse
import gzip
import hashlib
import heapq
import io
import json
import math
import sqlite3
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Generator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from codexa.data.production import atomic, digest, verify
from codexa.data.v2 import records
from codexa.data.v2_index import Index, Protection
from codexa.data.v2_policy import admission_error
from codexa.data.v2_quality import canonical, clean, features, source_error, source_text
from codexa.data.v2_storage import GIB, Storage
from codexa.evaluation.protection import QuestionProtection
from codexa.tokenizer.bpe import CodexaTokenizer

VERSION = "codexa-language-data-1"
V2_VERSION = "codexa-language-data-2"
V3_VERSION = "codexa-language-data-3"
V4_VERSION = "codexa-language-data-4"
SUPPORTED_VERSIONS = {VERSION, V2_VERSION, V3_VERSION, V4_VERSION}
V3_PRIOR_ACQUISITION_BYTES = 859990708
TOKENIZER_SHA = "60a5e06447542b9462d458577f4a3056551c135e5aa1b8859d39624a3eee8771"
SPLIT_TARGETS = {"train": 60_000_000, "val": 1_000_000, "test": 1_000_000}
V3_MIN_TRAIN_TOKENS = 50_003_969
V3_MIN_EVALUATION_TOKENS = 1_000_000
NARRATIVE_VALIDATION_FILE = "tinystories-gpt4--TinyStoriesV2-GPT4-valid.txt"
NARRATIVE_VALIDATION_SHA = "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
NARRATIVE_VALIDATION_BYTES = 22_502_601


class RangeReader(io.RawIOBase):
    """Seekable HTTP range reader with validated offsets and a hard wire-byte ceiling."""

    def __init__(self, url: str, size: int, cache: Path, maximum: int):
        super().__init__()
        self.url, self.size, self.cache, self.maximum = url, size, cache, maximum
        self.position = 0
        self.transferred = 0
        self.requests: list[dict[str, Any]] = []
        self.attempted: list[dict[str, int]] = []
        self.attempted_bytes = 0
        cache.mkdir(parents=True, exist_ok=False)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if position < 0 or whence not in {0, 1, 2}:
            raise ValueError("Invalid range seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if self.attempted_bytes + size > self.maximum:
            raise OSError("Development network byte budget exceeded")
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        self.attempted.append({"start": start, "end": end, "bytes": size})
        self.attempted_bytes += size
        atomic(
            self.cache / "attempted-ranges.json",
            {"url": self.url, "attempted_bytes": self.attempted_bytes, "requests": self.attempted},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError("Server ignored range or changed source size")
            payload = response.read(size + 1)
        if len(payload) != size:
            raise ValueError("Incomplete or excessive range payload")
        sha = hashlib.sha256(payload).hexdigest()
        name = f"{len(self.requests):05d}-{start}-{end}.bin"
        (self.cache / name).write_bytes(payload)
        self.requests.append({"start": start, "end": end, "sha256": sha, "file": name})
        self.transferred += size
        self.position += size
        return payload


def load_sources(config: Path, reference: Path) -> list[dict[str, Any]]:
    policy = yaml.safe_load(config.read_text())
    result = []
    for source in policy["sources"]:
        if source.get("local_glob"):
            paths = sorted((reference / "raw").glob(source["local_glob"]))
            if not paths:
                raise FileNotFoundError(source["local_glob"])
            for path in paths:
                receipt = path.with_name(path.name + ".receipt.json")
                spec = json.loads(receipt.read_text())
                result.append(
                    spec | source | {"local_path": str(path), "receipt_sha256": digest(receipt)}
                )
        else:
            if len(source.get("revision", "")) != 40 or len(source.get("sha256", "")) != 64:
                raise ValueError("Pinned source revision and payload identity required")
            result.append(source)
    return result


def source_rows(
    spec: dict[str, Any], cache: Path, budget: int
) -> Generator[tuple[int, dict[str, Any]], None, None]:
    if spec.get("local_path"):
        path = Path(spec["local_path"])
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError("Local source identity mismatch")
        from codexa.data.language_sources import ADAPTERS

        if spec.get("adapter") in ADAPTERS and path.name.endswith(".json.gz"):
            limit = spec.get("_development_limit")
            total = spec.get("rows", 54455)
            selected = (
                {
                    min(total - 1, int((i + 0.5) * total / limit)) + 1
                    for i in range(min(limit, total))
                }
                if limit
                else None
            )
            with gzip.open(path, "rb") as stream:
                ordinal = 0
                while line := stream.readline(2 * 1024**2 + 1):
                    ordinal += 1
                    if len(line) > 2 * 1024**2:
                        while line and not line.endswith(b"\n"):
                            line = stream.readline(2 * 1024**2 + 1)
                        yield ordinal, {"error": "oversize"}
                        continue
                    if selected is not None and ordinal not in selected:
                        continue
                    try:
                        row = json.loads(line)
                        yield ordinal, row if isinstance(row, dict) else {"error": "record"}
                    except (ValueError, UnicodeError):
                        yield ordinal, {"error": "encoding_or_json"}
            return
        if (
            spec.get("adapter") not in ADAPTERS
            and spec.get("adapter") != "codexa-common-modern-english-1"
        ):
            yield from records(path, spec)
            return
        import pyarrow.parquet as pq

        yield from parquet_rows(pq.ParquetFile(path), spec)
        return
    import pyarrow.parquet as pq

    if str(spec.get("path", "")).endswith(".json.gz"):
        raise ValueError("Acquire the complete pinned gzip payload before profiling")
    url = (
        f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    )
    reader = RangeReader(url, spec["bytes"], cache, budget)
    try:
        parquet = pq.ParquetFile(reader)
        yield from parquet_rows(parquet, spec)
    finally:
        atomic(
            cache / "ranges.json",
            {
                "url": url,
                "upstream_expected_sha256": spec["sha256"],
                "complete_payload_verified": False,
                "transferred_bytes": reader.transferred,
                "attempted_bytes": reader.attempted_bytes,
                "ranges": reader.requests,
            },
        )
        reader.close()


def parquet_rows(
    parquet: Any, spec: dict[str, Any]
) -> Generator[tuple[int, dict[str, Any]], None, None]:
    """Read complete metadata and sample distributed global ordinals in development."""
    from codexa.data.common_corpus import COLUMNS

    fields = (
        COLUMNS
        if spec.get("adapter") == "codexa-common-modern-english-1"
        else (
            "text",
            "url",
            "id",
            "title",
            "audience",
            "format",
            "seed_data",
            "edu_int_score",
            "edu_score",
            "language",
            "language_score",
            "fasttext_score",
            "int_score",
            "score",
        )
    )
    columns = [k for k in fields if k in parquet.schema_arrow.names]
    limit = spec.get("_development_limit")
    total = parquet.metadata.num_rows
    selected = (
        {min(total - 1, int((i + 0.5) * total / limit)) for i in range(min(limit, total))}
        if limit and total
        else None
    )
    ordinal = 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if spec.get("_development_groups") is not None and group not in spec["_development_groups"]:
            ordinal += size
            continue
        if selected is not None and not any(ordinal <= i < ordinal + size for i in selected):
            ordinal += size
            continue
        for batch in parquet.iter_batches(batch_size=32, columns=columns, row_groups=[group]):
            for row in batch.to_pylist():
                ordinal += 1
                if selected is None or ordinal - 1 in selected:
                    yield ordinal, row


def cleaned(row: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str | None]:
    from codexa.data.language_sources import ADAPTERS, cleaned_v3

    if spec.get("adapter") in ADAPTERS:
        return cleaned_v3(row, spec)
    if spec.get("adapter") == "codexa-common-modern-english-1":
        from codexa.data.common_corpus import eligible

        error = eligible(row, spec)
        if error:
            return "", error
    error = source_error(row, spec)
    if error:
        return "", error
    text, errors, _ = clean(source_text(row, spec["name"]), spec["category"])
    return text, errors[0] if errors else admission_error(text, row, spec.get("filter", {}))


def comparison_rows(
    specs: list[dict[str, Any]], cache: Path, maximum_records: int, budget: int
) -> Generator[tuple[dict[str, Any], int, dict[str, Any]], None, None]:
    """Spread the total row budget across payloads without increasing per-source scope."""
    for i, spec in enumerate(specs):
        limit = maximum_records // len(specs) + (i < maximum_records % len(specs))
        sampled = spec | {"_development_limit": limit} if spec.get("adapter") else spec
        rows = source_rows(sampled, cache / spec["sha256"][:16], budget)
        try:
            for ordinal, row in islice(rows, limit):
                yield spec, ordinal, row
        finally:
            rows.close()
        receipt = cache / spec["sha256"][:16] / "ranges.json"
        if receipt.exists():
            content = json.loads(receipt.read_text())
            budget -= content.get("attempted_bytes", content["transferred_bytes"])


def development_charge(root: Path) -> int:
    """Charge attempted requests fully, including interrupted reads, plus staged payloads."""
    cache = root / "cache"
    directories = {
        p.parent
        for pattern in ("**/attempted-ranges.json", "**/ranges.json")
        for p in cache.glob(pattern)
    }
    charge = 0
    for folder in directories:
        attempted, completed = folder / "attempted-ranges.json", folder / "ranges.json"
        if attempted.exists():
            charge += json.loads(attempted.read_text())["attempted_bytes"]
        elif completed.exists():
            receipt = json.loads(completed.read_text())
            charge += receipt.get("attempted_bytes", receipt["transferred_bytes"])
    charge += sum(
        p.stat().st_size
        for p in (root / "raw").iterdir()
        if p.is_file() and not p.name.endswith((".json", ".lock"))
    )
    return charge


def profile(
    root: Path,
    reference: Path,
    tokenizer: Path,
    config: Path,
    maximum_records: int = 10000,
    samples: int = 30,
    profile_name: str = "development",
    only_sources: list[str] | None = None,
) -> dict[str, Any]:
    if not 0 < maximum_records <= 10000 or not 0 < samples <= 30:
        raise ValueError(
            "Development comparison is bounded to 10000 rows and 30 documents per source"
        )
    storage = Storage.load(root)
    storage.check(256 * 1024**2)
    if digest(tokenizer) != TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer identity mismatch")
    tok = CodexaTokenizer.load(tokenizer)
    if Path(profile_name).name != profile_name:
        raise ValueError("Profile name must be a basename")
    out = root / "reports" / profile_name
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    version = yaml.safe_load(config.read_text()).get("version", VERSION)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("Unsupported language data version")
    prior_charge = (
        development_charge(root) if version in {V2_VERSION, V3_VERSION, V4_VERSION} else 0
    )
    network_remaining = GIB - prior_charge
    if network_remaining < 0:
        raise OSError("Cumulative development acquisition ceiling exceeded")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in load_sources(config, reference):
        if version == V3_VERSION and spec.get("repo") and not spec.get("local_path"):
            from codexa.data.v2_acquire import validate_spec

            staged = root / "raw" / validate_spec(spec)
            if staged.exists():
                spec = spec | {"local_path": str(staged)}
        if only_sources is None or spec["name"] in only_sources:
            groups[spec["name"]].append(spec)
    for name, specs in groups.items():
        start = time.monotonic()
        counts: Counter[str] = Counter()
        formats: Counter[str] = Counter()
        census: dict[str, Counter[str]] = {
            key: Counter() for key in ("collection", "license", "date", "language", "open_type")
        }
        retained_selected: list[tuple[int, int, dict[str, Any]]] = []
        inspected: list[dict[str, Any]] = []
        selected: list[tuple[int, int, dict[str, Any]]] = []
        cache = root / "cache" / (profile_name + "-" + name + "-ranges")
        rows = comparison_rows(specs, cache, maximum_records, network_remaining)
        try:
            for spec, ordinal, row in rows:
                counts["scanned"] += 1
                for key, counter in census.items():
                    counter[str(row.get(key, "missing"))] += 1
                text, error = cleaned(row, spec)
                inspected.append(
                    {
                        "source_sha256": spec["sha256"],
                        "ordinal": ordinal,
                        "document_id": hashlib.sha256(
                            canonical(text or str(row.get("text", ""))).encode()
                        ).hexdigest(),
                        "raw_document_id": hashlib.sha256(
                            canonical(str(row.get("text", ""))).encode()
                        ).hexdigest(),
                    }
                )
                counts[error or "filter_accepted"] += 1
                formats[str(row.get("format", "unspecified"))] += 1
                if not error:
                    ids = tok.encode(text)
                    counts["accepted_tokens_with_boundaries"] += len(ids) + 2
                    counts["accepted_text_bytes"] += len(text.encode())
                    if 0 in ids or 1 in ids or tok.decode(ids) != text:
                        raise ValueError("Tokenizer roundtrip mismatch")
                # Sample raw rows before quality selection; preserve failures and complete text.
                raw = row.get("text", "")
                if not isinstance(raw, str):
                    raw = ""
                score = int(
                    hashlib.sha256(
                        f"language-dev-1:{spec['sha256']}:{ordinal}".encode()
                    ).hexdigest(),
                    16,
                )
                item = {
                    "id": f"{name}:{spec['sha256'][:12]}:{ordinal}",
                    "source": name,
                    "ordinal": ordinal,
                    "source_sha256": spec["sha256"],
                    "document_id": hashlib.sha256(canonical(text or raw).encode()).hexdigest(),
                    "raw_text": raw,
                    "text": text,
                    "filter_error": error,
                    "metadata": {
                        k: row[k]
                        for k in (
                            "url",
                            "id",
                            "identifier",
                            "collection",
                            "license",
                            "date",
                            "title",
                            "creator",
                            "curator",
                            "language",
                            "language_type",
                            "open_type",
                            "audience",
                            "format",
                            "seed_data",
                        )
                        if k in row
                    },
                    "review_status": "unreviewed",
                    "coverage": "full document",
                }
                from codexa.data.language_sources import ADAPTERS, clean_source, source_metadata

                if spec.get("adapter") in ADAPTERS:
                    item["metadata"] = source_metadata(row)
                    item["normalization_decisions"] = clean_source(row, spec)[1]
                entry = (-score, ordinal, item)
                if not error:
                    if len(retained_selected) < samples:
                        heapq.heappush(retained_selected, entry)
                    elif score < -retained_selected[0][0]:
                        heapq.heapreplace(retained_selected, entry)
                if len(selected) < samples:
                    heapq.heappush(selected, entry)
                elif score < -selected[0][0]:
                    heapq.heapreplace(selected, entry)
        finally:
            rows.close()
        packet = out / (name + ".jsonl")
        packet.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for _, _, row in sorted(selected))
        )
        retained_packet = out / (name + "-retained.jsonl")
        retained_packet.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for _, _, row in sorted(retained_selected)
            )
        )
        inspected_packet = out / (name + "-inspected.jsonl")
        inspected_packet.write_text("".join(json.dumps(row) + "\n" for row in inspected))
        transfer = sum(
            json.loads(p.read_text())["transferred_bytes"] for p in cache.glob("*/ranges.json")
        )
        attempted = sum(
            json.loads(p.read_text()).get("attempted_bytes", 0)
            for p in cache.glob("*/attempted-ranges.json")
        )
        network_remaining -= attempted or transfer
        report = {
            "sources": specs,
            "counts": dict(counts),
            "formats": dict(formats),
            "metadata_census": {key: dict(value) for key, value in census.items()},
            "retained_sample_file": str(retained_packet),
            "retained_sample_sha256": digest(retained_packet),
            "inspected_identities_file": str(inspected_packet),
            "inspected_identities_sha256": digest(inspected_packet),
            "eligible_yield_is_rights_approved": False,
            "sample_sha256": digest(packet),
            "sample_file": str(packet),
            "seconds": time.monotonic() - start,
            "network_bytes": transfer,
            "development_only": True,
            "sampling_limitation": "Distributed global-row sample for adapter sources; otherwise equal row-budget prefixes. Bottom-hash full-document panels are development only; population representativeness is not established.",
            "rights_status": "pending",
            "quality_status": "unreviewed",
        }
        atomic(out / (name + ".json"), report)
        reports.append(report)
    result = {
        "version": version,
        "config_sha256": digest(config),
        "tokenizer_sha256": TOKENIZER_SHA,
        "maximum_records_per_source": maximum_records,
        "samples_per_source": samples,
        "network_bytes": sum(report["network_bytes"] for report in reports),
        "conservative_acquisition_charge_bytes": GIB - network_remaining,
        "prior_acquisition_charge_bytes": prior_charge,
        "sources": reports,
        "training_allowed": False,
    }
    atomic(out / "comparison.json", result)
    return result


class Reference(Index):
    """Reuse historical/evaluation indexes read-only without copying their storage."""

    def __init__(
        self, reference: Path, historical: Path, *, include_candidate_holdouts: bool = False
    ):
        from collections import OrderedDict

        self.old = sqlite3.connect(f"file:{historical}?mode=ro&immutable=1", uri=True)
        self.db = sqlite3.connect(":memory:", uri=True)
        self.db.execute(
            "ATTACH DATABASE ? AS ref", (f"file:{reference / 'indexes/documents.sqlite'}?mode=ro",)
        )
        scope = "role IN ('history','evaluation')"
        if include_candidate_holdouts:
            scope += " OR (role='new' AND split IN ('val','test'))"
        joined = scope.replace("role", "d.role").replace("split", "d.split")
        self.db.executescript(
            f"CREATE TEMP VIEW docs AS SELECT * FROM ref.docs WHERE {scope};"
            f"CREATE TEMP VIEW bands AS SELECT b.* FROM ref.bands b JOIN ref.docs d ON d.id=b.doc WHERE {joined};"
            f"CREATE TEMP VIEW anchors AS SELECT a.* FROM ref.anchors a JOIN ref.docs d ON d.id=a.doc WHERE {joined};"
        )
        self.shingle_cache = OrderedDict()
        self.shingle_bytes = 0

    def close(self) -> None:
        self.db.close()
        self.old.close()


def reference_holdout_identity(reference: Path) -> str:
    """Bind the previously untrained reference validation/test identities for V4."""
    value = hashlib.sha256()
    with sqlite3.connect(f"file:{reference}/indexes/documents.sqlite?mode=ro", uri=True) as db:
        # Filter sequentially before sorting the small heldout set. ORDER BY hash
        # makes SQLite walk the entire large payload table in random hash order.
        rows = db.execute(
            "SELECT hash,source,split,tokens FROM docs NOT INDEXED WHERE role='new' "
            "AND split IN ('val','test')"
        ).fetchall()
        for ident, source, split, tokens in sorted(rows, key=lambda row: row[0]):
            value.update(
                json.dumps(
                    [bytes(ident).hex(), source, split, tokens], separators=(",", ":")
                ).encode()
            )
            value.update(b"\n")
    return value.hexdigest()


def stage_narrative_validation(index: Index, reference: Path) -> int:
    """Reserve the existing publisher validation file using native duplicate checks."""
    path = reference / "raw" / NARRATIVE_VALIDATION_FILE
    if (
        path.stat().st_size != NARRATIVE_VALIDATION_BYTES
        or digest(path) != NARRATIVE_VALIDATION_SHA
    ):
        raise ValueError("Pinned official narrative validation file changed")
    count = 0
    for ordinal, row in records(path, {}):
        if row.get("error") or not isinstance(row.get("text"), str):
            raise ValueError("Cannot reconstruct an official narrative validation record")
        text = row["text"]
        ident = hashlib.sha256(canonical(text).encode()).digest()
        if index.db.execute("SELECT 1 FROM docs WHERE hash=?", (ident,)).fetchone():
            continue
        _, signature, anchors = features(text)
        index.insert(
            ident,
            "tinystories-gpt4-official-validation",
            "narrative",
            "val",
            text,
            signature,
            anchors,
            {"source_sha256": NARRATIVE_VALIDATION_SHA, "ordinal": ordinal},
            role="evaluation",
        )
        count += 1
    index.db.commit()
    return count


def stage_prior_candidate_holdouts(index: Index, decision: dict[str, Any]) -> int:
    """Reserve earlier measured candidate holdouts when extending a V4 corpus."""
    prior = decision.get("prior_candidate_holdouts", [])
    if prior and decision.get("version") != V4_VERSION:
        raise ValueError("Additional candidate holdouts require explicit version 4 admission")
    count = 0
    for item in prior:
        root, snapshot = Path(item["root"]), Path(item["snapshot"])
        if not snapshot.resolve().is_relative_to((root / "snapshots").resolve()):
            raise ValueError("Prior holdout snapshot is outside its recorded root")
        if digest(snapshot / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Prior candidate holdout manifest changed")
        manifest = verify(snapshot)
        listing = snapshot / "documents.jsonl"
        if digest(listing) != manifest["documents_sha256"]:
            raise ValueError("Prior candidate holdout listing changed")
        expected = {
            row["id"]: row["split"]
            for row in map(json.loads, listing.read_text().splitlines())
            if row["split"] in {"val", "test"}
        }
        found = set()
        with sqlite3.connect(f"file:{root}/indexes/documents.sqlite?mode=ro", uri=True) as db:
            for ident, split, payload in db.execute(
                "SELECT hash,split,payload FROM docs WHERE role='new' AND split IN ('val','test')"
            ):
                import zlib

                if expected.get(bytes(ident).hex()) != split:
                    raise ValueError("Prior candidate holdout index differs from listing")
                text = zlib.decompress(payload).decode()
                if hashlib.sha256(canonical(text).encode()).digest() != bytes(ident):
                    raise ValueError("Prior candidate holdout text changed")
                found.add(bytes(ident).hex())
                if index.db.execute("SELECT 1 FROM docs WHERE hash=?", (ident,)).fetchone():
                    continue
                _, signature, anchors = features(text)
                index.insert(
                    bytes(ident),
                    "prior-candidate-holdout",
                    "narrative",
                    split,
                    text,
                    signature,
                    anchors,
                    {"manifest_sha256": item["manifest_sha256"]},
                    role="evaluation",
                )
                count += 1
        if found != set(expected):
            raise ValueError("Prior candidate holdout index is incomplete")
    index.db.commit()
    return count


def prepared_rows(
    specs: list[dict[str, Any]],
) -> Generator[tuple[dict[str, Any], int, dict[str, Any]], None, None]:
    """Interleave approved payloads so a large first shard cannot consume the source quota."""
    readers = [(spec, source_rows(spec, Path(), 0)) for spec in specs]
    try:
        active = readers[:]
        while active:
            remaining = []
            for spec, iterator in active:
                item = next(iterator, None)
                if item is not None:
                    yield spec, item[0], item[1]
                    remaining.append((spec, iterator))
            active = remaining
    finally:
        for _, iterator in readers:
            iterator.close()


def rights_evidence(spec: dict[str, Any]) -> dict[str, str]:
    evidence = spec.get("rights_evidence")
    if spec.get("rights_status") != "approved" or not isinstance(evidence, dict):
        raise ValueError("Explicit reviewed source rights evidence required")
    path = Path(evidence.get("path", ""))
    if not path.is_file() or digest(path) != evidence.get("sha256"):
        raise ValueError("Source rights evidence missing or changed")
    return {"path": str(path), "sha256": digest(path)}


def source_shares(decision: dict[str, Any]) -> dict[str, float]:
    """Validate frozen per-source proportions independently of payload count."""
    selected = decision["sources"]
    if not selected or len({s["sha256"] for s in selected}) != len(selected):
        raise ValueError("Distinct pinned source payloads required")
    shares: dict[str, float] = {}
    for spec in selected:
        share = spec["target_share"]
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(share)
            or not 0 < share <= 1
        ):
            raise ValueError("Positive finite source shares required")
        if spec["name"] in shares and shares[spec["name"]] != share:
            raise ValueError("Inconsistent per-source shares")
        shares[spec["name"]] = share
    if not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Source shares must sum to one")
    if decision.get("version") == VERSION:
        if len(shares) != 3 or shares != {
            "wikipedia-en": 0.6,
            "tinystories-gpt4": 0.15,
            decision.get("educational_source"): 0.25,
        }:
            raise ValueError("Mixture differs from frozen language plan")
    elif decision.get("version") == V2_VERSION:
        from codexa.data.common_corpus import POLICY, REPO, REVISION

        if decision.get("policy") != POLICY or any(
            spec.get("repo") != REPO
            or spec.get("revision") != REVISION
            or spec.get("adapter") != POLICY
            for spec in selected
        ):
            raise ValueError("Version 2 requires the pinned admitted Common Corpus policy")
    elif decision.get("version") == V3_VERSION:
        from codexa.data.language_sources import POLICY, valid_source

        if (
            decision.get("policy") != POLICY
            or len(shares) != 1
            or any(not valid_source(spec) for spec in selected)
        ):
            raise ValueError("Version 3 requires one explicitly pinned authorized English source")
        names = {
            "codexa-dclm-english-1": "dclm-edu",
            "codexa-pressbooks-english-1": "pressbooks-filtered",
        }
        if any(spec["name"] != names[spec["adapter"]] for spec in selected):
            raise ValueError("Version 3 source name differs from authorized adapter")
    elif decision.get("version") == V4_VERSION:
        from codexa.data.language_sources import (
            TINYSTORIES_RIGHTS_POLICY,
            validate_narrative_rights,
        )

        if (
            decision.get("policy") != TINYSTORIES_RIGHTS_POLICY
            or shares != {"tinystories-gpt4": 1.0}
            or len(selected) != 1
            or selected[0].get("category") != "narrative"
            or not selected[0].get("local_path")
            or selected[0].get("adapter")
        ):
            raise ValueError("Version 4 requires the one preserved GPT-4 narrative source")
        proof = rights_evidence(selected[0])
        if not validate_narrative_rights(selected[0], json.loads(Path(proof["path"]).read_text())):
            raise ValueError("Version 4 requires reviewed creator text rights")
    else:
        raise ValueError("Unsupported language admission version")
    return shares


def admitted_split_targets(decision: dict[str, Any]) -> dict[str, int]:
    """Bind V3 candidate sizes without changing historical defaults or quality gates."""
    if decision.get("version") not in {V3_VERSION, V4_VERSION}:
        if "split_targets" in decision or "split_assignments" in decision:
            raise ValueError("Historical language versions reject split target overrides")
        return dict(SPLIT_TARGETS)
    targets = decision.get("split_targets", SPLIT_TARGETS)
    if (
        not isinstance(targets, dict)
        or set(targets) != {"train", "val", "test"}
        or any(type(value) is not int for value in targets.values())
        or targets["train"] < V3_MIN_TRAIN_TOKENS
        or targets["val"] < V3_MIN_EVALUATION_TOKENS
        or targets["test"] < V3_MIN_EVALUATION_TOKENS
    ):
        raise ValueError("Candidate token targets differ from approved capacity policy")
    if "split_assignments" in decision:
        assignments = decision["split_assignments"]
        if (
            decision.get("split_policy") != "book-hash-balanced-1"
            or {s.get("name") for s in decision.get("sources", [])} != {"pressbooks-filtered"}
            or not isinstance(assignments, dict)
            or not assignments
            or any(
                not isinstance(k, str)
                or not k
                or not isinstance(v, str)
                or v not in {"train", "val", "test"}
                for k, v in assignments.items()
            )
        ):
            raise ValueError("Invalid versioned whole-book split assignments")
    return dict(targets)


def source_split(group: str, decision: dict[str, Any]) -> str:
    """Use a frozen whole-book allocation or the unchanged historical hash split."""
    if "split_assignments" in decision:
        if decision.get("version") != V3_VERSION:
            raise ValueError("Historical language versions reject split assignments")
        split = decision["split_assignments"].get(group)
        if split not in {"train", "val", "test"}:
            raise ValueError("Original book absent from frozen split assignments")
        return str(split)
    bucket = (
        int(hashlib.sha256(("language-split-1:" + group).encode()).hexdigest()[:16], 16) / 2**64
    )
    return "test" if bucket < 1 / 62 else "val" if bucket < 2 / 62 else "train"


def split_assignment_hash(decision: dict[str, Any]) -> str | None:
    assignments = decision.get("split_assignments")
    return (
        hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
        if assignments
        else None
    )


def row_group(row: dict[str, Any], spec: dict[str, Any], identity: str) -> str:
    from codexa.data.language_sources import ADAPTERS
    from codexa.data.language_sources import grouping as language_grouping

    if spec.get("adapter") in ADAPTERS:
        return language_grouping(row, spec)
    if spec.get("adapter") == "codexa-common-modern-english-1":
        from codexa.data.common_corpus import grouping

        return grouping(row)
    return str(row.get("url") or row.get("id") or identity)


def row_rights(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    rights_evidence(spec)
    from codexa.data.language_sources import ADAPTERS
    from codexa.data.language_sources import covered as language_covered

    if spec.get("adapter") in ADAPTERS:
        return language_covered(
            row, spec, json.loads(Path(spec["rights_evidence"]["path"]).read_text())
        )
    if spec.get("adapter") == "codexa-common-modern-english-1":
        from codexa.data.common_corpus import covered

        evidence = json.loads(Path(spec["rights_evidence"]["path"]).read_text())
        return covered(row, spec, evidence)
    return True


def compiled_rights(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Load evidence and validate original proofs once per invocation, never globally."""
    from codexa.data.language_sources import ADAPTERS, compile_checker

    result = {}
    for spec in specs:
        if spec.get("adapter") in ADAPTERS:
            rights_evidence(spec)
            evidence = json.loads(Path(spec["rights_evidence"]["path"]).read_text())
            result[spec["sha256"]] = compile_checker(spec, evidence)
    return result


def prepare(
    root: Path, reference: Path, tokenizer: Path, fixtures: Path, admission: Path
) -> dict[str, Any]:
    """Prepare a new bounded corpus only from full verified locally staged sources."""
    storage = Storage.load(root)
    storage.check(512 * 1024**2)
    decision = json.loads(admission.read_text())
    if (
        decision.get("version") not in SUPPORTED_VERSIONS
        or decision.get("status") != "admitted"
        or not decision.get("frozen_development_sha256")
    ):
        raise ValueError("Frozen development and source rights decisions are required")
    if (
        digest(
            root
            / "reports"
            / decision.get("development_profile", "development")
            / "comparison.json"
        )
        != decision["frozen_development_sha256"]
    ):
        raise ValueError("Development comparison changed")
    if digest(tokenizer) != TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer identity mismatch")
    selected = decision["sources"]
    source_shares(decision)
    split_targets = admitted_split_targets(decision)
    if any(
        s.get("rights_status") != "approved"
        or not s.get("rights_evidence")
        or not s.get("local_path")
        for s in selected
    ):
        raise ValueError(
            "Each source needs explicit rights evidence and a verified complete local payload"
        )
    for spec in selected:
        rights_evidence(spec)
    rights_checkers = compiled_rights(selected)
    from codexa.evaluation.language import protected_texts

    extra = Protection.build([(text, True) for text in protected_texts(fixtures)])
    protected = Protection.load(reference / "indexes/protected-v2.npz")
    questions = QuestionProtection(reference / "benchmarks/v3")
    historical = Path(
        json.loads((reference / "reports/historical-index.json").read_text())["input"]
    )
    old = Reference(
        reference, historical, include_candidate_holdouts=decision["version"] == V4_VERSION
    )
    new = Index(storage, historical)
    if decision["version"] == V4_VERSION:
        stage_prior_candidate_holdouts(new, decision)
        stage_narrative_validation(new, reference)
    tok = CodexaTokenizer.load(tokenizer)
    snapshot_name = decision.get("snapshot_name", "language-60m")
    if Path(snapshot_name).name != snapshot_name:
        raise ValueError("Snapshot name must be a basename")
    output = root / "snapshots" / snapshot_name
    output.mkdir(parents=True, exist_ok=False)
    totals: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    docs = []
    streams = {s: (output / (s + "-00000.bin")).open("xb") for s in split_targets}
    offsets: dict[str, list[int]] = {s: [] for s in streams}
    dev_ids: set[str] = set()
    for report in decision.get("development_packets", []):
        p = Path(report["path"])
        if digest(p) != report["sha256"]:
            raise ValueError("Development packet changed")
        dev_ids.update(json.loads(line)["document_id"] for line in p.read_text().splitlines())
    try:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in selected:
            grouped[spec["name"]].append(spec)
        for specs in grouped.values():
            spec = specs[0]
            quotas = {
                split: int(total * spec["target_share"]) for split, total in split_targets.items()
            }
            retained: Counter[str] = Counter()
            for spec, ordinal, row in prepared_rows(specs):
                text, error = cleaned(row, spec)
                if error:
                    rejected[spec["name"] + ":" + error] += 1
                    continue
                checker = rights_checkers.get(spec["sha256"])
                grant = checker(row) if checker is not None else None
                if (checker is not None and grant is None) or (
                    checker is None and not row_rights(row, spec)
                ):
                    rejected[spec["name"] + ":document_rights_unverified"] += 1
                    continue
                ident = hashlib.sha256(canonical(text).encode()).digest()
                group = row_group(row, spec, ident.hex())
                split = source_split(group, decision)
                if retained[split] >= quotas[split]:
                    continue
                if ident.hex() in dev_ids:
                    rejected[spec["name"] + ":development_document"] += 1
                    continue
                shingles, signature, anchors = features(text)
                reason = (
                    "protected_span"
                    if protected.matches(text) or extra.matches(text) or questions.matches(text)
                    else old.duplicate(ident, text, shingles, signature, anchors)
                    or new.duplicate(ident, text, shingles, signature, anchors)
                )
                if reason:
                    rejected[spec["name"] + ":" + reason] += 1
                    continue
                ids = [2, *tok.encode(text), 3]
                if 0 in ids or 1 in ids or tok.decode(ids[1:-1]) != text:
                    raise ValueError("Frozen tokenizer roundtrip failure")
                metadata = {
                    "source_sha256": spec["sha256"],
                    "revision": spec["revision"],
                    "license": spec["license"],
                    "ordinal": ordinal,
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "id": row.get("id"),
                    "rights_evidence": spec["rights_evidence"],
                }
                if spec.get("adapter") == "codexa-common-modern-english-1":
                    from codexa.data.common_corpus import metadata as common_metadata

                    metadata["common_corpus"] = common_metadata(row)
                    metadata["source_group"] = group
                    metadata["source_license"] = spec["license"]
                    metadata["license"] = row["license"]
                from codexa.data.language_sources import (
                    ADAPTERS,
                    clean_source,
                    source_metadata,
                )

                if spec.get("adapter") in ADAPTERS:
                    assert grant is not None
                    metadata["language_source"] = source_metadata(row)
                    metadata["normalization_decisions"] = clean_source(row, spec)[1]
                    metadata["source_group"] = group
                    metadata["source_license"] = spec["license"]
                    metadata["license"] = grant["license"]
                    metadata["rights_grant"] = grant
                new.insert(
                    ident,
                    spec["name"],
                    spec["category"],
                    split,
                    text,
                    signature,
                    anchors,
                    metadata,
                    len(ids),
                )
                offsets[split].append(totals[split])
                streams[split].write(np.asarray(ids, dtype="<u2").tobytes())
                totals[split] += len(ids)
                retained[split] += len(ids)
                if split == "train":
                    source_counts[spec["name"]] += len(ids)
                docs.append(
                    {"id": ident.hex(), "source": spec["name"], "split": split, "tokens": len(ids)}
                )
                if len(docs) % 100 == 0:
                    new.db.commit()
                    storage.check(256 * 1024**2)
                if all(retained[s] >= quotas[s] for s in quotas):
                    break
            if any(retained[s] < quotas[s] for s in quotas):
                raise ValueError(
                    f"Source yield below frozen quota: {spec['name']} {dict(retained)}"
                )
        new.db.commit()
    finally:
        for stream in streams.values():
            stream.close()
        new.close()
        old.close()
    entries: dict[str, list[dict[str, Any]]] = {}
    for split in split_targets:
        path = output / (split + "-00000.bin")
        ix = path.with_suffix(".bin.idx")
        ix.write_bytes(np.asarray(offsets[split], dtype="<u8").tobytes())
        entries[split] = [
            {
                "file": path.name,
                "index": ix.name,
                "tokens": totals[split],
                "documents": len(offsets[split]),
                "sha256": digest(path),
                "index_sha256": digest(ix),
            }
        ]
    listing = output / "documents.jsonl"
    listing.write_text("".join(json.dumps(row) + "\n" for row in docs))
    parent_data = historical.parent.parent / "tokens/manifest.json"
    manifest = {
        "version": 2,
        "stage": "tokenized",
        "dtype": "<u2",
        "vocab_size": 16384,
        "tokenizer_hash": TOKENIZER_SHA,
        "dataset_name": "Codexa-v3-language-narrative"
        if decision["version"] == V4_VERSION
        else "Codexa-v3-language-60m",
        "boundary": "BOS document EOS; causal attention across boundaries",
        "identity": {
            "language_version": decision["version"],
            **(
                {
                    "split_policy": decision["split_policy"],
                    "split_assignments_sha256": split_assignment_hash(decision),
                }
                if "split_assignments" in decision
                else {}
            ),
            **(
                {"source_adapter_sha256": digest(Path(__file__).with_name("language_sources.py"))}
                if decision["version"] in {V3_VERSION, V4_VERSION}
                else {}
            ),
            **(
                {
                    "reference_heldouts_sha256": reference_holdout_identity(reference),
                    "official_validation_sha256": NARRATIVE_VALIDATION_SHA,
                }
                if decision["version"] == V4_VERSION
                else {}
            ),
            **(
                {"split_targets": split_targets}
                if decision["version"] in {V2_VERSION, V3_VERSION, V4_VERSION}
                else {}
            ),
            "admission": digest(admission),
            "language_evaluation": digest(fixtures / "manifest.json"),
            "historical": digest(historical.parent / "manifest.json"),
            "excluded_dataset_fingerprints": [digest(parent_data)],
        },
        "splits": entries,
        "source_tokens": dict(source_counts),
        "documents_sha256": digest(listing),
        "preparation_complete": True,
    }
    atomic(output / "manifest.json", manifest)
    verify(output)
    audit = {
        "manifest_sha256": digest(output / "manifest.json"),
        "all_accepted_documents_screened_against_final_policy": True,
        "status": "pending_independent_verification",
        "rejections": dict(rejected),
        "split_tokens": dict(totals),
        "training_allowed": False,
    }
    atomic(output / "audit.json", audit)
    return audit


def verify_snapshot(
    root: Path, snapshot: Path, reference: Path, tokenizer: Path, fixtures: Path, admission: Path
) -> dict[str, Any]:
    """Independent full reconstruction pass; does not trust preparation audit booleans."""
    import zlib

    from codexa.data.v2_finalize import token_documents
    from codexa.evaluation.language import protected_texts
    from codexa.repository import source_identity

    storage = Storage.load(root)
    storage.check(512 * 1024**2)
    output = snapshot / "independent-verification.json"
    if output.exists():
        raise FileExistsError(
            "Preserve prior verification; use a new candidate for changed evidence"
        )
    manifest = verify(snapshot)
    if manifest["identity"].get("admission") != digest(admission) or manifest["identity"].get(
        "language_evaluation"
    ) != digest(fixtures / "manifest.json"):
        raise ValueError("Candidate admission or benchmark identity changed")
    if digest(tokenizer) != TOKENIZER_SHA or manifest["tokenizer_hash"] != TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer identity mismatch")
    tok = CodexaTokenizer.load(tokenizer)
    decision = json.loads(admission.read_text())
    if decision.get("status") != "admitted" or decision.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError("Candidate source admission is not approved")
    shares = source_shares(decision)
    if decision["version"] in {V3_VERSION, V4_VERSION} and manifest["identity"].get(
        "source_adapter_sha256"
    ) != digest(Path(__file__).with_name("language_sources.py")):
        raise ValueError("Language adapter implementation differs from frozen candidate")
    split_targets = admitted_split_targets(decision)
    if (
        decision["version"] in {V2_VERSION, V3_VERSION, V4_VERSION}
        and manifest["identity"].get("split_targets") != split_targets
    ):
        raise ValueError("Candidate token targets differ from frozen preparation policy")
    if decision["version"] == V4_VERSION and manifest["identity"].get(
        "reference_heldouts_sha256"
    ) != reference_holdout_identity(reference):
        raise ValueError("Previously protected reference holdouts changed")
    if (
        decision["version"] == V4_VERSION
        and manifest["identity"].get("official_validation_sha256") != NARRATIVE_VALIDATION_SHA
    ):
        raise ValueError("Official narrative validation identity changed")
    if manifest["identity"].get("split_assignments_sha256") != split_assignment_hash(decision):
        raise ValueError("Whole-book split assignment identity changed")
    if (
        "split_assignments" in decision
        and manifest["identity"].get("split_policy") != decision["split_policy"]
    ):
        raise ValueError("Whole-book split policy changed")
    if manifest["identity"].get("language_version") != decision["version"]:
        raise ValueError("Candidate language version differs from admission")
    specs = {spec["sha256"]: spec for spec in decision["sources"]}
    rights = [rights_evidence(spec) for spec in specs.values()]
    rights_checkers = compiled_rights(list(specs.values()))
    historical = Path(
        json.loads((reference / "reports/historical-index.json").read_text())["input"]
    )
    parent_fingerprint = digest(historical.parent.parent / "tokens/manifest.json")
    if parent_fingerprint not in manifest["identity"].get("excluded_dataset_fingerprints", []):
        raise ValueError("Missing parent-dataset exclusion identity")
    protected = Protection.load(reference / "indexes/protected-v2.npz")
    questions = QuestionProtection(reference / "benchmarks/v3")
    extra = Protection.build([(text, True) for text in protected_texts(fixtures)])
    rows = [json.loads(line) for line in (snapshot / "documents.jsonl").read_text().splitlines()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate document identity across candidate splits")
    db = sqlite3.connect(f"file:{root / 'indexes/documents.sqlite'}?mode=ro", uri=True)
    by_id = {}
    try:
        for ident, source, split, payload, metadata, count in db.execute(
            "SELECT hash,source,split,payload,metadata,tokens FROM docs WHERE role='new' ORDER BY id"
        ):
            by_id[ident.hex()] = (
                source,
                split,
                zlib.decompress(payload).decode(),
                json.loads(metadata),
                count,
            )
    finally:
        db.close()
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Candidate payload index does not exactly match snapshot")
    expected_origins: dict[str, dict[int, str]] = defaultdict(dict)
    split_tokens: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    source_split_tokens: Counter[tuple[str, str]] = Counter()
    source_largest_document: Counter[tuple[str, str]] = Counter()
    previous = Reference(
        reference, historical, include_candidate_holdouts=decision["version"] == V4_VERSION
    )
    verification_root = root / "tmp" / ("independent-" + str(time.time_ns()))
    scratch = Storage(verification_root, maximum=2 * GIB, raw_maximum=0)
    scratch.initialize()
    independent = Index(scratch, historical)
    if decision["version"] == V4_VERSION:
        stage_prior_candidate_holdouts(independent, decision)
        stage_narrative_validation(independent, reference)
    streams = {
        split: token_documents(snapshot, entries) for split, entries in manifest["splits"].items()
    }
    try:
        for document_number, row in enumerate(rows, 1):
            ident = row["id"]
            source, split, text, metadata, count = by_id[ident]
            if row["source"] != source or row["split"] != split or row["tokens"] != count:
                raise ValueError("Document/source/split identity mismatch")
            if hashlib.sha256(canonical(text).encode()).hexdigest() != ident:
                raise ValueError("Candidate text identity mismatch")
            source_sha = metadata["source_sha256"]
            spec = specs[source_sha]
            if (
                spec["name"] != source
                or metadata["revision"] != spec["revision"]
                or metadata.get("source_license", metadata["license"]) != spec["license"]
            ):
                raise ValueError("Source provenance mismatch")
            from codexa.data.language_sources import ADAPTERS

            if spec.get("adapter") in ADAPTERS:
                origin = metadata.get("language_source", {})
                grant = rights_checkers[spec["sha256"]](origin | {"text": text})
                if (
                    grant is None
                    or metadata.get("rights_grant") != grant
                    or metadata["license"] != grant["license"]
                ):
                    raise ValueError("Original document rights or granted licence changed")
                group = row_group(origin, spec, ident)
                if not group or metadata.get("source_group") != group:
                    raise ValueError("Language originating document group changed")
            elif spec.get("adapter") == "codexa-common-modern-english-1":
                origin = metadata.get("common_corpus", {})
                if not row_rights(origin | {"text": text}, spec) or metadata[
                    "license"
                ] != origin.get("license"):
                    raise ValueError(
                        "Document rights or original licence differs from reviewed coverage"
                    )
                group = row_group(origin, spec, ident)
                if metadata.get("source_group") != group:
                    raise ValueError("Common Corpus originating document group changed")
            else:
                group = str(metadata.get("url") or metadata.get("id") or ident)
            expected_split = source_split(group, decision)
            if split != expected_split:
                raise ValueError("Document split differs from frozen grouped split policy")
            encoded = np.asarray([2, *tok.encode(text), 3], dtype="<u2")
            tokens = next(streams[split], None)
            if (
                tokens is None
                or len(encoded) != count
                or not np.array_equal(tokens, encoded)
                or np.any(np.isin(encoded, [0, 1]))
            ):
                raise ValueError("Complete document token reconstruction failed")
            if protected.matches(text) or extra.matches(text) or questions.matches(text):
                raise ValueError("Candidate overlaps protected evaluation text")
            shingles, signature, anchors = features(text)
            if previous.duplicate(
                bytes.fromhex(ident), text, shingles, signature, anchors
            ) or independent.duplicate(bytes.fromhex(ident), text, shingles, signature, anchors):
                raise ValueError("Historical exposure or cross-split duplicate detected")
            independent.insert(
                bytes.fromhex(ident),
                source,
                "verification",
                split,
                text,
                signature,
                anchors,
                {},
                count,
            )
            ordinal = metadata["ordinal"]
            if ordinal in expected_origins[source_sha]:
                raise ValueError("Duplicate source origin")
            expected_origins[source_sha][ordinal] = ident
            split_tokens[split] += count
            source_split_tokens[source, split] += count
            source_largest_document[source, split] = max(
                source_largest_document[source, split], count
            )
            if split == "train":
                source_tokens[source] += count
            if document_number % 100 == 0:
                independent.db.commit()
                scratch.check(64 * 1024**2)
                storage.check(64 * 1024**2)
        if any(next(stream, None) is not None for stream in streams.values()):
            raise ValueError("Unaccounted tokenized document")
        for sha, origins in expected_origins.items():
            spec = specs[sha]
            iterator = source_rows(spec, Path(), 0)
            try:
                remaining = dict(origins)
                for ordinal, original in iterator:
                    if ordinal in remaining:
                        text, error = cleaned(original, spec)
                        ident = remaining.pop(ordinal)
                        if error or by_id[ident][2] != text:
                            raise ValueError(
                                "Original source reconstruction or current filtering failed"
                            )
                        metadata = by_id[ident][3]
                        from codexa.data.language_sources import (
                            ADAPTERS,
                            clean_source,
                            source_metadata,
                        )

                        if spec.get("adapter") in ADAPTERS and (
                            source_metadata(original) != metadata.get("language_source")
                            or rights_checkers[spec["sha256"]](original) is None
                            or clean_source(original, spec)[1]
                            != metadata.get("normalization_decisions")
                        ):
                            raise ValueError(
                                "Original language metadata, score, normalization or rights changed"
                            )
                        if spec.get("adapter") == "codexa-common-modern-english-1":
                            from codexa.data.common_corpus import metadata as common_metadata

                            if common_metadata(original) != metadata.get(
                                "common_corpus"
                            ) or not row_rights(original, spec):
                                raise ValueError(
                                    "Original Common Corpus metadata or rights coverage changed"
                                )
                        if any(
                            original.get(key) != metadata.get(key) for key in ("url", "id", "title")
                        ):
                            raise ValueError("Original source grouping or title provenance changed")
                    if not remaining:
                        break
                if remaining:
                    raise ValueError("Candidate origin absent from pinned source")
            finally:
                iterator.close()
        for split, target in split_targets.items():
            if split_tokens[split] < target:
                raise ValueError("Candidate below frozen split token requirement")
        for source, share in shares.items():
            for split, target in split_targets.items():
                quota = int(target * share)
                actual = source_split_tokens[source, split]
                maximum_overshoot = max(0, source_largest_document[source, split] - 1)
                if not quota <= actual <= quota + maximum_overshoot:
                    raise ValueError("Frozen source quota or full-document overshoot violated")
        if dict(source_tokens) != manifest["source_tokens"]:
            raise ValueError("Source token totals changed")
    finally:
        independent.close()
        previous.close()
    report = {
        "status": "passed",
        "manifest_sha256": digest(snapshot / "manifest.json"),
        "tokenizer_sha256": TOKENIZER_SHA,
        "documents_reconstructed": len(rows),
        "all_tokens_reencoded": True,
        "protection_verified": True,
        "split_dedup_verified": True,
        "prior_exposure_verified": True,
        "rights_verified": True,
        "rights_evidence": rights,
        "split_tokens": dict(split_tokens),
        "source_tokens": dict(source_tokens),
        "source": source_identity(),
        "verifier_sha256": digest(Path(__file__)),
        "limitations": "Rights check binds explicit reviewed evidence; approximate duplicate/protection checks do not prove semantic novelty or universal rights clearance.",
        "optimizer_updates": 0,
    }
    atomic(output, report)
    return report


def acquisition_admission(root: Path, path: Path | None, requested: dict[str, Any]) -> None:
    """Require frozen development and reviewed rights before exceeding development scope."""
    if path is None:
        raise ValueError(
            "Production acquisition requires a frozen development/source-rights admission"
        )
    decision = json.loads(path.read_text())
    if (
        decision.get("version") not in {V2_VERSION, V3_VERSION}
        or decision.get("status") != "admitted"
    ):
        raise ValueError("Production acquisition requires an admitted version-2 decision")
    profile_name = decision.get("development_profile", "development")
    if Path(profile_name).name != profile_name:
        raise ValueError("Development profile must be a basename")
    comparison = root / "reports" / profile_name / "comparison.json"
    if not comparison.is_file() or digest(comparison) != decision.get("frozen_development_sha256"):
        raise ValueError("Production acquisition development comparison missing or changed")
    source_shares(decision)
    fields = (
        "name",
        "repo",
        "revision",
        "path",
        "sha256",
        "bytes",
        "license",
        "category",
        "adapter",
        "target_share",
        "filter",
        "collections",
        "min_edu_score",
    )
    admitted = next(
        (
            spec
            for spec in decision["sources"]
            if all(spec.get(key) == requested.get(key) for key in fields)
        ),
        None,
    )
    if admitted is None:
        raise ValueError("Requested source or filter differs from frozen acquisition admission")
    rights_evidence(admitted)
    evidence = json.loads(Path(admitted["rights_evidence"]["path"]).read_text())
    from codexa.data.common_corpus import reviewed_grants
    from codexa.data.language_sources import ADAPTERS
    from codexa.data.language_sources import reviewed_grants as language_grants

    grants = (
        language_grants(admitted, evidence)
        if admitted.get("adapter") in ADAPTERS
        else reviewed_grants(admitted, evidence)
    )
    if not grants:
        raise ValueError(
            "Production acquisition requires reviewed original-document rights coverage"
        )


def acquire_next(
    root: Path, reference: Path, config: Path, admission: Path | None = None
) -> dict[str, Any]:
    """Acquire at most one explicitly configured pinned shard per invocation."""
    from codexa.data.v2_acquire import acquire, validate_spec

    policy = yaml.safe_load(config.read_text())
    if policy.get("version") not in {V2_VERSION, V3_VERSION}:
        raise ValueError("Incremental language acquisition requires an authorized language version")
    storage = Storage.load(root)
    if storage.raw_maximum > 16 * GIB:
        raise ValueError("Acquisition policy exceeds authorized 16 GiB ceiling")
    specs = load_sources(config, reference)
    for spec in specs:
        name = validate_spec(spec)
        path = root / "raw" / name
        if path.exists():
            if (
                path.is_symlink()
                or path.stat().st_size != spec["bytes"]
                or digest(path) != spec["sha256"]
            ):
                raise ValueError("Existing acquired payload failed integrity")
            continue
        partial = path.with_name(path.name + ".partial")
        remaining = spec["bytes"] - (partial.stat().st_size if partial.exists() else 0)
        local_charge = development_charge(root) + remaining
        charge = local_charge
        if policy["version"] == V3_VERSION:
            charge += V3_PRIOR_ACQUISITION_BYTES
        if charge > 16 * GIB:
            raise OSError("Cumulative development/acquisition ceiling exceeded")
        if local_charge > GIB:
            acquisition_admission(root, admission, spec)
        path = acquire(storage, spec)
        report = {
            "version": policy["version"],
            "acquired": str(path),
            "specification": spec | {"local_path": str(path)},
            "training_allowed": False,
        }
        return report
    return {
        "version": policy["version"],
        "all_configured_payloads_acquired": True,
        "training_allowed": False,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("initialize", "profile", "prepare", "verify", "acquire"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--version", choices=sorted(SUPPORTED_VERSIONS), default=VERSION)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--sources", type=Path, default=Path("configs/data/language_sources.yaml"))
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--admission", type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--maximum-records", type=int, default=10000)
    p.add_argument("--profile-name", default="development")
    p.add_argument("--only-source", action="append")
    args = p.parse_args(argv)
    if args.action == "initialize":
        if args.root.exists():
            raise FileExistsError(args.root)
        maximum, raw_maximum = (
            (24 * GIB, 16 * GIB)
            if args.version in {V2_VERSION, V3_VERSION, V4_VERSION}
            else (12 * GIB, 4 * GIB)
        )
        Storage(args.root, maximum=maximum, raw_maximum=raw_maximum).initialize()
        print(json.dumps({"root": str(args.root), "training_allowed": False}))
    elif args.action == "acquire":
        print(
            json.dumps(
                acquire_next(args.root, args.reference_root, args.sources, args.admission), indent=2
            )
        )
    elif args.action == "profile":
        print(
            json.dumps(
                profile(
                    args.root,
                    args.reference_root,
                    args.tokenizer,
                    args.sources,
                    args.maximum_records,
                    profile_name=args.profile_name,
                    only_sources=args.only_source,
                ),
                indent=2,
            )
        )
    else:
        if args.fixtures is None or args.admission is None:
            p.error("--fixtures and --admission required")
        if args.action == "verify":
            if args.snapshot is None:
                p.error("--snapshot required")
            result = verify_snapshot(
                args.root,
                args.snapshot,
                args.reference_root,
                args.tokenizer,
                args.fixtures,
                args.admission,
            )
        else:
            result = prepare(
                args.root, args.reference_root, args.tokenizer, args.fixtures, args.admission
            )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
