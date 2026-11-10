"""Explicit, byte-bounded acquisition of pinned, reviewed Parquet snapshots."""

import hashlib
import heapq
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .production import atomic, digest


def sample(config: dict, output: str, max_bytes: int) -> Path:
    import pyarrow.parquet as pq
    import yaml

    if max_bytes <= 0:
        raise ValueError("Positive download budget required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    downloaded = 0
    sources = []
    receipts = []
    for source in config["sources"]:
        for key in ("name", "url", "sha256", "snapshot", "license", "attribution", "text_bytes"):
            if not source.get(key):
                raise ValueError("Missing sampling provenance: " + key)
        path = root / (source["name"] + ".parquet")
        with urllib.request.urlopen(source["url"], timeout=60) as response, path.open("xb") as f:
            if int(response.headers.get("Content-Length", 0)) > max_bytes - downloaded:
                raise ValueError("Snapshot exceeds remaining download budget")
            while chunk := response.read(min(1048576, max_bytes - downloaded + 1)):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError("Download budget exceeded; partial snapshot retained")
                f.write(chunk)
        if digest(path) != source["sha256"]:
            raise ValueError("Snapshot checksum mismatch")
        # Keep lowest seeded document hashes under a text-byte budget, not a file prefix.
        selected: list[tuple[int, int, int]] = []
        used = ordinal = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32, columns=["text", "url"]):
            for row in batch.to_pylist():
                text = row["text"]
                size = len(text.encode())
                score = int.from_bytes(hashlib.sha256(("42:" + text).encode()).digest(), "big")
                if 32 <= size <= 1048576:
                    heapq.heappush(selected, (-score, ordinal, size))
                    used += size
                    while used > source["text_bytes"]:
                        used -= heapq.heappop(selected)[2]
                ordinal += 1
        ordinals = {item[1] for item in selected}
        target = root / (source["name"] + ".jsonl")
        ordinal = 0
        with target.open("x") as f:
            for batch in parquet.iter_batches(batch_size=32, columns=["text", "url"]):
                for row in batch.to_pylist():
                    if ordinal in ordinals:
                        url = urlsplit(row["url"] or "")
                        group = url.netloc.lower() + url.path.rstrip("/")
                        if not group:
                            group = hashlib.sha256(row["text"].encode()).hexdigest()
                        f.write(json.dumps({**row, "group_id": group}, ensure_ascii=False) + "\n")
                    ordinal += 1
        receipts.append(
            {
                **source,
                "downloaded_bytes": path.stat().st_size,
                "selected_documents": len(selected),
                "selected_text_bytes": used,
                "jsonl_sha256": digest(target),
            }
        )
        sources.append(
            {k: source[k] for k in ("name", "snapshot", "license", "attribution")}
            | {"paths": [str(target)], "format": "jsonl"}
        )
        print(json.dumps(receipts[-1]), flush=True)
    atomic(
        root / "source.json",
        {
            "downloaded_bytes": downloaded,
            "budget": max_bytes,
            "sources": receipts,
            "selection": "seed-42 bottom hash by text",
        },
    )
    (root / "config.yaml").write_text(yaml.safe_dump({"sources": sources}))
    return root / "config.yaml"
