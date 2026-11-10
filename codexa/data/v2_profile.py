"""Bounded source inspection before admission. Reports never imply human ratings."""

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.v2 import records
from codexa.data.v2_policy import admission_error, admission_text
from codexa.data.v2_quality import CONTACT, VERSION, clean, source_error, source_text
from codexa.data.v2_storage import Storage


def profile(
    storage: Storage,
    spec: dict[str, Any],
    maximum_records: int = 10000,
    approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = storage.root / "raw" / (spec["name"] + "--" + spec["path"].replace("/", "--"))
    if digest(path) != spec["sha256"]:
        raise ValueError("Unverified source")
    out = storage.root / "reports" / "profiles"
    out.mkdir(exist_ok=True)
    key = spec["name"] + "-" + spec["sha256"][:12] + "-" + VERSION
    if approval:
        key += "-" + hashlib.sha256(json.dumps(approval, sort_keys=True).encode()).hexdigest()[:12]
        key += "-p" + digest(Path(__file__).with_name("v2_policy.py"))[:8]
    target = out / (key + ".jsonl")
    if target.exists():
        return json.loads((out / (key + ".json")).read_text())
    counts: Counter[str] = Counter()
    samples: list[tuple[int, int, dict[str, Any]]] = []
    for ordinal, row in records(path, spec):
        if ordinal > maximum_records:
            break
        reason = source_error(row, spec)
        if reason:
            counts[reason] += 1
            continue
        cleaned, errors, stats = clean(
            admission_text(source_text(row, spec["name"]), approval or {}), spec["category"]
        )
        if not errors and approval:
            error = admission_error(cleaned, row, approval)
            if error:
                errors = [error]
        counts["inspected"] += 1
        counts.update(errors or ["cleaned_candidate"])
        if errors:
            continue
        before = "\n".join(
            "[CONTACT LINE REDACTED]" if CONTACT.search(line) else line
            for line in row["text"].splitlines()
        )
        item = {
            "source": spec["name"],
            "ordinal": ordinal,
            "original_characters": len(row["text"]),
            "before": before[:3000],
            "after": cleaned[:6000],
            "end": cleaned[-800:],
            "metadata": {
                k: row[k] for k in ("url", "title", "audience", "format", "seed_data") if k in row
            },
            "stats": stats,
            "review_status": "unreviewed",
        }
        score = int(hashlib.sha256(("42:" + cleaned).encode()).hexdigest(), 16)
        heap_item = (-score, ordinal, item)
        if len(samples) < 100:
            heapq.heappush(samples, heap_item)
        elif score < -samples[0][0]:
            heapq.heapreplace(samples, heap_item)
    selected = [item for _, _, item in sorted(samples, reverse=True)]
    target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected))
    report = {
        "cleaner": VERSION,
        "source": spec["name"],
        "source_sha256": spec["sha256"],
        "source_revision": spec["revision"],
        "admission_policy": approval,
        "admission_implementation_sha256": digest(Path(__file__).with_name("v2_policy.py")),
        "counts": dict(counts),
        "samples": len(selected),
        "selection": "seed-42 bottom-hash among first 10000 eligible file records",
        "sample_file": str(target),
        "admission": "pending review; filter survival is not linguistic quality",
    }
    atomic(out / (key + ".json"), report)
    return report
