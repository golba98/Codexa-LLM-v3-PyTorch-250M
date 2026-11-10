"""Build grade-band reference indexes without copying canonical mathematics records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from codexa.data.math_curriculum_senior_v2 import json_bytes, jsonl_bytes, write_idempotent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(grade: int) -> str:
    if not 1 <= grade <= 12:
        raise ValueError("invalid school grade")
    return (
        "A:1-3" if grade <= 3 else "B:4-7" if grade <= 7 else "C:8-9" if grade <= 9 else "D:10-12"
    )


def index(source: Path, synthetic: Path) -> list[dict[str, Any]]:
    data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"source_record_ids": [], "synthetic_record_ids": []}
    )
    for path, field in ((source, "source_record_ids"), (synthetic, "synthetic_record_ids")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            grades = row.get("grades") if field == "source_record_ids" else [row.get("grade")]
            targets = {band(g) for g in grades or [] if type(g) is int and 1 <= g <= 12}
            if not targets:
                targets = {"unmapped-grade"}
            for name in sorted(targets):
                data[name][field].append(row["record_id"])
    return [
        {
            "collection": name,
            **data[name],
            "status": "references only; rights, mapping, overlap and admission remain independent",
            "limitations": "source identities may overlap across parent/child units or duplicated sources",
        }
        for name in ("A:1-3", "B:4-7", "C:8-9", "D:10-12", "unmapped-grade")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.candidate / "records/source-canonical.jsonl"
    synthetic = args.candidate / "records/synthetic-candidates.jsonl"
    items = index(source, synthetic)
    payload = jsonl_bytes(items)
    write_idempotent(args.output / "grade-band-index.jsonl", payload)
    manifest = {
        "input_sha256": {str(source): digest(source), str(synthetic): digest(synthetic)},
        "index_sha256": hashlib.sha256(payload).hexdigest(),
        "counts": {
            item["collection"]: {
                "source_references": len(item["source_record_ids"]),
                "synthetic_references": len(item["synthetic_record_ids"]),
            }
            for item in items
        },
        "status": "candidate references only; no corpus admission",
    }
    write_idempotent(args.output / "manifest.json", json_bytes(manifest))
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
