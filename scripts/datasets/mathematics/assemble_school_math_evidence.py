"""Assemble worker provenance and reviews without changing the worker evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "raw"
    sources: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for folder in sorted(raw.glob("grades-*")):
        for path in sorted(folder.glob("*.jsonl")):
            if path.name == "sources.jsonl":
                for row in read(path):
                    row["source_file_sha256"] = row.get("source_file_sha256") or row.get("sha256")
                    row["eligibility"] = (
                        "unresolved"
                        if str(row.get("source_id", "")).startswith("im-")
                        else "eligible"
                        if str(row.get("source_id", "")).startswith("siyavula-")
                        else row.get("eligibility", "unresolved")
                    )
                    sources.append(row)
            elif path.name in (
                "inspection-registry.jsonl",
                "inspections.jsonl",
                "inspection.jsonl",
            ):
                inspections.extend(read(path))
            elif path.name.startswith("review-"):
                reviews.extend(read(path))
    page = raw / "grades-10-12/grade10-real-numbers.html"
    if page.exists():
        digest = hashlib.sha256(page.read_bytes()).hexdigest()
        url = "https://www.siyavula.com/read/za/mathematics/grade-10/algebraic-expressions/01-algebraic-expressions-01"
        sources.append(
            {
                "source_id": "siyavula-za-maths-grade-10-online",
                "publisher": "Siyavula Education",
                "source_url": url,
                "title": "Mathematics Grade 10",
                "edition": "Online CAPS edition, undated",
                "licence": "CC-BY-3.0",
                "licence_statement_url": url,
                "attribution": "Siyavula Education; From siyavula.com; CC BY 3.0; indicate changes",
                "restrictions": "Textbook content only; third-party materials and trademarks excluded; no training-specific prohibition located in site terms.",
                "access_date": "2026-11-10",
                "source_file_sha256": digest,
                "raw_file": str(page),
                "eligibility": "eligible",
            }
        )
        inspections.append(
            {
                "source_url": url,
                "source_file_sha256": digest,
                "location": "1.2 The real number system",
                "reason": "development extraction",
                "raw_file": str(page),
            }
        )
    for filename, url in (
        ("grade10-index.html", "https://www.siyavula.com/read/za/mathematics/grade-10"),
        ("robots.txt", "https://www.siyavula.com/robots.txt"),
        ("terms.html", "https://www.siyavula.com/info/terms-and-conditions"),
    ):
        file = raw / "grades-10-12" / filename
        if file.exists():
            inspections.append(
                {
                    "source_url": url,
                    "source_file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "raw_file": str(file),
                    "reason": "source navigation or permission inspection",
                }
            )
    write(args.root / "sources.jsonl", sources)
    write(args.root / "inspection-registry.jsonl", inspections)
    write(args.root / "reviews.jsonl", reviews)


if __name__ == "__main__":
    main()
