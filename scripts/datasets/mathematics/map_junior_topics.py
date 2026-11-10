"""Link observed 2026 ATP topic headings to retained textbook chapters.

The links are chapter-level navigational leads, not verified objective coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Grade, 2026 term, official ATP PDF page, heading, textbook chapter numbers.
TOPICS = [
    (8, 1, 1, "Whole numbers", [1]),
    (8, 1, 1, "Integers", [2]),
    (8, 1, 1, "Common fractions", [13]),
    (8, 1, 1, "Decimal fractions", [14]),
    (8, 1, 1, "Numeric and geometric patterns", [4]),
    (8, 2, 5, "Exponents", [3]),
    (8, 2, 5, "Algebraic expressions", [6, 8]),
    (8, 2, 5, "Equations", [7, 9, 22]),
    (8, 2, 5, "Functions and relationships", [5, 21]),
    (8, 2, 5, "Graphs", [23]),
    (8, 3, 8, "Data handling", [18, 19, 20]),
    (8, 3, 8, "Geometry of straight lines", [12]),
    (8, 3, 8, "Geometry of 2D shapes", [10, 11]),
    (8, 3, 9, "Probability", [26]),
    (8, 4, 12, "Theorem of Pythagoras", [15]),
    (8, 4, 12, "Transformation geometry", [24]),
    (8, 4, 12, "Area and perimeter of 2-D shapes", [16]),
    (8, 4, 12, "Surface area and volume of 3-D objects", [17, 25]),
    (9, 1, 1, "Whole numbers", [1]),
    (9, 1, 1, "Integers", [2]),
    (9, 1, 1, "Exponents", [5]),
    (9, 1, 1, "Numeric and geometric patterns", [6]),
    (9, 1, 1, "Functions and relationships", [7]),
    (9, 1, 1, "Algebraic expressions", [8]),
    (9, 2, 4, "Algebraic expressions", [8]),
    (9, 2, 4, "Algebraic equations", [9]),
    (9, 2, 4, "Graphs", []),
    (9, 2, 4, "Geometry of straight lines", [13]),
    (9, 3, 7, "Geometry of 2D shapes", [11, 12]),
    (9, 3, 7, "Area and perimeter", [15]),
    (9, 3, 7, "Collect, organize and summarize data", []),
    (9, 3, 7, "Probability", []),
    (9, 3, 8, "Represent data", []),
    (9, 3, 8, "Interpret, analyse, and report data", []),
    (9, 4, 11, "Geometry of 3-D objects", []),
    (9, 4, 11, "Surface area and volume of 3-D objects", []),
    (9, 4, 11, "Transformation geometry", []),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sections = [
        json.loads(line)
        for line in (a.inventory / "source-sections.jsonl").read_text().splitlines()
        if line.strip()
    ]
    policy = json.loads((a.inventory / "manifest.json").read_text())["policy_sources"]
    byname = {v["name"]: v for v in policy}
    output: list[dict[str, Any]] = []
    for grade, term, page, title, chapters in TOPICS:
        v = byname[f"grade{grade}-atp"]
        ident = hashlib.sha256(
            f"{v['sha256']}\0{grade}\0{term}\0{page}\0{title}".encode()
        ).hexdigest()
        selected = []
        for section in sections:
            if section["grade"] != grade:
                continue
            match = re.match(r"^Chapter\s+(\d+):", section.get("chapter") or "")
            if match and int(match.group(1)) in chapters:
                selected.append(section["section_id"])
        output.append(
            {
                "id": ident,
                "grade": grade,
                "official_phase": "senior",
                "grade_band": "8-9",
                "official_topic_heading": title,
                "term": term,
                "policy": {
                    "url": v["url"],
                    "sha256": v["sha256"],
                    "pdf_page": page,
                    "version": "2026 ATP",
                },
                "source_chapter_numbers": chapters,
                "candidate_source_section_ids": selected,
                "classification": "topic_heading_verified_objectives_unreviewed",
                "mapping_status": "chapter_links_provisional_not_coverage",
                "limitation": "Heading-level navigational mapping; topic subskills, current circular reconciliation and instructional coverage require independent review.",
            }
        )
    with a.output.open("x", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "topics": len(output),
                "grade8": sum(r["grade"] == 8 for r in output),
                "grade9": sum(r["grade"] == 9 for r in output),
                "zero_matched_source_topics": sum(
                    not r["candidate_source_section_ids"] for r in output
                ),
                "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
