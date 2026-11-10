#!/usr/bin/env python3
"""Recover review candidates from an exact 2011 Grade 9 PDF, not approved data."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SOURCE_HASH = "bff41fb07fc0ff853ea54061c7b45775a1a9e84b7f4e2524c9fea80426b945b7"
# Page numbers are printed textbook page numbers, from physical PDF page 5.
MODULES = [
    ("1.1", "Numbers - where do they come from?", 1),
    ("1.2", "Easier algebra with exponents", 12),
    ("1.3", "Why all the fuss about Pythagoras?", 19),
    ("1.4", "How long is a piece of string?", 28),
    ("1.5", "Money Matters", 34),
    ("2.1", "The algebra of the four basic operations", 41),
    ("2.2", "Geometry of lines and triangles", 53),
    ("2.3", "Space and shape", 64),
    ("2.4", "Congruency", 72),
    ("2.5", "Similarity", 83),
    ("2.6", "Worksheets", 92),
    ("3.1", "Number patterns", 101),
    ("3.2", "Understanding what graphs tell us", 112),
    ("3.3", "Understanding how equations are represented on a graph", 123),
    ("3.4", "Finding the equation of a straight line graph from a diagram", 134),
    ("3.5", "Solving simple problems by forming and solving equations", 139),
    ("3.6", "Collecting information to answer general questions", 147),
    ("3.7", "Analyse data for meaningful patterns and measures", 153),
    ("3.8", "Extract meaningful information from data", 162),
    ("3.9", "Understanding the context and vocabulary of probability", 173),
    ("4.1", "Explore and identify the characteristics of some quadrilaterals", 181),
    ("4.2", "Compare quadrilaterals for similarities and differences", 187),
    ("4.3", "Understanding quadrilaterals and their properties in problems", 193),
    ("4.4", "Drawing plan and side views of three-dimensional objects to scale", 196),
    ("4.5", "Understand and use the principle of translation, learning suitable notations", 200),
]
BOUNDARY = re.compile(
    r"^\s*(?:CLASS WORK|HOMEWORK ASSIGNMENT|CLASS ASSIGNMENT|ENRICHMENT ASSIGNMENT|ASSESSMENT|ACTIVITY(?:\s+\d+)?|MEMORANDUM)\s*$",
    re.IGNORECASE,
)
VISUAL = re.compile(
    r"\b(?:figure|diagram|graph|picture|illustrat\w*|shown (?:above|below)|table|sketch|draw)\b",
    re.IGNORECASE,
)


def recover(pages: list[str]) -> tuple[list[dict], list[dict]]:
    if len(pages) < 213:
        raise ValueError("PDF has fewer pages than the verified table of contents")
    records, manifest = [], []
    # Several modules share a physical page. Split at the numbered heading,
    # not at the page boundary, so preceding material is not misattributed.
    module_lines: list[list[tuple[int, str]]] = [[] for _ in MODULES]
    current = -1
    for physical in range(7, 214):
        for line in pages[physical - 1].splitlines(keepends=True):
            if current + 1 < len(MODULES):
                upcoming = MODULES[current + 1]
                if physical == upcoming[2] + 6 and re.match(
                    r"^\s*" + re.escape(upcoming[0]) + r"\s+\S", line
                ):
                    current += 1
            if current >= 0:
                module_lines[current].append((physical, line))
    if current != len(MODULES) - 1:
        raise ValueError(f"Could not find all numbered module headings: last index {current}")
    for i, (section, title, printed_start) in enumerate(MODULES):
        printed_end = MODULES[i + 1][2] if i + 1 < len(MODULES) else 207
        # Printed page 1 is physical PDF page 7; list indexing starts at 0.
        chunks: list[tuple[int, int, str]] = []
        buf: list[str] = []
        start_page = printed_start + 6
        end_page = start_page
        for physical, line in module_lines[i]:
            if BOUNDARY.match(line) and buf and "".join(buf).strip():
                chunks.append((start_page, end_page, "".join(buf)))
                buf = []
            if not buf:
                start_page = physical
            buf.append(line)
            end_page = physical
        if buf and "".join(buf).strip():
            chunks.append((start_page, end_page, "".join(buf)))
        ids = []
        for ordinal, (start, end, text) in enumerate(chunks, 1):
            digest = hashlib.sha256(text.encode()).hexdigest()
            rid = f"siyavula2011:g9:{section}:{ordinal:04d}:{digest[:12]}"
            ids.append(rid)
            records.append(
                {
                    "record_id": rid,
                    "origin": "source_derived",
                    "grade": 9,
                    "source_id": "siyavula-grade9-col11056-2011",
                    "source_hash": SOURCE_HASH,
                    "source_section": section,
                    "source_section_title": title,
                    "physical_pdf_page_start": start,
                    "physical_pdf_page_end": end,
                    "original_text": text,
                    "text_sha256": digest,
                    "record_type": "mixed_source_unit_pending_classification",
                    "visual_reference_detected": bool(VISUAL.search(text)),
                    "control_or_missing_glyph_detected": bool(
                        re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]|\[U\+[0-9A-F]+\]", text)
                    ),
                    "source_fidelity": "pending",
                    "mathematical_verification": "pending",
                    "rights": "edition_CC_BY_3.0_module_and_asset_review_pending",
                    "curriculum_mapping": "pending_pre_CAPS_source",
                    "evaluation_overlap": "not_screened",
                    "admission": "blocked",
                    "normalization": "none; pdftotext layout output",
                }
            )
        manifest.append(
            {
                "source_id": "siyavula-grade9-col11056-2011",
                "section": section,
                "title": title,
                "printed_page_start": printed_start,
                "printed_page_end": printed_end,
                "physical_page_start": printed_start + 6,
                "physical_page_end": printed_end + 6,
                "acquisition": "available",
                "extraction": "partial_pending_fidelity_and_visual_review",
                "record_ids": ids,
            }
        )
    return records, manifest


def main() -> None:
    pdf, output = map(Path, sys.argv[1:])
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Unexpected edition or altered PDF")
    output.mkdir(parents=True, exist_ok=True)
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True
    ).stdout.decode("utf-8")
    records, manifest = recover(text.split("\f"))
    for filename, rows in [("records.jsonl", records), ("sections.jsonl", manifest)]:
        path = output / filename
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Refusing to overwrite changed artifact: {path}")
        path.write_text(payload)
    print(
        json.dumps(
            {
                "records": len(records),
                "sections": len(manifest),
                "glyph_issues": sum(r["control_or_missing_glyph_detected"] for r in records),
                "visual_references": sum(r["visual_reference_detected"] for r in records),
            }
        )
    )


if __name__ == "__main__":
    main()
