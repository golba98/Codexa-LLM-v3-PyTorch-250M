"""Recover bounded English teacher-guide concept sections as held source units."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHOOL = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
SOURCES = SCHOOL / "corpus-v1/workers"
START = re.compile(r"(?im)^concept development\s*$")
STOP = re.compile(r"(?im)^what to look out for this week\s*$")
FOREIGN = re.compile(r"(?i)\b(leerders|onderwysers|hierdie|getalle|leerlinge|jy|hulle)\b")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recover(row: dict[str, Any], source_root: Path) -> dict[str, Any] | None:
    if "teacher" not in str(row.get("source_id", "")):
        return None
    text = row.get("original_source_text")
    if not isinstance(text, str):
        return None
    starts = list(START.finditer(text))
    stops = list(STOP.finditer(text))
    if len(starts) != 1 or len(stops) != 1 or starts[0].end() >= stops[0].start():
        return None
    unit = text[starts[0].end() : stops[0].start()].strip()
    # Preserve line breaks and hyphenation exactly; no text is translated or repaired.
    if len(unit) < 80:
        return None
    path = source_root / row["raw_file"]
    if digest(path.read_bytes()) != row["source_file_sha256"]:
        raise ValueError(f"source PDF hash mismatch: {path}")
    lines = [line.strip() for line in text[: starts[0].start()].splitlines() if line.strip()]
    title = next((line for line in lines if not re.fullmatch(r"\d+", line)), None)
    identity = f"{row['source_id']}\0{row['page_or_section']}\0concept-development"
    flags = [
        "visual dependency requires source-page review",
        "source fidelity requires visual review",
        "full section context and mathematical review pending",
        "CAPS mapping pending",
        "third-party rights and protected assessment review pending",
    ]
    if FOREIGN.search(unit):
        flags.append("possible non-English text requires review")
    if not re.search(r"[.!?]$", unit):
        flags.append("possible incomplete final sentence")
    return {
        "record_id": digest(identity.encode()),
        "grade": row["grade"],
        "grade_band": "1-3",
        "original_grade": row.get("source_grade"),
        "original_curriculum": row.get("source_curriculum"),
        "origin": "source_extracted",
        "record_type": "teacher_concept_section_candidate",
        "title": title,
        "original_text": unit,
        "normalized_text": unit,
        "source_id": row["source_id"],
        "source_url": row.get("source_url"),
        "source_hash": row["source_file_sha256"],
        "source_pdf": str(path),
        "source_page": row["page_or_section"],
        "source_parent_id": row["id"],
        "source_span": {"start": starts[0].end(), "end": stops[0].start()},
        "licence": row.get("licence"),
        "attribution": row.get("attribution"),
        "source_fidelity": "pending",
        "math_verification": "pending",
        "visual_dependencies": "unknown",
        "rights": "exact edition notice retained; per-unit third-party status pending",
        "curriculum_mapping": "unmapped",
        "evaluation_overlap": "not_screened",
        "admission_status": "blocked",
        "flags": flags,
        "review_exposure": "development",
        "normalization": "none; outer whitespace trimmed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = []
    input_hashes = {}
    for version in (1, 2, 3):
        root = SOURCES / f"foundation-bala-v{version}"
        source = root / "records.jsonl"
        data = source.read_bytes()
        input_hashes[str(source)] = digest(data)
        for line in data.splitlines():
            row = json.loads(line)
            item = recover(row, root)
            if item is not None:
                items.append(item)
    ids = [item["record_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("record ID collision")
    args.output.mkdir(parents=True)
    output = args.output / "teacher-concepts.jsonl"
    with output.open("x", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "records": len(items),
        "per_grade": dict(Counter(str(x["grade"]) for x in items)),
        "input_sha256": input_hashes,
        "output_sha256": digest(output.read_bytes()),
        "status": "all held; English source fidelity, visuals, correctness, rights and protected screening pending",
        "limitations": [
            "Teacher prose is not a student-facing textbook or verified curriculum coverage.",
            "Section-level visual dependencies may be internal to the source page or neighbouring pages.",
            "The deterministic sentinel extraction does not discover all instructional content.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["per_grade"]))


if __name__ == "__main__":
    main()
