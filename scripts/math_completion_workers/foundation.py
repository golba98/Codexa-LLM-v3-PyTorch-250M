"""Recover bounded English teacher guidance, keeping every review gate open."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

SCHOOL = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
WORKERS = SCHOOL / "corpus-v1/workers"
MENTAL = re.compile(r"(?im)^mental maths\s*$")
GAME = re.compile(r"(?im)^game\s*$")
CONCEPT = re.compile(r"(?im)^concept development\s*$")
WATCH = re.compile(r"(?im)^what to look out for this week\s*$")
FOREIGN = re.compile(r"(?i)\b(leerders|onderwysers|hierdie|getalle|leerlinge|jy|hulle)\b")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "teacher" not in str(row.get("source_id", "")):
        return []
    text = row.get("original_source_text")
    if not isinstance(text, str):
        return []
    mental, concept, watch = (
        list(MENTAL.finditer(text)),
        list(CONCEPT.finditer(text)),
        list(WATCH.finditer(text)),
    )
    if len(mental) != 1 or len(concept) != 1 or len(watch) != 1:
        return []
    if not (mental[0].end() < concept[0].start() < watch[0].start()):
        return []
    games = [m for m in GAME.finditer(text) if mental[0].end() < m.start() < concept[0].start()]
    if len(games) > 1:
        return []
    stop = games[0].start() if games else concept[0].start()
    spans = [
        ("mental_mathematics_guidance", mental[0].end(), stop),
        ("teacher_diagnostic_guidance", watch[0].end(), len(text)),
    ]
    output = []
    for kind, start, end in spans:
        body = text[start:end].strip()
        if len(body) < 80 or not re.search(r"[.!?][)\]\s]*$", body):
            continue
        # Mixed-language/visual uncertainty is recorded, never auto-cleared.
        flags = [
            "source-page visual inspection pending",
            "mathematical and instructional review pending",
            "exact objective mapping pending",
            "third-party rights and protected assessment screening pending",
        ]
        if FOREIGN.search(body):
            flags.append("possible non-English text")
        if re.search(
            r"(?i)\b(picture|diagram|image|table|draw|worksheet|video|cards|apparatus|beads|number track)\b",
            body,
        ):
            flags.append("possible visual or physical-resource dependency")
        key = f"{row['source_id']}\0{row['page_or_section']}\0{kind}"
        output.append(
            {
                "record_id": digest(key.encode()),
                "grade": row["grade"],
                "grade_band": "1-3",
                "record_type": kind,
                "origin": "source_extracted",
                "original_text": body,
                "normalized_text": body,
                "source_id": row["source_id"],
                "source_hash": row["source_file_sha256"],
                "source_url": row.get("source_url"),
                "source_pdf": row.get("absolute_pdf"),
                "page_or_section": row["page_or_section"],
                "parent_record_id": row["id"],
                "source_span": {"start": start, "end": end},
                "licence": row.get("licence"),
                "attribution": row.get("attribution"),
                "extraction": "bounded_source_section",
                "source_fidelity": "pending",
                "mathematical_verification": "pending",
                "rights": "exact edition notice retained; third-party review pending",
                "curriculum_mapping": "unresolved",
                "overlap": "not_screened",
                "admission_status": "blocked",
                "review_exposure": "development",
                "flags": flags,
            }
        )
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    all_rows = []
    input_hashes = {}
    layout_checks = []
    cache = {}
    for ver in (1, 2, 3):
        folder = WORKERS / f"foundation-bala-v{ver}"
        source = folder / "records.jsonl"
        data = source.read_bytes()
        input_hashes[str(source)] = digest(data)
        for line in data.splitlines():
            row = json.loads(line)
            if "teacher" not in str(row.get("source_id", "")):
                continue
            pdf = folder / row["raw_file"]
            key = str(pdf)
            if key not in cache:
                cache[key] = digest(pdf.read_bytes())
            if cache[key] != row["source_file_sha256"]:
                raise ValueError(f"source PDF hash mismatch: {pdf}")
            row["absolute_pdf"] = key
            found = candidates(row)
            if not found:
                continue
            match = re.fullmatch(r"PDF page (\d+)", row["page_or_section"])
            if not match:
                raise ValueError("unrecognized PDF page")
            page = int(match.group(1))
            proc = subprocess.run(
                ["pdftotext", "-f", str(page), "-l", str(page), "-layout", key, "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            for item in found:
                match_layout = normalized(item["original_text"]) in normalized(proc.stdout)
                layout_checks.append(
                    {
                        "id": item["record_id"],
                        "layout_text_agrees": match_layout,
                        "method": "independent layout-mode text extraction; not independent mathematical or visual verification",
                    }
                )
                if not match_layout:
                    item["flags"].append("layout-mode extraction did not match")
            all_rows.extend(found)
    if len({x["record_id"] for x in all_rows}) != len(all_rows):
        raise ValueError("duplicate source identities")
    args.output.mkdir(parents=True)
    for name, rows in [("records.jsonl", all_rows), ("layout-agreement.jsonl", layout_checks)]:
        with (args.output / name).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "records": len(all_rows),
        "by_grade": dict(Counter(str(r["grade"]) for r in all_rows)),
        "by_type": dict(Counter(r["record_type"] for r in all_rows)),
        "layout_agreement": sum(c["layout_text_agrees"] for c in layout_checks),
        "inputs_sha256": input_hashes,
        "source_pdfs_sha256": cache,
        "output_sha256": {
            n: digest((args.output / n).read_bytes())
            for n in ("records.jsonl", "layout-agreement.jsonl")
        },
        "status": "all held; no final admission",
        "limitations": [
            "Sections are teacher guidance, not standalone student lessons or complete objective coverage.",
            "Layout agreement does not prove visual, mathematical, language, copyright or assessment clearance.",
            "Some teacher pages lack exact sentinels and remain unprocessed.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: report[k] for k in ("records", "by_grade", "by_type", "layout_agreement")})
    )


if __name__ == "__main__":
    main()
