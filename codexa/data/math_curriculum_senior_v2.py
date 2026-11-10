"""Create auditable senior mathematics curriculum and EPUB inventory candidates.

The original policy and EPUB bytes remain the source of truth. This module
never marks records verified, rights-cleared, overlap-cleared or admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

SEEDS = {
    10: [
        "Algebraic expressions",
        "Exponents",
        "Number patterns",
        "Equations and inequalities",
        "Functions and graphs",
        "Trigonometry",
        "Euclidean geometry",
        "Analytical geometry",
        "Financial mathematics",
        "Statistics",
        "Measurement",
        "Probability",
    ],
    11: [
        "Exponents and surds",
        "Equations and inequalities",
        "Number patterns",
        "Functions and graphs",
        "Analytical geometry",
        "Trigonometry",
        "Euclidean geometry",
        "Measurement",
        "Financial mathematics",
        "Probability",
        "Statistics",
        "Linear programming",
    ],
    12: [
        "Sequences and series",
        "Functions and inverses",
        "Financial mathematics",
        "Trigonometry",
        "Polynomial algebra",
        "Differential calculus",
        "Analytical geometry",
        "Euclidean geometry",
        "Statistics",
        "Probability and counting",
    ],
}
NS = {"x": "http://www.w3.org/1999/xhtml", "e": "http://www.idpf.org/2007/ops"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values
    ).encode()


def write_idempotent(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def atp_candidates(
    grade: int, pdf: Path, url: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pdf_hash = digest(pdf.read_bytes())
    objectives: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    for page in range(2, 6):
        text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-raw", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        term = page - 1
        if not re.search(rf"GRADE\s+{grade}\s*\(TERM\s*{term}\)", text, re.IGNORECASE):
            raise ValueError(f"ATP page/term mismatch: grade {grade}, page {page}")
        pages.append(
            {
                "grade": grade,
                "term": term,
                "pdf_page": page,
                "policy_sha256": pdf_hash,
                "policy_url": url,
                "page_text": text,
                "status": "raw_reading_order_requires_visual_comparison",
            }
        )
        # Numbered blocks may span parallel table columns or include headings; each
        # is therefore a candidate with page evidence, never an official count.
        blocks = re.split(r"(?m)^\s*(?=\d{1,2}(?:\.\d+)?[.)]\s*\S)", text)
        ordinal = 0
        for block in blocks:
            match = re.match(r"^\s*(\d{1,2}(?:\.\d+)?)[.)]\s*(\S.*)", block, re.DOTALL)
            if not match:
                continue
            body = re.split(r"(?m)^\s*(?:DATE\s*$|COMPLETED\s*$|SBA\s)", match.group(2))[0].strip()
            if not body:
                continue
            ordinal += 1
            key = f"{pdf_hash}:{page}:{ordinal}"
            objectives.append(
                {
                    "id": digest(key.encode()),
                    "grade": grade,
                    "official_phase": "FET",
                    "term": term,
                    "pdf_page": page,
                    "source_number": match.group(1),
                    "ordinal_on_page": ordinal,
                    "objective_text": body,
                    "topic": None,
                    "classification": "unresolved",
                    "mapping_status": "unreviewed_pdf_reading_order",
                    "policy": {"url": url, "sha256": pdf_hash, "page": page},
                    "limitations": [
                        "equations and column order may be damaged by PDF text extraction",
                        "requires comparison with policy page and baseline CAPS",
                    ],
                }
            )
    return objectives, pages


def toc_inventory(grade: int, epub: Path, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    archive_hash = digest(epub.read_bytes())
    if archive_hash != inventory["source_epub_sha256"]:
        raise ValueError(f"EPUB hash differs from recovery inventory: {epub}")
    by_member = {item["member"]: item for item in inventory["inventory"] if "member" in item}
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(epub) as book:
        nav_path = f"OPS/xhtml/maths{grade}/maths{grade}.nav.xhtml"
        root = ET.fromstring(book.read(nav_path))
        nav = root.find('.//x:nav[@e:type="toc"]', NS)
        if nav is None:
            raise ValueError(f"no EPUB table of contents: {epub}")
        base = str(Path(nav_path).parent)
        names = set(book.namelist())

        def walk(ol: ET.Element, parents: list[str]) -> None:
            for li in ol.findall("x:li", NS):
                title_el = next(
                    (node for node in li if node.tag in (f"{{{NS['x']}}}span", f"{{{NS['x']}}}a")),
                    None,
                )
                if title_el is None:
                    continue
                title = " ".join("".join(title_el.itertext()).split())
                href = title_el.get("href")
                hierarchy = [*parents, title]
                if href:
                    path_part, _, anchor = href.partition("#")
                    member = f"{base}/{path_part}"
                    item = by_member.get(member, {})
                    exists = member in names
                    anchor_status = "not_checked"
                    member_hash = None
                    if exists:
                        data = book.read(member)
                        member_hash = digest(data)
                        try:
                            parsed = ET.fromstring(data)
                            anchor_status = (
                                "present"
                                if not anchor or any(n.get("id") == anchor for n in parsed.iter())
                                else "missing"
                            )
                        except ET.ParseError:
                            anchor_status = "unparseable_member"
                    section_key = f"{archive_hash}\0{href}"
                    records.append(
                        {
                            "section_id": digest(section_key.encode()),
                            "grade": grade,
                            "source_id": f"siyavula-math-grade-{grade}-unbranded-epub-{archive_hash[:16]}",
                            "source_hash": archive_hash,
                            "hierarchy": hierarchy,
                            "title": title,
                            "href": href,
                            "member": member,
                            "member_sha256": member_hash,
                            "anchor": anchor or None,
                            "anchor_status": anchor_status,
                            "acquisition": "available" if exists else "missing",
                            "extraction": item.get("status", "not_processed"),
                            "member_candidate_paragraphs": item.get("candidate_paragraphs", 0),
                            "objective_mapping": "unreviewed",
                            "rights": "unresolved_edition_notice_conflict",
                            "section_review": "not_started",
                        }
                    )
                for nested in li.findall("x:ol", NS):
                    walk(nested, hierarchy)

        for ol in nav.findall("x:ol", NS):
            walk(ol, [])
    return records


def run(policy_dir: Path, epub_dir: Path, recovery_dir: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    policies = json.loads((policy_dir / "acquisition.json").read_text())
    urls = {
        item["id"]: item["url"] for item in policies if item.get("status") in {"cached", "acquired"}
    }
    if any(f"grade-{g}-atp" not in urls for g in (10, 11, 12)):
        raise ValueError("all three senior ATP acquisition receipts required")
    seeds: list[dict[str, Any]] = []
    objectives: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    inputs = {
        str(policy_dir / "acquisition.json"): digest((policy_dir / "acquisition.json").read_bytes())
    }
    for grade in (10, 11, 12):
        pdf = policy_dir / f"grade-{grade}-atp.pdf"
        epub = epub_dir / f"grade-{grade}.epub"
        inventory_file = recovery_dir / f"grade-{grade}-inventory.json"
        unit_file = recovery_dir / f"grade-{grade}-units.jsonl"
        for file in (pdf, epub, inventory_file, unit_file):
            inputs[str(file)] = digest(file.read_bytes())
        for topic in SEEDS[grade]:
            seeds.append(
                {
                    "grade": grade,
                    "official_phase": "FET",
                    "topic": topic,
                    "classification": "user_supplied_seed_unverified",
                    "objective_id": None,
                    "status": "mapping_pending",
                }
            )
        candidate, evidence = atp_candidates(grade, pdf, urls[f"grade-{grade}-atp"])
        objectives.extend(candidate)
        pages.extend(evidence)
        inventory = json.loads(inventory_file.read_text())
        sections.extend(toc_inventory(grade, epub, inventory))
        source_id = f"siyavula-math-grade-{grade}-unbranded-epub-{inputs[str(epub)][:16]}"
        sources.append(
            {
                "source_id": source_id,
                "grade": grade,
                "title": f"Siyavula Mathematics Grade {grade} unbranded English EPUB",
                "source_file": str(epub),
                "sha256": inputs[str(epub)],
                "acquisition": "available",
                "edition": "publisher EPUB; no numbered edition established",
                "rights": "unresolved_edition_notice_conflict",
                "rights_evidence": str(recovery_dir / "caps-rights-review-01"),
            }
        )
        unit_rows = read_jsonl(unit_file)
        for row in unit_rows:
            if row.get("grade") != grade or row.get("source_epub_sha256") != inputs[str(epub)]:
                raise ValueError(f"unit source mismatch: {row.get('id')}")
        write_idempotent(output / f"grade-{grade}-source-records.jsonl", jsonl_bytes(unit_rows))
    write_idempotent(output / "topic-seeds.jsonl", jsonl_bytes(seeds))
    write_idempotent(output / "objective-candidates.jsonl", jsonl_bytes(objectives))
    write_idempotent(output / "policy-page-evidence.jsonl", jsonl_bytes(pages))
    write_idempotent(output / "source-section-manifest.jsonl", jsonl_bytes(sections))
    write_idempotent(output / "source-registry.jsonl", jsonl_bytes(sources))
    report = {
        "inputs_sha256": inputs,
        "seed_topics": len(seeds),
        "objective_candidates": len(objectives),
        "atp_pages": len(pages),
        "toc_nodes": len(sections),
        "by_grade_toc_nodes": dict(Counter(str(r["grade"]) for r in sections)),
        "section_states": dict(Counter(r["anchor_status"] for r in sections)),
        "records": {
            str(g): len(read_jsonl(recovery_dir / f"grade-{g}-units.jsonl")) for g in (10, 11, 12)
        },
        "limitations": [
            "ATP numbered blocks are unreviewed candidates, not a confirmed objective denominator.",
            "User topic seeds require reconciliation with current CAPS and ATPs.",
            "EPUB paragraphs are unreviewed; image and contextual dependencies remain.",
            "Edition licence discrepancies, evaluation boundaries and admission remain unresolved.",
            "Anchor existence and processed member do not prove that a subsection has been completely extracted.",
        ],
    }
    write_idempotent(output / "report.json", json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-dir", "epub-dir", "recovery-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.policy_dir, args.epub_dir, args.recovery_dir, args.output)))


if __name__ == "__main__":
    main()
