"""Inventory retained Grade 8–9 textbook evidence and official ATP work items.

This is a deterministic development queue; no source or curriculum item is
approved by its presence. ATP bullet extraction is a source locator requiring
human reconciliation with complete page context and current notices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )


def split_bullets(page: str) -> list[tuple[int, str]]:
    """Capture each apparent ATP bullet with physical line locator; no approval implied."""
    output: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for line_no, line in enumerate(page.splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        if re.match(r"^[\uf0b7•−–]\s*", text):
            if current:
                output.append((start, " ".join(current)))
            start = line_no
            current = [text]
        elif current:
            # Full source text remains in raw PDF; never infer missing table cells.
            current.append(text)
    if current:
        output.append((start, " ".join(current)))
    return output


def policy_items(
    path: Path, grade: int, policy_url: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError("policy input is not PDF")
    proc = subprocess.run(["pdftotext", "-raw", str(path), "-"], check=True, capture_output=True)
    text = proc.stdout.decode("utf-8")
    items = []
    pages = []
    term = None
    for pageno, page in enumerate(text.split("\f"), 1):
        if not page.strip():
            continue
        match = re.search(
            r"2026\s+ANNUAL\s+TEACHING\s+PLANS[^\n]*GRADE\s+"
            + str(grade)
            + r"[^\n]*TERM\s*([1-4])",
            page,
            re.IGNORECASE,
        )
        if match:
            term = int(match.group(1))
        bullets = split_bullets(page)
        pages.append(
            {
                "grade": grade,
                "page": pageno,
                "term": term,
                "bullet_candidates": len(bullets),
                "status": "manual_reconciliation_required",
                "source_sha256": sha(data),
            }
        )
        for lineno, bullet in bullets:
            ident = sha(f"{sha(data)}\0{pageno}\0{lineno}\0{bullet}".encode())
            items.append(
                {
                    "id": ident,
                    "grade": grade,
                    "official_phase": "senior",
                    "grade_band": "8-9",
                    "strand": None,
                    "topic": None,
                    "subtopic": None,
                    "objective_text": bullet,
                    "term": term,
                    "classification": "unresolved",
                    "mapping_status": "unreviewed_bullet_candidate",
                    "policy": {
                        "url": policy_url,
                        "sha256": sha(data),
                        "pdf_page": pageno,
                        "text_line": lineno,
                        "version": "2026 ATP",
                    },
                    "warning": "Raw PDF bullet extraction may merge columns, headings or assessment lines; validate source page before treating as an objective.",
                }
            )
    return items, pages


def build(source: Path, policy: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    paths = [
        source / n for n in ("records.jsonl", "sources.jsonl", "inventory.jsonl", "toc-items.jsonl")
    ]
    input_hash = {str(p): sha(p.read_bytes()) for p in paths}
    records, sources, inventory, toc = (lines(p) for p in paths)
    source_by_id = {r["source_id"]: r for r in sources}
    inventories = {r["url"]: r for r in inventory if r.get("url")}
    units = []
    raw_hashes = {}
    ids = set()
    for r in records:
        if r["id"] in ids:
            raise ValueError("duplicate record ID " + r["id"])
        ids.add(r["id"])
        raw = source / r["raw_file"]
        if str(raw) not in raw_hashes:
            raw_hashes[str(raw)] = sha(raw.read_bytes())
        if raw_hashes[str(raw)] != r.get("source_file_sha256"):
            raise ValueError("raw hash mismatch " + str(raw))
        s = source_by_id.get(r.get("source_id"))
        if not s or s.get("sha256") != r.get("source_file_sha256"):
            raise ValueError("missing or mismatched source registry")
        units.append(
            {
                "record_id": r["id"],
                "origin": "source_derived",
                "grade": r["grade"],
                "grade_band": "8-9",
                "original_record": r,
                "source_evidence_path": str(raw),
                "objective_ids": [],
                "curriculum_mapping": "pending_official_objective_link",
                "acquisition": "available",
                "extraction": "visual-dependent"
                if r.get("diagram_dependency") is not False
                else "pending_fidelity",
                "source_fidelity": "pending",
                "math_answer": "pending" if r.get("final_answer") else "not_applicable",
                "solution_steps": "pending" if r.get("source_solution") else "not_applicable",
                "rights": "unresolved_independent_edition_and_asset_review",
                "overlap": "not_screened",
                "review_exposure": "development",
                "admission": "not_requested",
            }
        )
    sections = []
    chapter = {}
    count_by_url = Counter(r["source_url"] for r in records)
    for position, t in enumerate(toc, 1):
        grade = t["grade"]
        label = t["label"]
        url = t["document_url"]
        if re.match(r"^Chapter\s+\d+", label, re.IGNORECASE):
            chapter[grade] = label
        known = inventories.get(url, {})
        identity = sha(f"{t['source_toc_url']}\0{t['url']}\0{label}".encode())
        sections.append(
            {
                "section_id": identity,
                "grade": grade,
                "source_toc_url": t["source_toc_url"],
                "chapter": chapter.get(grade),
                "label": label,
                "url": t["url"],
                "document_url": url,
                "toc_position": position,
                "source_id": known.get("source_id"),
                "acquisition_status": "available"
                if known.get("sha256")
                else "not_acquired_or_unknown",
                "extraction_status": known.get("status", "unknown"),
                "extraction_reason": known.get("reason"),
                "extracted_parent_records_at_url": count_by_url[url],
                "rights": "unresolved_independent_edition_and_asset_review",
                "objective_ids": [],
            }
        )
    acquisition = json.loads((policy / "acquisition.json").read_text())
    policies = {x["name"]: x for x in acquisition if x.get("status") == "acquired"}
    objectives = []
    page_inventory = []
    for grade in (8, 9):
        entry = policies[f"grade{grade}-atp"]
        path = policy / f"grade{grade}-atp.pdf"
        if sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError("policy hash mismatch")
        obj, pages = policy_items(path, grade, entry["url"])
        objectives.extend(obj)
        page_inventory.extend(pages)
    output.mkdir(parents=True)
    dump(output / "objective-candidates.jsonl", objectives)
    dump(output / "policy-pages.jsonl", page_inventory)
    dump(output / "source-registry.jsonl", sources)
    dump(output / "source-sections.jsonl", sections)
    dump(output / "records-source.jsonl", units)
    chapter_counts = Counter(
        s["grade"] for s in sections if re.match(r"^Chapter\s+\d+", s["label"], re.IGNORECASE)
    )
    counts = {
        "record_count": len(units),
        "records_by_grade": dict(Counter(r["grade"] for r in units)),
        "source_sections": len(sections),
        "sections_by_grade": dict(Counter(s["grade"] for s in sections)),
        "toc_chapters": dict(chapter_counts),
        "objective_bullet_candidates_by_grade": dict(Counter(o["grade"] for o in objectives)),
        "atp_pages_by_grade": dict(Counter(p["grade"] for p in page_inventory)),
        "extraction_dispositions": dict(Counter(s["extraction_status"] for s in sections)),
        "rights_cleared": 0,
        "final_admitted": 0,
        "limitations": [
            "All records remain held; source content may be incomplete or visually dependent.",
            "Bullet candidates are not verified objectives or complete curriculum coverage.",
            "Grade 9 online contents end in Term 2; full-year alternative source is needed.",
            "Source links and objective mappings have not been reviewed for third-party rights or protected evaluation.",
            "No tokenizer totals are claimed for unreviewed source text.",
        ],
    }
    report = {
        "inputs": input_hash,
        "raw_file_sha256": raw_hashes,
        "policy_sources": acquisition,
        "counts": counts,
        "outputs": {p.name: sha(p.read_bytes()) for p in sorted(output.glob("*.jsonl"))},
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.policy, a.output)))


if __name__ == "__main__":
    main()
