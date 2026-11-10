"""Preserve paired Siyavula EPUB exercises and worked examples as held evidence.

Source image references and exact hashes remain explicit; no image is transcribed.
Nothing in the output is approved for model training or evaluated for correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.math_curriculum_senior_v2 import json_bytes, jsonl_bytes, write_idempotent

X = "{http://www.w3.org/1999/xhtml}"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classes(node: ET.Element) -> set[str]:
    return set(node.get("class", "").split())


def render(node: ET.Element, assets: list[str]) -> str:
    """Keep original words and explicit unresolved image locators."""
    parts: list[str] = []

    def visit(elem: ET.Element) -> None:
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "img":
            src = elem.get("src") or "missing-src"
            assets.append(src)
            parts.append(f" [SOURCE-IMAGE:{src}] ")
        else:
            if tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
                parts.append("\n")
            if elem.text:
                parts.append(elem.text)
            for child in elem:
                visit(child)
        if elem.tail:
            parts.append(elem.tail)

    visit(node)
    return "\n".join(" ".join(line.split()) for line in "".join(parts).splitlines() if line.strip())


def extract_member(
    grade: int, source_id: str, archive_hash: str, member: str, data: bytes, book: zipfile.ZipFile
) -> list[dict[str, Any]]:
    root = ET.fromstring(data)
    body = root.find(".//" + X + "body")
    if body is None:
        return []
    parents = {child: parent for parent in body.iter() for child in parent}
    members = set(book.namelist())
    output = []
    ordinal: Counter[str] = Counter()
    for elem in body.iter(X + "div"):
        cs = classes(elem)
        if "worked_example" in cs:
            questions = [child for child in elem if "question" in classes(child)]
            steps = [child for child in elem if "workstep" in classes(child)]
            if len(questions) != 1:
                continue
            kind = "worked_example"
        elif "entry" in cs:
            questions = [child for child in elem if "problem" in classes(child)]
            steps = [child for child in elem if "solution" in classes(child)]
            if len(questions) != 1:
                continue
            kind = "exercise_with_source_solution" if steps else "practice_only"
        else:
            continue
        refs: list[str] = []
        q = render(questions[0], refs)
        solution = [render(step, refs) for step in steps]
        has_solution = any(part.strip() for part in solution)
        if kind == "exercise_with_source_solution" and not has_solution:
            kind = "practice_only"
        ordinal[kind] += 1
        context = []
        ancestor = parents.get(elem)
        while ancestor is not None:
            if "problemset" in classes(ancestor):
                for child in ancestor:
                    if "header" in classes(child) or "exerciseTitle" in classes(child):
                        content = render(child, refs)
                        if content:
                            context.append(content)
            ancestor = parents.get(ancestor)
        context.reverse()
        if kind == "worked_example":
            for child in elem:
                if child.tag in (X + "h1", X + "h2") and "title" in classes(child):
                    context.append(render(child, refs))
        evidence = []
        for src in dict.fromkeys(refs):
            path = posixpath.normpath(
                posixpath.join(posixpath.dirname(member), src.split("#", 1)[0])
            )
            exists = path in members
            evidence.append(
                {
                    "source_ref": src,
                    "member": path,
                    "sha256": digest(book.read(path)) if exists else None,
                    "status": "available" if exists else "missing",
                }
            )
        location = f"{member}#{kind}-{ordinal[kind]}"
        row = {
            "record_id": digest((source_id + "\0" + location).encode()),
            "grade": grade,
            "grade_band": "10-12",
            "origin": "source_extracted",
            "record_type": kind,
            "source_id": source_id,
            "source_hash": archive_hash,
            "source_member": member,
            "source_member_sha256": digest(data),
            "page_or_section": location,
            "context": context,
            "question": q,
            "source_solution": solution,
            "source_solution_status": "present_unverified" if has_solution else "absent_or_empty",
            "final_answer": None,
            "source_fragment_xml": ET.tostring(elem, encoding="unicode"),
            "visual_assets": evidence,
            "extraction": "visual-dependent" if refs else "pending_source_fidelity_and_context",
            "source_fidelity": "pending",
            "answer_verification": "pending" if has_solution else "not_applicable",
            "step_verification": "pending" if has_solution else "not_applicable",
            "rights": "unresolved_edition_notice_conflict",
            "curriculum_mapping": "pending",
            "objective_ids": [],
            "overlap": "not_screened_under_approved_policy",
            "review_exposure": "development",
            "admission_status": "blocked",
            "limitations": [
                "Source solution is extracted, not independently verified.",
                "Parent exercise context and source rights require review.",
                "Unresolved image references are not mathematical transcriptions.",
            ],
        }
        output.append(row)
    return output


def run(raw: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "grades": {},
        "status": "all held; no verified solutions or approved data",
    }
    for grade in (10, 11, 12):
        epub = raw / f"grade-{grade}.epub"
        archive_hash = digest(epub.read_bytes())
        source_id = f"siyavula-math-grade-{grade}-unbranded-epub-{archive_hash[:16]}"
        units = []
        failures = []
        with zipfile.ZipFile(epub) as book:
            for member in sorted(book.namelist()):
                if not member.startswith(f"OPS/xhtml/maths{grade}/") or not member.endswith(
                    ".cnxmlplus.html"
                ):
                    continue
                try:
                    units.extend(
                        extract_member(
                            grade, source_id, archive_hash, member, book.read(member), book
                        )
                    )
                except (ET.ParseError, KeyError, UnicodeError) as error:
                    failures.append({"member": member, "error": str(error)})
        if len({u["record_id"] for u in units}) != len(units):
            raise ValueError(f"duplicate identity for grade {grade}")
        write_idempotent(output / f"grade-{grade}-learning-units.jsonl", jsonl_bytes(units))
        report["grades"][str(grade)] = {
            "source_sha256": archive_hash,
            "units": len(units),
            "types": dict(Counter(u["record_type"] for u in units)),
            "visual_dependent": sum(u["extraction"] == "visual-dependent" for u in units),
            "missing_assets": sum(
                any(a["status"] == "missing" for a in u["visual_assets"]) for u in units
            ),
            "failures": failures,
        }
    write_idempotent(output / "report.json", json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.output)))


if __name__ == "__main__":
    main()
