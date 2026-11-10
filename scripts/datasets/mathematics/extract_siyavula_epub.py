"""Extract auditable, *unapproved* section candidates from local Siyavula EPUBs.

The EPUB image-based equations are not converted into guessed text. This script
is offline and never alters its input archives. Inputs must be the publisher's
unbranded English CC BY editions; inspect rights and obtain them separately.
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

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "o": "http://www.idpf.org/2007/opf",
    "x": "http://www.w3.org/1999/xhtml",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plain(element: ET.Element) -> str:
    """Preserve explicit missing image markers; never drop visual mathematics."""
    out: list[str] = []

    def visit(node: ET.Element) -> None:
        name = node.tag.rsplit("}", 1)[-1]
        if name == "img":
            out.append(" [IMAGE: " + node.get("src", "missing-source") + "] ")
        else:
            if name in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
                out.append("\n")
            if node.text:
                out.append(node.text)
            for child in node:
                visit(child)
        if node.tail:
            out.append(node.tail)

    visit(element)
    return "\n".join(" ".join(line.split()) for line in "".join(out).splitlines() if line.strip())


def extract(
    archive: Path, grade: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    archive_hash = sha(archive.read_bytes())
    source_id = f"siyavula-math-grade-{grade}-unbranded-epub-{archive_hash[:16]}"
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as book:
        container = ET.fromstring(book.read("META-INF/container.xml"))
        rootfile = container.find(".//c:rootfile", NS)
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB package path missing")
        package_path = str(rootfile.get("full-path"))
        package = ET.fromstring(book.read(package_path))
        items = {item.get("id"): item for item in package.findall("./o:manifest/o:item", NS)}
        for position, ref in enumerate(package.findall("./o:spine/o:itemref", NS), 1):
            item = items.get(ref.get("idref"))
            if item is None or item.get("media-type") != "application/xhtml+xml":
                inventory.append(
                    {
                        "spine_position": position,
                        "status": "unsupported",
                        "item_id": ref.get("idref"),
                    }
                )
                continue
            href = item.get("href")
            if not href:
                inventory.append({"spine_position": position, "status": "missing_href"})
                continue
            member = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), href))
            if member.startswith(("../", "/")):
                raise ValueError("Unsafe EPUB member")
            try:
                content = book.read(member)
                doc = ET.fromstring(content)
            except (KeyError, ET.ParseError) as error:
                inventory.append(
                    {
                        "spine_position": position,
                        "member": member,
                        "status": "failed",
                        "reason": str(error),
                    }
                )
                continue
            body = doc.find(".//x:body", NS)
            if body is None:
                inventory.append(
                    {"spine_position": position, "member": member, "status": "missing_body"}
                )
                continue
            heading_node = body.find(".//x:h2", NS)
            heading = plain(heading_node) if heading_node is not None else member
            units = list(body)
            count = 0
            for index, unit in enumerate(units, 1):
                value = plain(unit)
                if not value:
                    continue
                location = f"{member}#body-child-{index}"
                image = any(
                    el.tag.rsplit("}", 1)[-1] in ("img", "svg", "math") for el in unit.iter()
                )
                classes = unit.get("class", "").split()
                kind = (
                    "worked_example"
                    if "worked_example" in classes
                    else ("exercise" if "exercises" in classes else "section")
                )
                # A section may contain nested worked examples and exercises; do not claim counts.
                rows.append(
                    {
                        "id": sha((source_id + "\0" + location).encode()),
                        "grade": grade,
                        "grade_band": "10-12",
                        "topic_source_heading": heading,
                        "record_type": kind,
                        "source_id": source_id,
                        "source_location": location,
                        "source_member_sha256": sha(content),
                        "original_source_text": value,
                        "source_markup_member": member,
                        "has_visual_or_math_images": image,
                        "content_origin": "source_extracted",
                        "verification_status": "quarantined" if image else "pending",
                        "verification_notes": "Equation/diagram images not transcribed; source context and rights require independent review."
                        if image
                        else "Pending source comparison, rights, curriculum mapping and protected-evaluation checks.",
                    }
                )
                count += 1
            inventory.append(
                {
                    "spine_position": position,
                    "member": member,
                    "sha256": sha(content),
                    "status": "extracted" if count else "empty",
                    "units": count,
                    "heading": heading,
                }
            )
    source = {
        "source_id": source_id,
        "grade": grade,
        "publisher": "Siyavula Education",
        "title": f"Mathematics Grade {grade}, English unbranded EPUB",
        "edition": "publisher EPUB; undated catalogue version",
        "source_url": f"https://www.siyavula.com/downloads/books/maths/Gr{grade}_Mathematics_Learner_Eng_CC-BY.epub",
        "catalogue_url": "https://www.siyavula.com/read",
        "licence": "CC BY 3.0",
        "attribution": "Siyavula Education, Mathematics, siyavula.com, CC BY 3.0; indicate changes",
        "raw_file": str(archive),
        "sha256": archive_hash,
        "restrictions": "Exclude separately owned content and marks; review source licence and terms before any use or distribution.",
        "eligibility": "pending rights and content review",
    }
    return rows, inventory, source


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for grade in (10, 11, 12):
        archive = args.raw / f"grade-{grade}.epub"
        if not archive.exists():
            report[str(grade)] = {"status": "missing_archive"}
            continue
        rows, inventory, source = extract(archive, grade)
        dump(args.output / f"grade-{grade}-records.jsonl", rows)
        dump(args.output / f"grade-{grade}-inventory.jsonl", inventory)
        dump(args.output / f"grade-{grade}-source.jsonl", [source])
        report[str(grade)] = {
            "source_sha256": source["sha256"],
            "records": len(rows),
            "status_counts": dict(Counter(row["verification_status"] for row in rows)),
            "spine_counts": dict(Counter(row["status"] for row in inventory)),
        }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
