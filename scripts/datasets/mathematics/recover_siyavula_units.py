"""Recover source-linked paragraph units from locally acquired Siyavula EPUBs.

This tool never approves data. All units require independent context, content,
rights, CAPS and protected-evaluation review. Images are never guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
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
# Each PNG was individually viewed on 2026-11-10. SHA-256 checks are essential:
# the same filename represents separately rendered images in each EPUB.
REVIEWED = {
    "e24ef761386789a042e7c0af2521bf65e2087551a09ea1ae641fb451fbc22723": "x",
    "df1383ab8d65e982f819228bacefa64f7bbead48fdadd1975193cc1e18449960": "y",
    "efc9626262b5ebe67881a43b4c7d425e7512e3b5261657043f7d03654e6b99e2": "n",
    "67788fc27360e25bdd661b4d9900a2ffe9b6c5c081e58fd658a13ef1cef0f699": "x",
    "26dd33262684695d6d5de886629fab11e48039a05b29430068ecf1fc467a1139": "y",
    "2c8b729d083f3c3b8f5433795ffd1d7539dfaa7e498e64dd54fa04b35f805f11": "n",
    "df896dd628386722e4e5d932a911eb8f37b19b9dcc69e3b1f541f81792352d60": "x",
    "6b18495c90eb8239cedd27cb66f106db79bfa8ff3702d1e5c664572f2562e45b": "y",
    "7f1323cd416cb17ac531cf982c1e391096564d34430f5d028b3ce322877173e3": "n",
}


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def render(
    node: ET.Element, book: zipfile.ZipFile, member: str
) -> tuple[str, list[dict[str, Any]], list[str]]:
    parts: list[str] = []
    assets: list[dict[str, Any]] = []
    flags: list[str] = []

    def visit(el: ET.Element) -> None:
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "img":
            src = el.get("src", "")
            name = posixpath.normpath(posixpath.join(posixpath.dirname(member), src))
            if not src or name.startswith(("../", "/")):
                flags.append("unsafe_or_missing_asset")
                parts.append(" [UNRESOLVED IMAGE] ")
            else:
                try:
                    value = book.read(name)
                except KeyError:
                    flags.append("asset_missing")
                    parts.append(" [UNRESOLVED IMAGE] ")
                else:
                    filehash = digest(value)
                    transcript = (
                        REVIEWED.get(filehash)
                        if "math-inline" in el.get("class", "").split()
                        else None
                    )
                    assets.append(
                        {
                            "member": name,
                            "sha256": filehash,
                            "transcription": transcript,
                            "review": "two_visual_reviews_same_bytes_context_unreviewed"
                            if transcript
                            else "unresolved",
                        }
                    )
                    if transcript:
                        parts.append(transcript)
                    else:
                        parts.append(" [UNRESOLVED IMAGE] ")
                        flags.append("unresolved_image")
        elif tag in {"math", "svg", "object", "canvas", "video"}:
            flags.append("unresolved_embedded_content")
            parts.append(" [UNRESOLVED EMBEDDED CONTENT] ")
        else:
            if el.text:
                parts.append(el.text)
            for child in el:
                visit(child)
        if el.tail:
            parts.append(el.tail)

    visit(node)
    text = re.sub(r"\s+", " ", "".join(parts)).strip()
    return text, assets, sorted(set(flags))


def recover(archive: Path, grade: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_hash = digest(archive.read_bytes())
    source_id = f"siyavula-math-grade-{grade}-unbranded-epub-{archive_hash[:16]}"
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as book:
        rootfile = ET.fromstring(book.read("META-INF/container.xml")).find(".//c:rootfile", NS)
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB rootfile missing")
        package_path = str(rootfile.get("full-path"))
        package = ET.fromstring(book.read(package_path))
        items = {x.get("id"): x for x in package.findall("./o:manifest/o:item", NS)}
        for spine, itemref in enumerate(package.findall("./o:spine/o:itemref", NS), 1):
            item = items.get(itemref.get("idref"))
            if (
                item is None
                or item.get("media-type") != "application/xhtml+xml"
                or not item.get("href")
            ):
                inventory.append({"spine": spine, "status": "unsupported"})
                continue
            member = posixpath.normpath(
                posixpath.join(posixpath.dirname(package_path), str(item.get("href")))
            )
            if member.startswith(("../", "/")):
                raise ValueError("unsafe archive member")
            try:
                data = book.read(member)
                document = ET.fromstring(data)
            except (KeyError, ET.ParseError) as exc:
                inventory.append(
                    {"spine": spine, "member": member, "status": "failed", "reason": str(exc)}
                )
                continue
            body = document.find(".//x:body", NS)
            if body is None:
                inventory.append({"spine": spine, "member": member, "status": "missing_body"})
                continue
            parents = {child: parent for parent in body.iter() for child in parent}
            source_heading = next(
                (
                    " ".join("".join(h.itertext()).split())
                    for h in body.iter()
                    if h.tag.rsplit("}", 1)[-1] == "h2"
                ),
                member,
            )
            count = 0
            for ordinal, p in enumerate(body.iter(f"{{{NS['x']}}}p"), 1):
                ancestors = []
                current = p
                while current in parents:
                    current = parents[current]
                    ancestors.append(current)
                classes = {c for a in ancestors for c in a.get("class", "").split()}
                # Context/solutions/exercises need whole-unit extraction, never one paragraph alone.
                excluded = {
                    "worked_example",
                    "workstep",
                    "question",
                    "exercise",
                    "exercises",
                    "note",
                    "activity",
                }
                if classes & excluded or any(
                    a.tag.rsplit("}", 1)[-1] in {"li", "table"} for a in ancestors
                ):
                    continue
                content, assets, flags = render(p, book, member)
                if not content:
                    continue
                if any(
                    x in content.lower()
                    for x in ("figure", "diagram", "graph below", "following", "above", "below")
                ):
                    flags.append("possible_context_dependency")
                if content.endswith((":", ";", ",")):
                    flags.append("incomplete_sentence_or_list_lead")
                if re.match(
                    r"(?i)^(this|these|those|such|it|they|therefore|then|so|also|however|for example|in this|we have seen)\b",
                    content,
                ):
                    flags.append("possible_context_dependency")
                if any(
                    term in source_heading.casefold()
                    for term in (
                        "copyright",
                        "acknowledg",
                        "license",
                        "licence",
                        "contents",
                        "preface",
                    )
                ):
                    flags.append("front_matter_or_rights_text")
                if p.find(".//x:a", NS) is not None:
                    flags.append("hyperlink_or_external_content")
                # The p ordinal is within the immutable source member, not an extraction-order index.
                loc = f"{member}#paragraph-{ordinal}"
                top = p
                while top in parents and parents[top] is not body:
                    top = parents[top]
                siblings = list(body)
                top_index = siblings.index(top) + 1 if top in siblings else None
                old_loc = f"{member}#body-child-{top_index}"
                rows.append(
                    {
                        "id": digest(f"{source_id}\0{loc}".encode()),
                        "grade": grade,
                        "source_id": source_id,
                        "source_epub_sha256": archive_hash,
                        "source_member": member,
                        "source_member_sha256": digest(data),
                        "source_location": loc,
                        "parent_section_id": digest(f"{source_id}\0{old_loc}".encode()),
                        "source_heading": source_heading,
                        "record_type": "explanation_candidate",
                        "text": content,
                        "source_fragment": ET.tostring(p, encoding="unicode"),
                        "assets": assets,
                        "flags": sorted(set(flags)),
                        "licence": "CC BY 3.0; individual content rights pending",
                        "attribution": "Siyavula Education, Mathematics, siyavula.com; CC BY 3.0; extracted and changed",
                        "status": "quarantined" if flags else "pending_independent_review",
                        "training_admission": "not_approved",
                        "review_requirements": [
                            "source fidelity and surrounding context",
                            "independent transcription where applicable",
                            "math correctness",
                            "third-party rights",
                            "CAPS subtopic",
                            "protected evaluation overlap",
                        ],
                    }
                )
                count += 1
            inventory.append(
                {
                    "spine": spine,
                    "member": member,
                    "status": "processed",
                    "candidate_paragraphs": count,
                }
            )
    return rows, {"grade": grade, "source_epub_sha256": archive_hash, "inventory": inventory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for grade in (10, 11, 12):
        rows, report = recover(args.raw / f"grade-{grade}.epub", grade)
        with (args.output / f"grade-{grade}-units.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["units"] = len(rows)
        report["status_counts"] = dict(Counter(x["status"] for x in rows))
        report["flags"] = dict(Counter(flag for row in rows for flag in row["flags"]))
        report["transcribed_asset_occurrences"] = sum(
            len([a for a in row["assets"] if a["transcription"]]) for row in rows
        )
        (args.output / f"grade-{grade}-inventory.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in report.items() if k != "inventory"}))


if __name__ == "__main__":
    main()
