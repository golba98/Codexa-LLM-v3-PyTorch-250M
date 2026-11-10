"""Recover auditable Grade 7 subquestions from retained source fragments.

This is a source preparation step. It never asserts rights or training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

SCHOOL = Path("/mnt/codexa/pytorch/codexa-v3/school-math")
WORKER = SCHOOL / "corpus-v1/workers/grades-4-7"
INPUT = WORKER / "grade7-records.jsonl"
PRIOR = SCHOOL / "recovery-v1/grades-7-practice/candidates-03.jsonl"
REVIEW = SCHOOL / "recovery-foundation-v1/grade7-independent-review/per-record-review.jsonl"
VISUAL_TAGS = {"img", "svg", "table", "canvas", "video", "iframe", "object", "math"}
CROSSREF = re.compile(
    r"(?i)\b(?:above|below|following (?:diagram|table|figure|picture)|the (?:diagram|table|figure|picture|graph)|previous|preceding|next page|on page|question\s+\d|\d\s*\([a-z]\))\b"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.split())


def local(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1] if isinstance(node.tag, str) else ""


def node_text(node: ET.Element) -> str:
    return normalized(" ".join(node.itertext()))


def direct_content(node: ET.Element) -> str:
    """Text not belonging to nested lists; keep all context, including notes."""
    parts = [node.text or ""]
    for child in node:
        if local(child) not in {"ol", "ul"}:
            parts.extend(child.itertext())
        parts.append(child.tail or "")
    return normalized(" ".join(parts))


class PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def source_location(row: dict[str, Any], top_index: int, child_index: int | None) -> str:
    loc = f"{row['page_or_section']}/ol-{top_index}"
    return loc if child_index is None else f"{loc}/li-{child_index}"


def source_id(row: dict[str, Any], location: str) -> str:
    return digest(f"{row['source_id']}\0{location}".encode())


def prepare_one(
    row: dict[str, Any],
    top: ET.Element,
    top_index: int,
    item: ET.Element,
    child_index: int | None,
    page_text: str,
    prior: set[str],
) -> dict[str, Any]:
    parent_text = direct_content(top) if child_index is not None else ""
    item_text = direct_content(item)
    nested_lists = [node for node in item if local(node) in {"ol", "ul"}]
    selected_tags = {local(node) for node in item.iter()}
    parent_tags = {
        local(node) for child in top if local(child) not in {"ol", "ul"} for node in child.iter()
    }
    flags = []
    if nested_lists:
        flags.append("nested_subquestions_not_split")
    if (selected_tags | parent_tags) & VISUAL_TAGS:
        flags.append("visual_or_mathml_dependency")
    if any(
        local(node) == "div" and "note" in (node.get("class") or "").split() for node in item.iter()
    ):
        flags.append("embedded_item_note_needs_association_review")
    if not item_text:
        flags.append("empty_exercise")
    if child_index is not None and not parent_text:
        flags.append("parent_instruction_missing")
    if CROSSREF.search(parent_text + " " + item_text):
        flags.append("possible_external_context_reference")
    if item_text and item_text not in page_text:
        flags.append("item_text_not_located_in_source_html")
    if parent_text and parent_text not in page_text:
        flags.append("parent_text_not_located_in_source_html")
    source_context = row.get("original_source_text") or ""
    precondition = (
        "Do not use a calculator at all in section 1.1."
        if "Do not use a calculator at all in section 1.1." in source_context
        else None
    )
    loc = source_location(row, top_index, child_index)
    ident = source_id(row, loc)
    return {
        "record_id": ident,
        "origin": "source_extracted",
        "grade": 7,
        "original_grade": row.get("source_grade"),
        "original_curriculum": row.get("source_curriculum"),
        "target_curriculum": "South African CAPS",
        "grade_band": "4-7",
        "source_id": row.get("source_id"),
        "source_url": row.get("source_url"),
        "source_hash": row.get("source_file_sha256"),
        "source_raw_file": str(WORKER / row["raw_file"]),
        "source_location": loc,
        "source_parent_record_id": row.get("id"),
        "source_parent_location": row.get("page_or_section"),
        "source_hierarchy": row.get("source_hierarchy"),
        "source_parent_value": top.get("value"),
        "source_item_value": item.get("value"),
        "source_fragment": ET.tostring(item, encoding="unicode"),
        "source_instruction": parent_text or None,
        "source_item_text": item_text,
        "source_group_context": source_context,
        "source_precondition": precondition,
        "question": normalized(" ".join(x for x in (precondition, parent_text, item_text) if x)),
        "answer": None,
        "source_solution": None,
        "record_type": "practice_only_candidate",
        "licence": row.get("licence"),
        "attribution": row.get("attribution"),
        "objective_ids": [],
        "source_review_previous": ident in prior,
        "extraction_status": "quarantined" if flags else "pending_source_review",
        "flags": sorted(flags),
        "source_fidelity": "pending",
        "rights": "unresolved",
        "curriculum_mapping": "unresolved",
        "evaluation_overlap": "not_screened",
        "review_exposure": "development",
        "admission_status": "blocked",
        "native_token_count": None,
    }


def extract_group(
    row: dict[str, Any], page_text: str, prior: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    fragment = row.get("source_fragment_html")
    if not fragment:
        return [], ["missing_source_fragment"]
    try:
        root = ET.fromstring(fragment)
    except ET.ParseError:
        return [], ["invalid_source_fragment_xml"]
    if local(root) != "ol":
        return [], ["non_ordered_list_fragment"]
    result = []
    for top_index, top in enumerate(root, 1):
        if local(top) != "li":
            continue
        lists = [node for node in top if local(node) == "ol"]
        if not lists:
            result.append(prepare_one(row, top, top_index, top, None, page_text, prior))
        elif len(lists) == 1:
            for child_index, item in enumerate(lists[0], 1):
                if local(item) == "li":
                    result.append(
                        prepare_one(row, top, top_index, item, child_index, page_text, prior)
                    )
        else:
            result.append(prepare_one(row, top, top_index, top, None, page_text, prior))
    return result, []


def reconcile_previous_review(row: dict[str, Any], prior_questions: dict[str, str]) -> None:
    previous = prior_questions.get(row["record_id"])
    row["previous_review_question_match"] = (
        None if previous is None else normalized(row["question"]) == previous
    )
    if previous is not None and not row["previous_review_question_match"]:
        row["flags"] = sorted(set(row["flags"]) | {"previous_review_question_text_differs"})
        row["extraction_status"] = "quarantined"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = INPUT.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    prior_rows = [json.loads(line) for line in PRIOR.read_text().splitlines() if line.strip()]
    prior = {row["id"] for row in prior_rows}
    review_rows = [json.loads(line) for line in REVIEW.read_text().splitlines() if line.strip()]
    if {row["id"] for row in review_rows} != prior:
        raise ValueError("prior review does not cover prior records")
    cache: dict[str, str] = {}
    collected = []
    groups = []
    for row in rows:
        if row.get("record_type") != "practice_only":
            continue
        path = WORKER / row["raw_file"]
        data = path.read_bytes()
        if digest(data) != row.get("source_file_sha256"):
            raise ValueError(f"source hash mismatch: {path}")
        if str(path) not in cache:
            reader = PageText()
            reader.feed(data.decode("utf-8"))
            cache[str(path)] = normalized(" ".join(reader.parts))
        items, failures = extract_group(row, cache[str(path)], prior)
        collected.extend(items)
        groups.append(
            {
                "parent_id": row["id"],
                "source_id": row["source_id"],
                "extracted_units": len(items),
                "failure_reasons": failures,
            }
        )
    prior_questions = {row["id"]: normalized(row["question"]) for row in prior_rows}
    for row in collected:
        reconcile_previous_review(row, prior_questions)
    ids = [row["record_id"] for row in collected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source location identity; preserve input and investigate")
    normalized_text = defaultdict(list)
    for row in collected:
        normalized_text[digest(normalized(row["question"]).casefold().encode())].append(
            row["record_id"]
        )
    duplicates = [
        {"canonical_question_sha256": key, "record_ids": values}
        for key, values in normalized_text.items()
        if len(values) > 1
    ]
    args.output.mkdir(parents=True)
    for name, values in [
        ("records.jsonl", collected),
        ("group-dispositions.jsonl", groups),
        ("duplicate-questions.jsonl", duplicates),
    ]:
        (args.output / name).write_text(
            "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in values),
            encoding="utf-8",
        )
    report = {
        "input_sha256": digest(raw),
        "prior_sha256": digest(PRIOR.read_bytes()),
        "review_sha256": digest(REVIEW.read_bytes()),
        "groups": len(groups),
        "record_count": len(collected),
        "status_counts": dict(Counter(x["extraction_status"] for x in collected)),
        "flag_counts": dict(Counter(flag for x in collected for flag in x["flags"])),
        "previously_reviewed_ids_recovered": len(prior.intersection(ids)),
        "previous_review_exact_question_matches": sum(
            row["previous_review_question_match"] is True for row in collected
        ),
        "previous_review_question_differences": sum(
            row["previous_review_question_match"] is False for row in collected
        ),
        "prior_reviewed_total": len(prior),
        "duplicate_question_groups": len(duplicates),
        "training_admitted": 0,
        "limitations": [
            "HTML source comparison is a presence check only and does not establish source fidelity.",
            "No generated solution or official answer is inferred; evaluation separation remains unscreened.",
            "Cross-references and visual dependencies can evade heuristics; all records remain held.",
            "Grade 7 online source omits the second half of the school year.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
