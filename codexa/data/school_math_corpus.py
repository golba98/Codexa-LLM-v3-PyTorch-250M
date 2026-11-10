"""Conservative, bounded validation helpers for source-grounded mathematics."""

import hashlib
import re
from typing import Any

from codexa.data.school_math_pilot import BANDS, FIELDS, record_id

CORPUS_FIELDS = frozenset(FIELDS) | {"source_hierarchy", "raw_file", "extraction_version"}
KINDS = {
    "explanation",
    "definition",
    "worked_example",
    "exercise",
    "practice_only",
    "objective",
    "context",
}
REQUIRED = (
    "source_id",
    "source_url",
    "source_file_sha256",
    "source_title",
    "source_edition",
    "page_or_section",
    "licence",
    "attribution",
    "original_source_text",
    "extraction_worker",
    "source_grade",
    "source_curriculum",
    "topic",
    "raw_file",
    "extraction_version",
)


def validate(row: dict[str, Any]) -> list[str]:
    errors = []
    if not CORPUS_FIELDS.issubset(row):
        errors.append("missing fields: " + ", ".join(sorted(CORPUS_FIELDS - row.keys())))
    grade = row.get("grade")
    if type(grade) is not int or BANDS.get(grade) != row.get("grade_band"):
        errors.append("invalid grade band")
    for field in REQUIRED:
        if not row.get(field):
            errors.append("missing " + field)
    if row.get("id") != record_id(str(row.get("source_id")), str(row.get("page_or_section"))):
        errors.append("invalid id")
    if row.get("record_type") not in KINDS:
        errors.append("invalid type")
    if row.get("grade_mapping_status") not in ("source_labelled", "mapped"):
        errors.append("invalid grade mapping")
    if row.get("language") != "en" or row.get("content_origin") != "source_extracted":
        errors.append("invalid language or origin")
    if row.get("diagram_dependency") not in (True, False, None):
        errors.append("invalid visual flag")
    if not isinstance(row.get("source_hierarchy"), list):
        errors.append("invalid hierarchy")
    if not isinstance(row.get("normalization_applied"), list):
        errors.append("invalid normalization")
    text = row.get("original_source_text")
    if isinstance(text, str):
        if "\ufffd" in text or "\x00" in text:
            errors.append("corrupt text")
        if any(s in text for s in ("<bos>", "<eos>", "<pad>", "<unk>")):
            errors.append("reserved tokenizer token")
        if any(text.count(a) != text.count(b) for a, b in ((r"\(", r"\)"), (r"\[", r"\]"))):
            errors.append("unbalanced TeX delimiter")
    return errors


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def arithmetic_holdout(text: str) -> bool:
    """Flag exact numeric expressions assigned to original arithmetic val/test families."""
    for match in re.finditer(r"(?<![\d.])(\d{1,2})\s*([+−-])\s*(\d{1,2})(?![\d.])", text):
        a, op, b = int(match[1]), match[2].replace("−", "-"), int(match[3])
        if a > 20 or b > 20 or (op == "+" and a + b > 20) or (op == "-" and a < b):
            continue
        if op == "+" and a > b:
            a, b = b, a
        identity = hashlib.sha256(f"v1:{op}:{a}:{b}".encode()).hexdigest()
        if int(identity, 16) % 10 in (0, 1):
            return True
    return False
