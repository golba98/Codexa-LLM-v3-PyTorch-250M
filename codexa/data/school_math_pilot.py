"""Shared schema and conservative validation for source-derived school mathematics."""

import hashlib
from typing import Any

FIELDS = [
    "id",
    "grade",
    "grade_band",
    "source_grade",
    "source_curriculum",
    "target_curriculum",
    "grade_mapping_status",
    "topic",
    "subtopic",
    "record_type",
    "language",
    "source_id",
    "source_url",
    "source_file_sha256",
    "source_title",
    "source_edition",
    "page_or_section",
    "source_exercise_id",
    "licence",
    "attribution",
    "original_source_text",
    "explanation",
    "question",
    "source_solution",
    "final_answer",
    "diagram_dependency",
    "content_origin",
    "normalization_applied",
    "extraction_worker",
    "verification_status",
    "verification_methods",
    "verification_notes",
    "codexa_token_count",
]
BANDS = {
    **dict.fromkeys(range(1, 4), "1-3"),
    **dict.fromkeys(range(4, 8), "4-7"),
    **dict.fromkeys(range(8, 10), "8-9"),
    **dict.fromkeys(range(10, 13), "10-12"),
}


def record_id(source_id: str, location: str) -> str:
    """Identify a source location without depending on extraction order."""
    return hashlib.sha256(f"{source_id}\0{location}".encode()).hexdigest()


def validate_record(row: dict[str, Any]) -> list[str]:
    errors = []
    if set(row) != set(FIELDS):
        errors.append("schema fields differ")
    grade = row.get("grade")
    if type(grade) is not int or BANDS.get(grade) != row.get("grade_band"):
        errors.append("invalid grade/band")
    for key in (
        "id",
        "topic",
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
    ):
        if not row.get(key):
            errors.append(f"missing {key}")
    if row.get("id") != record_id(str(row.get("source_id")), str(row.get("page_or_section"))):
        errors.append("invalid deterministic id")
    if row.get("content_origin") != "source_extracted":
        errors.append("invalid origin")
    if row.get("verification_status") not in ("pending", "verified", "quarantined"):
        errors.append("invalid status")
    if row.get("language") != "en":
        errors.append("language not English")
    if row.get("grade_mapping_status") not in ("source_labelled", "mapped"):
        errors.append("invalid mapping status")
    if row.get("record_type") not in (
        "explanation",
        "worked_example",
        "exercise_answer",
        "exercise_solution",
        "practice_only",
    ):
        errors.append("invalid record type")
    if row.get("verification_status") == "verified" and (
        row.get("diagram_dependency") is not False
        or not row.get("verification_methods")
        or not row.get("verification_notes")
    ):
        errors.append("verified record lacks visual clearance or review evidence")
    return errors
