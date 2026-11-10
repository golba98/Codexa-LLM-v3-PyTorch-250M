from codexa.data.school_math_pilot import BANDS, FIELDS, record_id, validate_record


def test_schema_rejects_invalid_grade_and_missing_provenance():
    row = dict.fromkeys(FIELDS)
    row.update(grade=3, grade_band="4-7", content_origin="source_extracted")
    errors = validate_record(row)
    assert "invalid grade/band" in errors
    assert "missing licence" in errors


def test_identity_is_stable_and_location_sensitive():
    assert record_id("source", "page 1") == record_id("source", "page 1")
    assert record_id("source", "page 1") != record_id("source", "page 2")
    assert BANDS[7] == "4-7"


def test_verified_requires_source_review_and_visual_clearance():
    row = dict.fromkeys(FIELDS)
    row.update(
        id=record_id("s", "loc"),
        grade=8,
        grade_band="8-9",
        source_id="s",
        page_or_section="loc",
        content_origin="source_extracted",
        verification_status="verified",
        diagram_dependency=True,
    )
    assert "verified record lacks visual clearance or review evidence" in validate_record(row)


def test_invalid_source_origin_is_not_accepted():
    row = dict.fromkeys(FIELDS)
    row["content_origin"] = "synthetic"
    assert "invalid origin" in validate_record(row)
