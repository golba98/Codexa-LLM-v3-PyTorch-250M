"""Regression tests for source-level split and quarantine boundaries."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "recover_grade7_units.py"
spec = importlib.util.spec_from_file_location("middle_grade7_recovery", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def base_row(fragment):
    return {
        "grade": 7,
        "source_id": "source-1",
        "page_or_section": "chapter/1",
        "id": "parent",
        "source_url": "https://example.invalid",
        "source_file_sha256": "hash",
        "raw_file": "raw/book.html",
        "source_fragment_html": fragment,
        "record_type": "practice_only",
        "source_grade": "7",
        "source_curriculum": "CAPS",
        "original_source_text": "Do not use a calculator at all in section 1.1.",
        "licence": "CC-BY",
        "attribution": "publisher",
    }


def test_visual_sibling_does_not_quarantine_complete_unit():
    fragment = (
        "<ol><li><p>Calculate each.</p><ol><li><p>2 + 3</p></li>"
        '<li><p>4 + 5</p><img src="missing.png"/></li></ol></li></ol>'
    )
    row = base_row(fragment)
    items, failures = module.extract_group(row, "Calculate each. 2 + 3 4 + 5", set())
    assert not failures and len(items) == 2
    assert (
        items[0]["question"]
        == "Do not use a calculator at all in section 1.1. Calculate each. 2 + 3"
    )
    assert items[0]["extraction_status"] == "pending_source_review"
    assert "visual_or_mathml_dependency" in items[1]["flags"]
    assert all(item["answer"] is None and item["admission_status"] == "blocked" for item in items)
    assert len({item["record_id"] for item in items}) == 2


def test_cross_reference_and_missing_source_are_quarantined():
    fragment = "<ol><li><p>Use the example above.</p><ol><li><p>7 + 2</p></li></ol></li></ol>"
    items, _ = module.extract_group(base_row(fragment), "Use the example above.", set())
    assert len(items) == 1
    assert {"possible_external_context_reference", "item_text_not_located_in_source_html"} <= set(
        items[0]["flags"]
    )
    assert items[0]["extraction_status"] == "quarantined"


def test_invalid_fragment_is_reported():
    items, failures = module.extract_group(base_row("<ol><li>"), "", set())
    assert items == [] and failures == ["invalid_source_fragment_xml"]


def test_prior_review_does_not_transfer_to_changed_text():
    row = {
        "record_id": "a",
        "question": "Use note: 1 + 2",
        "flags": [],
        "extraction_status": "pending_source_review",
    }
    module.reconcile_previous_review(row, {"a": "1 + 2"})
    assert row["previous_review_question_match"] is False
    assert row["extraction_status"] == "quarantined"
    assert "previous_review_question_text_differs" in row["flags"]
    matching = {
        "record_id": "a",
        "question": "1 + 2",
        "flags": [],
        "extraction_status": "pending_source_review",
    }
    module.reconcile_previous_review(matching, {"a": "1 + 2"})
    assert matching["previous_review_question_match"] is True
    assert matching["extraction_status"] == "pending_source_review"
