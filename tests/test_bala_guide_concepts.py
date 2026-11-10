"""Source-section recovery tests with synthetic CC0 fixtures."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "foundation_guide_concepts",
    Path(__file__).parents[1]
    / "scripts"
    / "datasets"
    / "mathematics"
    / "extract_bala_guide_concepts.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extracts_only_complete_source_bounded_text(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    unit = (
        "This week, learners count five counters and compare groups. "
        "They discuss which of the two groups is larger or smaller."
    )
    row = {
        "source_id": "teacher-book",
        "original_source_text": f"12\nComparing groups\nConcept development\n{unit}\nWhat to look out for this week\nExclude this text.",
        "raw_file": "source.pdf",
        "source_file_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_or_section": "PDF page 4",
        "grade": 1,
        "id": "parent",
    }
    record = module.recover(row, tmp_path)
    assert record["original_text"] == unit
    assert record["title"] == "Comparing groups"
    assert record["visual_dependencies"] == "unknown"
    assert record["admission_status"] == "blocked"
    pdf.write_bytes(b"%PDF-1.4 modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.recover(row, tmp_path)


def test_incomplete_or_ambiguous_boundaries_are_rejected(tmp_path):
    row = {
        "source_id": "teacher-book",
        "original_source_text": "Concept development\nText with no end marker.",
    }
    assert module.recover(row, tmp_path) is None
    row["original_source_text"] = (
        "Concept development\nConcept development\nText\nWhat to look out for this week"
    )
    assert module.recover(row, tmp_path) is None
