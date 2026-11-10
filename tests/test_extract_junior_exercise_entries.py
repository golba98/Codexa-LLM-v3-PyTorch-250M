"""Tests for the offline, source-linked exercise-entry splitter."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "junior_exercises", Path(__file__).parents[1] / "scripts" / "extract_junior_exercise_entries.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_static_problem_solution_association_and_visual_hold():
    source = {
        "source_id": "book-section",
        "grade": 8,
        "source_url": "https://example.test/page",
        "sha256": "abc",
        "raw_file": "source.html",
        "licence": "unresolved",
        "attribution": "test",
    }
    raw = b"""<html><div id="content"><section class="exercises" id="ex1"><div class="entry">
    <div class="problem"><p>Calculate.</p><ol start="2"><li>1 + 2</li><li>3 &amp; 4</li></ol></div>
    <div class="solution"><p>3</p><svg></svg></div></div></section>
    <section class="exercises interactive_questions">temp text</section></div></html>"""
    found, stats = module.extract(source, raw)
    assert len(found) == 1 and stats["interactive_placeholder"] == 1
    assert "1 + 2" in found[0]["question"] and "2." in found[0]["question"]
    assert "3 & 4" in found[0]["question"]
    assert found[0]["source_solution"] == "3"
    assert found[0]["visual_dependency"] is True
    assert found[0]["source_entry_html"].startswith('<div class="entry">')
    assert found[0]["math_answer_verification"] == "pending"
    assert found[0]["rights"] == "unresolved" and found[0]["final_admission"] == "not_requested"
    assert module.extract(source, raw)[0][0]["id"] == found[0]["id"]


def test_ambiguous_pairing_is_not_extracted():
    source = {
        "source_id": "book",
        "grade": 8,
        "source_url": "https://example.test",
        "sha256": "x",
        "raw_file": "raw",
    }
    raw = b'<div id="content"><section class="exercises"><div class="entry"><div class="problem">Q</div><div class="solution">A</div><div class="solution">B</div></div></section></div>'
    found, stats = module.extract(source, raw)
    assert found == [] and stats["ambiguous_entry"] == 1
