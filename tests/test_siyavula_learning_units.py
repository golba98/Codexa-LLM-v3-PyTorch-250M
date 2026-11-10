import hashlib
import zipfile
from pathlib import Path

from scripts.datasets.mathematics.extract_siyavula_learning_units import extract_member


def test_pairing_context_assets_and_empty_solutions(tmp_path: Path) -> None:
    member = "OPS/xhtml/maths10/chapter.cnxmlplus.html"
    raw = b"""<html xmlns="http://www.w3.org/1999/xhtml"><body>
    <div class="problemset"><span class="exerciseTitle">Exercise 1.1</span>
      <div class="header"><p>Show your work.</p></div>
      <div class="entry"><div class="problem"><p>Calculate <img src="equation/a.png"/></p></div>
        <div class="solution"><p>Four.</p></div></div>
      <div class="entry"><div class="problem"><p>State a number.</p></div>
        <div class="solution"></div></div>
    </div>
    <div class="worked_example"><h1 class="title">Example 1</h1>
      <div class="question"><p>Find two.</p></div><div class="workstep"><p>Answer two.</p></div></div>
    </body></html>"""
    asset = b"fake image source bytes"
    archive = tmp_path / "book.epub"
    with zipfile.ZipFile(archive, "w") as book:
        book.writestr(member, raw)
        book.writestr("OPS/xhtml/maths10/equation/a.png", asset)
    with zipfile.ZipFile(archive) as book:
        result = extract_member(10, "source", "bookhash", member, raw, book)
    assert [r["record_type"] for r in result] == [
        "exercise_with_source_solution",
        "practice_only",
        "worked_example",
    ]
    assert result[0]["source_solution"] == ["Four."]
    assert "Show your work." in result[0]["context"]
    assert result[0]["visual_assets"][0]["sha256"] == hashlib.sha256(asset).hexdigest()
    assert result[0]["extraction"] == "visual-dependent"
    assert result[1]["source_solution_status"] == "absent_or_empty"
    assert result[1]["answer_verification"] == "not_applicable"
    assert result[2]["step_verification"] == "pending"
    assert all(r["admission_status"] == "blocked" for r in result)
