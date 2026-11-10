import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from codexa.data.math_curriculum_senior_v2 import atp_candidates, toc_inventory, write_idempotent


def test_toc_reports_missing_anchors_and_checks_source_hash(tmp_path: Path) -> None:
    epub = tmp_path / "grade-10.epub"
    nav = """<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol><li><span>Chapter 1</span><ol><li><a href="one.xhtml#present">A</a></li><li><a href="one.xhtml#missing">B</a></li></ol></li></ol></nav></body></html>"""
    with zipfile.ZipFile(epub, "w") as book:
        book.writestr("OPS/xhtml/maths10/maths10.nav.xhtml", nav)
        book.writestr(
            "OPS/xhtml/maths10/one.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p id="present">Text</p></body></html>',
        )
    hashed = hashlib.sha256(epub.read_bytes()).hexdigest()
    inv = {
        "source_epub_sha256": hashed,
        "inventory": [
            {
                "member": "OPS/xhtml/maths10/one.xhtml",
                "status": "processed",
                "candidate_paragraphs": 1,
            }
        ],
    }
    records = toc_inventory(10, epub, inv)
    assert [x["anchor_status"] for x in records] == ["present", "missing"]
    assert records[0]["hierarchy"] == ["Chapter 1", "A"]
    assert records[0]["rights"] == "unresolved_edition_notice_conflict"
    with pytest.raises(ValueError, match="hash differs"):
        toc_inventory(10, epub, {**inv, "source_epub_sha256": "bad"})


def test_atp_candidates_are_unreviewed_and_require_matching_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"policy")

    def fake_run(command, **kwargs):
        page = int(command[command.index("-f") + 1])
        return SimpleNamespace(
            stdout=f"GRADE 10 (TERM {page - 1})\n1. First objective\n2. Second objective\nDATE\n"
        )

    monkeypatch.setattr("codexa.data.math_curriculum_senior_v2.subprocess.run", fake_run)
    objectives, pages = atp_candidates(10, pdf, "https://example.test/policy.pdf")
    assert len(objectives) == 8 and len(pages) == 4
    assert {x["term"] for x in objectives} == {1, 2, 3, 4}
    assert all(x["classification"] == "unresolved" for x in objectives)
    assert all(x["policy"]["sha256"] == hashlib.sha256(b"policy").hexdigest() for x in objectives)


def test_resume_refuses_changed_output(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file"
    write_idempotent(target, b"one")
    write_idempotent(target, b"one")
    with pytest.raises(FileExistsError):
        write_idempotent(target, b"two")
    assert target.read_bytes() == b"one"
