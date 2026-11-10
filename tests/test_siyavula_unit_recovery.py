"""Conservative tests for paragraph recovery and unresolved image isolation."""

import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "recover_siyavula_units.py"
spec = importlib.util.spec_from_file_location("recover_siyavula_units", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unit_identity_source_context_and_missing_diagram(tmp_path):
    path = tmp_path / "tiny.epub"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        z.writestr(
            "OPS/book.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="a" href="chapter.html" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="a"/></spine></package>',
        )
        z.writestr(
            "OPS/chapter.html",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><div class="section"><h2>Numbers</h2><p>Integers include zero.</p><p>See the diagram <img src="missing.png"/></p><div class="worked_example"><p>Unpaired solution step.</p></div><p>Whole numbers include zero.</p></div></body></html>',
        )
    rows, info = module.recover(path, 10)
    assert [r["text"] for r in rows] == [
        "Integers include zero.",
        "See the diagram [UNRESOLVED IMAGE]",
        "Whole numbers include zero.",
    ]
    assert rows[0]["status"] == "pending_independent_review"
    assert rows[1]["status"] == "quarantined"
    assert rows[2]["status"] == "pending_independent_review"
    assert len({r["id"] for r in rows}) == 3
    assert len({r["parent_section_id"] for r in rows}) == 1
    assert all(r["training_admission"] == "not_approved" for r in rows)
    assert module.recover(path, 10) == (rows, info)


def test_only_exact_reviewed_asset_hash_can_be_transcribed(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    path = tmp_path / "asset.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("OPS/x.png", b"reviewed")
        z.writestr("OPS/y.png", b"other")
    monkeypatch.setitem(module.REVIEWED, module.digest(b"reviewed"), "x")
    with zipfile.ZipFile(path) as book:
        node = ET.fromstring(
            '<p>Let <img class="math-inline" src="x.png"/> equal <img class="math-inline" src="y.png"/>.</p>'
        )
        content, assets, flags = module.render(node, book, "OPS/chapter.html")
    assert content == "Let x equal [UNRESOLVED IMAGE] ."
    assert assets[0]["review"] == "two_visual_reviews_same_bytes_context_unreviewed"
    assert assets[1]["transcription"] is None
    assert flags == ["unresolved_image"]
