"""Tests for conservative unbranded EPUB section extraction."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "datasets" / "mathematics" / "extract_siyavula_epub.py"
)
spec = importlib.util.spec_from_file_location("extract_siyavula_epub", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_book(path: Path, markup: str) -> None:
    with zipfile.ZipFile(path, "w") as book:
        book.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        )
        book.writestr(
            "OPS/book.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="ch" href="a.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="ch"/></spine></package>',
        )
        book.writestr("OPS/a.xhtml", markup)


def test_image_equations_are_not_silently_dropped(tmp_path: Path) -> None:
    archive = tmp_path / "book.epub"
    make_book(
        archive,
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><div><h2>Algebra</h2><p>Let <img src="equation/a.png"/> equal 2.</p></div></body></html>',
    )
    rows, inventory, source = module.extract(archive, 10)
    assert len(rows) == 1
    assert "[IMAGE: equation/a.png]" in rows[0]["original_source_text"]
    assert rows[0]["verification_status"] == "quarantined"
    assert inventory[0]["status"] == "extracted"
    assert source["sha256"]
    assert module.extract(archive, 10)[0] == rows


def test_missing_section_is_inventory_failure(tmp_path: Path) -> None:
    archive = tmp_path / "book.epub"
    make_book(archive, "<html")
    rows, inventory, _ = module.extract(archive, 11)
    assert rows == []
    assert inventory[0]["status"] == "failed"


def test_unsafe_epub_member_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.epub"
    with zipfile.ZipFile(archive, "w") as book:
        book.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="book.opf"/></rootfiles></container>',
        )
        book.writestr(
            "book.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="a" href="../escape" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="a"/></spine></package>',
        )
    with pytest.raises(ValueError, match="Unsafe"):
        module.extract(archive, 12)
