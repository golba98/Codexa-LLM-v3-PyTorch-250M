"""Regression tests for section recovery from current Siyavula HTML."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("bs4")
from bs4 import BeautifulSoup

from scripts.acquire_siyavula_senior_html import parse_online_page


class FakeLegacy:
    def parse_page(
        self, grade: int, source: dict[str, Any], page: str
    ) -> tuple[list[dict[str, Any]], None]:
        soup = BeautifulSoup(page, "html.parser")
        text = soup.select_one("#content").get_text(" ", strip=True)
        return ([{"id": "old", "page_or_section": "unit-000001", "body": text}], None)


def test_nested_sections_preserve_formula_and_distinct_identity() -> None:
    html = r'<div class="sv-book-section-view"><section class="section" id="a"><p>First \(x^2\)</p><section class="section" id="b"><p>Second \(x+1\)</p></section></section></div>'
    records, error = parse_online_page(FakeLegacy(), 10, {"source_id": "s"}, html)
    assert error is None
    assert len(records) == 2
    assert records[0]["body"] == r"First \(x^2\)"
    assert records[1]["body"] == r"Second \(x+1\)"
    assert records[0]["id"] != records[1]["id"]
    assert [x["source_section_id"] for x in records] == ["a", "b"]


def test_navigation_does_not_become_instructional_content() -> None:
    records, error = parse_online_page(
        FakeLegacy(),
        10,
        {"source_id": "s"},
        '<div class="sv-book-section-view"><a>navigation</a></div>',
    )
    assert records == []
    assert error is not None
