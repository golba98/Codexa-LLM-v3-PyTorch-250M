"""Explicit source-admission restrictions, versioned separately from text cleaning."""

import re
import urllib.parse
from typing import Any


def admission_text(text: Any, approval: dict[str, Any]) -> Any:
    if not isinstance(text, str) or not approval.get("strip_educational_navigation"):
        return text
    text = re.sub(r"\A\s*Courses\s*\nCourses for Kids.*?\n\s*\n", "", text, flags=re.DOTALL)
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().startswith(
            (
                "Courses Courses for Kids",
                "During AMC testing,",
                "Last updated date:",
                "Total views:",
                "Views today:",
            )
        )
    )
    return re.sub(r"\[asy\].*?\[/asy\]", "", text, flags=re.DOTALL)


def admission_error(text: str, metadata: dict[str, Any], approval: dict[str, Any]) -> str | None:
    if approval.get("reject_bare_is_opening") and re.match(r"(?i)\A\s*is\b", text):
        return "admission_missing_subject"
    if approval.get("reject_extraction_markers") and re.search(
        r"(?im)\{\[\s*promptMessage\s*\]\}|differences? between (?:the selected |revisions)|"
        r"you visited us \d+ times|save this pdf as:|\(diff\).*older revision|"
        r"if you['’]re seeing this message|"
        r"^\s*\#{0,6}\s*(?:Search by Topic|Resources tagged with|Tagged Questions)\b",
        text,
    ):
        return "admission_extraction_template"
    if approval.get("formats") and metadata.get("format") not in approval["formats"]:
        return "admission_format"
    if approval.get("require_terminal") and not re.search(r"""[.!?][\'"”’)*$]*\s*$""", text):
        return "admission_incomplete_prose"
    if approval.get("maximum_word_length") and any(
        len(word) > approval["maximum_word_length"] for word in re.findall(r"\w+", text)
    ):
        return "admission_scrambled_word"
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        return "admission_sensitive_identifier"
    domains = approval.get("url_domains")
    if domains:
        host = urllib.parse.urlsplit(metadata.get("url", "")).hostname or ""
        if not any(host == d or host.endswith("." + d) for d in domains):
            return "admission_domain"
    if len(re.findall(r"\b[A-Za-z]{2,}\b", text)) < approval.get("minimum_words", 0):
        return "admission_short_prose"
    if sum(c.isalpha() for c in text) / max(1, len(text)) < approval.get("minimum_alpha", 0):
        return "admission_nonprose"
    if approval.get("require_explanation") and not re.search(
        r"(?i)\b(?:because|therefore|hence|solution|explanation|step|means|example|thus)\b", text
    ):
        return "admission_no_explanation"
    if approval.get("reject_code") and re.search(
        r"(?im)^\s*(?:>>>|import |from \w+ import|def \w+\(|#include|public static|```(?:python|cpp|java))|\\(?:documentclass|usepackage|newcommand)",
        text,
    ):
        return "admission_code"
    if approval.get("reject_tables") and re.search(
        r"(?i)\|-|colspan\s*=|rowspan\s*=|style\s*=", text
    ):
        return "admission_table_markup"
    return None
