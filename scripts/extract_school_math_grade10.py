"""Extract one bounded development unit from a locally preserved Siyavula page."""

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

from codexa.data.school_math_pilot import FIELDS, record_id


class Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def extract(raw: Path) -> dict:
    data = raw.read_bytes()
    html = data.decode("utf-8")
    start = html.index("<p>We use the following definitions:</p>")
    end = html.index("</ul>", start) + len("</ul>")
    parser = Text()
    parser.feed(html[start:end])
    text = "\n".join(
        " ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip()
    )
    source_id = "siyavula-za-maths-grade-10-online"
    location = "1.2 The real number system, natural/whole/integer definitions"
    row = dict.fromkeys(FIELDS)
    row.update(
        id=record_id(source_id, location),
        grade=10,
        grade_band="10-12",
        source_grade="10",
        source_curriculum="South Africa CAPS",
        target_curriculum="South Africa CAPS",
        grade_mapping_status="source_labelled",
        topic="Algebraic expressions",
        subtopic="Real number system",
        record_type="explanation",
        language="en",
        source_id=source_id,
        source_url="https://www.siyavula.com/read/za/mathematics/grade-10/algebraic-expressions/01-algebraic-expressions-01",
        source_file_sha256=hashlib.sha256(data).hexdigest(),
        source_title="Mathematics Grade 10",
        source_edition="Online CAPS edition, undated",
        page_or_section=location,
        licence="CC-BY-3.0",
        attribution="Siyavula Education, Mathematics Grade 10, CC BY 3.0",
        original_source_text=text,
        explanation=text,
        diagram_dependency=False,
        content_origin="source_extracted",
        normalization_applied=["HTML text extraction", "whitespace normalization"],
        extraction_worker="coordinator",
        verification_status="pending",
        verification_methods=[],
        verification_notes="Awaiting source comparison.",
        codexa_token_count=None,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(extract(args.raw), ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
