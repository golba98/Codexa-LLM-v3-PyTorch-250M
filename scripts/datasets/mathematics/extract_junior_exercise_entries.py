"""Split saved Siyavula exercise sections into source-linked entry candidates.

The HTML does not supply verified question/answer correctness. Entries remain
held, and interactive placeholders, images and context needs are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
VISUAL = {
    "img",
    "svg",
    "math",
    "figure",
    "table",
    "canvas",
    "iframe",
    "video",
    "audio",
    "object",
    "embed",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Node:
    def __init__(self, tag: str, attrs: dict[str, str | None], start: int) -> None:
        self.tag = tag
        self.attrs = attrs
        self.start = start
        self.end: int | None = None
        self.children: list[Node | str] = []

    def has(self, name: str) -> bool:
        return name in str(self.attrs.get("class") or "").split()

    def walk(self) -> list[Node]:
        result = [self]
        for child in self.children:
            if isinstance(child, Node):
                result.extend(child.walk())
        return result


class Tree(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_starts = [0]
        self.line_starts.extend(m.end() for m in re.finditer("\n", source))
        self.root = Node("document", {}, 0)
        self.stack = [self.root]
        self.feed(source)

    def position_offset(self) -> int:
        line, col = self.getpos()
        return self.line_starts[line - 1] + col

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, dict(attrs), self.position_offset())
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)
        else:
            node.end = self.position_offset() + len(self.get_starttag_text() or "")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, dict(attrs), self.position_offset())
        node.end = self.position_offset() + len(self.get_starttag_text() or "")
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        pos = self.position_offset()
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                end = self.source.find(">", pos)
                self.stack[i].end = end + 1 if end >= 0 else pos
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def text(node: Node) -> str:
    """Preserve source text and materialize ordered list positions."""
    parts: list[str] = []

    def visit(n: Node) -> None:
        if n.tag in {"script", "style"}:
            return
        if n.tag in {"p", "div", "section", "ol", "ul", "li", "br", "tr"}:
            parts.append("\n")
        if n.tag == "ol":
            number = (
                int(n.attrs.get("start") or 1) if str(n.attrs.get("start") or "1").isdigit() else 1
            )
            for child in n.children:
                if isinstance(child, Node) and child.tag == "li":
                    value = child.attrs.get("value")
                    if value is not None and str(value).isdigit():
                        number = int(value)
                    parts.append(f"\n{number}. ")
                    visit(child)
                    number += 1
                elif isinstance(child, Node):
                    visit(child)
                else:
                    parts.append(child)
            return
        for child in n.children:
            if isinstance(child, Node):
                visit(child)
            else:
                parts.append(child)

    visit(node)
    return "\n".join(
        re.sub(r"[\t\r ]+", " ", line).strip()
        for line in "".join(parts).splitlines()
        if line.strip()
    )


def extract(source: dict[str, Any], data: bytes) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw = data.decode("utf-8")
    tree = Tree(raw)
    contents = [
        node for node in tree.root.walk() if node.attrs.get("id") in {"content", "contents"}
    ]
    if len(contents) != 1:
        return [], {"content_missing_or_ambiguous": 1}
    sections = [n for n in contents[0].walk() if n.tag == "section" and n.has("exercises")]
    rows = []
    stats: Counter[str] = Counter()
    for section_no, section in enumerate(sections, 1):
        if section.has("interactive_questions"):
            stats["interactive_placeholder"] += 1
            continue
        entry_nodes = [n for n in section.walk() if n.has("entry")]
        if not entry_nodes:
            stats["static_no_entries"] += 1
        for ordinal, entry in enumerate(entry_nodes, 1):
            problems = [n for n in entry.children if isinstance(n, Node) and n.has("problem")]
            solutions = [n for n in entry.children if isinstance(n, Node) and n.has("solution")]
            if len(problems) != 1 or len(solutions) > 1 or entry.end is None:
                stats["ambiguous_entry"] += 1
                continue
            q = text(problems[0])
            solution = text(solutions[0]) if solutions else None
            if not q:
                stats["empty_problem"] += 1
                continue
            section_id = str(section.attrs.get("id") or f"section-{section_no}")
            loc = f"{section_id}/entry-{ordinal}"
            ident = digest((source["source_id"] + "\0" + loc).encode())
            fragment = raw[entry.start : entry.end]
            visual = any(n.tag in VISUAL for n in entry.walk()) or bool(
                re.search(
                    r"\b(figure|diagram|graph|table|image|above|below|shown)\b", q, re.IGNORECASE
                )
            )
            rows.append(
                {
                    "id": ident,
                    "grade": source["grade"],
                    "grade_band": "8-9",
                    "record_type": "exercise_entry_candidate",
                    "question": q,
                    "source_solution": solution,
                    "answer": None,
                    "source_id": source["source_id"],
                    "source_url": source["source_url"],
                    "source_file_sha256": source["sha256"],
                    "source_raw_file": source["raw_file"],
                    "source_location": loc,
                    "source_section_id": section_id,
                    "source_entry_html": fragment,
                    "licence": source.get("licence"),
                    "attribution": source.get("attribution"),
                    "source_association": "problem and solution are siblings of the same source entry; association unreviewed",
                    "visual_dependency": visual,
                    "external_context": "pending section-level and question numbering review",
                    "source_fidelity": "pending",
                    "math_answer_verification": "pending" if solution else "not_applicable",
                    "solution_steps_verification": "pending" if solution else "not_applicable",
                    "rights": "unresolved",
                    "curriculum_mapping": "unresolved",
                    "overlap": "not_screened",
                    "final_admission": "not_requested",
                }
            )
            stats["extracted_entries"] += 1
    return rows, dict(stats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [
        json.loads(x) for x in (args.source / "sources.jsonl").read_text().splitlines() if x.strip()
    ]
    inventory = [
        json.loads(x)
        for x in (args.source / "inventory.jsonl").read_text().splitlines()
        if x.strip()
    ]
    status = {r["source_id"]: r["status"] for r in inventory if r.get("source_id")}
    rows = []
    pages = []
    all_stats: Counter[str] = Counter()
    for source in sources:
        if not source["source_id"].startswith("siyavula-grade-8-"):
            continue
        if status.get(source["source_id"]) not in {"parsed", "unparsed"}:
            pages.append(
                {
                    "source_id": source["source_id"],
                    "status": "not_processed_due_to_rights_or_other_status",
                }
            )
            continue
        path = args.source / source["raw_file"]
        data = path.read_bytes()
        if digest(data) != source["sha256"]:
            raise ValueError("source hash mismatch: " + str(path))
        source = {**source, "grade": 8, "raw_file": str(path)}
        found, stats = extract(source, data)
        rows.extend(found)
        all_stats.update(stats)
        pages.append(
            {
                "source_id": source["source_id"],
                "url": source["source_url"],
                "sha256": source["sha256"],
                "counts": stats,
            }
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("ID collision, no data written")
    args.output.mkdir(parents=True)
    for name, values in [("records.jsonl", rows), ("pages.jsonl", pages)]:
        (args.output / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in values),
            encoding="utf-8",
        )
    report = {
        "records": len(rows),
        "pages_considered": len(pages),
        "counts": all_stats,
        "with_source_solution": sum(bool(r["source_solution"]) for r in rows),
        "visual_flagged": sum(r["visual_dependency"] for r in rows),
        "status": "all held: fidelity, context, rights, math, CAPS and assessment screen pending",
        "limitations": [
            "Extracts source entry groups, not necessarily one exercise each; numbering and shared instructions require review.",
            "Source solution siblings are not independently verified. No official answers are invented.",
            "Interactive placeholders do not contain questions; no dynamic endpoint was accessed.",
            "An unflagged visual dependency is not evidence that an entry is self-contained.",
        ],
        "inputs": {
            name: digest((args.source / name).read_bytes())
            for name in ("sources.jsonl", "inventory.jsonl")
        },
        "outputs": {
            name: digest((args.output / name).read_bytes())
            for name in ("records.jsonl", "pages.jsonl")
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
