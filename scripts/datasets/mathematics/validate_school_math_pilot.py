"""Merge a school-math pilot conservatively, recording all validation findings."""

import argparse
import hashlib
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from codexa.data.school_math_pilot import validate_record
from codexa.evaluation.protection import QuestionProtection
from codexa.tokenizer.bpe import CodexaTokenizer


def lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--protected", type=Path, required=True)
    p.add_argument("--context", type=int, default=2048)
    args = p.parse_args()
    tokenizer = CodexaTokenizer.load(args.tokenizer)
    protection = QuestionProtection(args.protected)
    rows: list[dict[str, Any]] = []
    for f in sorted((args.root / "workers").glob("grades-*.jsonl")):
        rows.extend(lines(f))
    reviews = {r["id"]: r for r in lines(args.root / "reviews.jsonl")}
    sources = {
        (r.get("source_id"), r.get("source_file_sha256")): r
        for r in lines(args.root / "sources.jsonl")
    }
    hashes = {
        hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (args.root / "raw").rglob("*")
        if f.is_file()
    }
    seen: dict[str, str] = {}
    normalized: list[tuple[str, str]] = []
    results: list[dict[str, Any]] = []
    for row in rows:
        ident = str(row.get("id"))
        errors = validate_record(row)
        content = str(row.get("original_source_text") or "")
        canonical = " ".join(content.casefold().split())
        if ident in seen:
            errors.append("duplicate ID")
        seen[ident] = canonical
        for previous_id, previous in normalized:
            if canonical and (
                canonical == previous
                or (
                    len(canonical) >= 80
                    and len(previous) >= 80
                    and SequenceMatcher(None, canonical, previous).ratio() >= 0.9
                )
            ):
                errors.append(f"duplicate or near duplicate of {previous_id}")
                break
        normalized.append((ident, canonical))
        if row.get("source_file_sha256") not in hashes:
            errors.append("source file hash absent from raw evidence")
        source = sources.get((row.get("source_id"), row.get("source_file_sha256")))
        if not source or source.get("eligibility") != "eligible":
            errors.append("source rights not established for training eligibility")
        match = protection.matches(content)
        if match:
            errors.append(f"protected question overlap: {match}")
        if row.get("diagram_dependency") is not False:
            errors.append("unresolved visual dependency")
        kind = row.get("record_type")
        if kind == "explanation" and not row.get("explanation"):
            errors.append("missing explanation")
        if kind in ("exercise_answer", "exercise_solution") and (
            not row.get("question")
            or not row.get("source_exercise_id")
            or not (row.get("source_solution") or row.get("final_answer"))
        ):
            errors.append("question or answer reference missing")
        if kind == "worked_example" and (not row.get("question") or not row.get("source_solution")):
            errors.append("worked example incomplete")
        row["codexa_token_count"] = len(tokenizer.encode(content))
        if row["codexa_token_count"] + 2 > args.context:
            errors.append("over context with BOS/EOS")
        review = reviews.get(ident)
        if (
            review
            and review.get("status") == "verified"
            and review.get("methods")
            and review.get("notes")
            and not errors
        ):
            row["verification_status"] = "verified"
            row["verification_methods"] = review["methods"]
            row["verification_notes"] = review["notes"]
        elif errors or (review and review.get("status") == "quarantined"):
            row["verification_status"] = "quarantined"
            row["verification_notes"] = "; ".join(errors) or str(
                review.get("notes") if review else ""
            )
        else:
            row["verification_status"] = "pending"
        results.append({"id": ident, "errors": errors, "status": row["verification_status"]})
    write(args.root / "records.jsonl", rows)
    write(
        args.root / "verified.jsonl",
        [
            r
            for r in rows
            if r["verification_status"] == "verified" and r["record_type"] != "practice_only"
        ],
    )
    write(
        args.root / "practice-only.jsonl", [r for r in rows if r["record_type"] == "practice_only"]
    )
    write(
        args.root / "quarantine.jsonl",
        [r for r in rows if r["verification_status"] == "quarantined"],
    )
    coverage = {
        str(g): dict(Counter(r["verification_status"] for r in rows if r.get("grade") == g))
        for g in range(1, 13)
    }
    (args.root / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (args.root / "validation-results.json").write_text(
        json.dumps(
            {
                "records": results,
                "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                "context": args.context,
                "limitations": [
                    "Approximate duplicate screening and protected-question heuristics cannot rule out paraphrases or all short overlaps.",
                    "Language mixing, mathematical fidelity and source answer pairing require source review.",
                    "Verified status is not a source-rights or training approval.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"records": len(rows), "coverage": coverage}))


if __name__ == "__main__":
    main()
