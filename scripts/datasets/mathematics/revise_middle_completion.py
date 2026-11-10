"""Preserve and correct independently identified middle-grade answer defects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CORRECTIONS: dict[str, tuple[str, str]] = {
    "middle-original-837b117ec734e0b227e2d131": (
        "3000 + 700 + 40 + 5",
        "The requested answer is the expanded form.",
    ),
    "middle-original-6a11ac80b2a039cfd4fb6982": (
        "5000 + 10 + 9",
        "The requested answer is the expanded form.",
    ),
    "middle-original-a022cb712d73ac3bb6f2209a": (
        "600000 + 400 + 5",
        "The requested answer is the expanded form.",
    ),
    "middle-original-20e99315e45f8685c34cf0cb": (
        "200000 + 70000 + 4000 + 10 + 8",
        "The requested answer is the expanded form.",
    ),
    "middle-original-76e89f8058f671772cfa0a1e": (
        "3478",
        "The difference is 3478; adding 2659 recovers 6137.",
    ),
}


def revise(record: dict[str, Any]) -> dict[str, Any]:
    revised = dict(record)
    ident = str(record["record_id"])
    if ident in CORRECTIONS:
        answer, reason = CORRECTIONS[ident]
        revised["previous_record_id"] = ident
        revised["previous_answer"] = record["answer"]
        revised["answer"] = answer
        revised["correction_reason"] = reason
        revised["record_id"] = ident + "-answer-v2"
        if ident == "middle-original-76e89f8058f671772cfa0a1e":
            revised["solution"] = "6137 - 2659 = 3478; 3478 + 2659 = 6137."
    if record.get("record_type") == "worked_example":
        revised["independent_question_answer_review"] = (
            "independent model review; protected overlap, grade-scope and full admission remain pending"
        )
    return revised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.input.read_bytes()
    records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    ids = {str(r["record_id"]) for r in records}
    if not CORRECTIONS.keys() <= ids:
        raise ValueError("Input is not the expected complete draft")
    if args.output.exists():
        raise FileExistsError("Preserve the prior revision; choose a fresh output")
    args.output.mkdir(parents=True)
    updated = [revise(r) for r in records]
    content = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in updated
    ).encode()
    (args.output / "records.jsonl").write_bytes(content)
    evidence = {
        "method": "Independent model inspection of all 36 worked questions and answers, comparison against the separately computed SymPy equalities; five answer fields corrected. This does not establish grade mapping, rights, overlap, instructional completeness or admission.",
        "questions_reviewed": 36,
        "corrected_answer_fields": list(CORRECTIONS),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "reviewer": "coordinator independent of original generator worker",
        "status": "held",
    }
    (args.output / "review.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
