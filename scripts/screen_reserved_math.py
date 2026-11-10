"""Screen pre-existing JSONL shards against a frozen confidential maths holdout.

Only identifiers and reason codes are emitted. Unknown, unlabelled and unmatched
records are NOT cleared: lexical/family screens cannot establish disjointness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[\w]+|[+*/=−-]", text.casefold(), re.UNICODE))


def shape(text: str) -> str:
    return re.sub(r"\b\d+(?:\.\d+)?\b", "#", normalise(text))


def scan(protected: Path, files: list[Path], output: Path) -> dict[str, Any]:
    pool = protected / "private-assessment.jsonl"
    exclusions = protected / "family-exclusions.json"
    manifest = json.loads((protected / "freeze-manifest.json").read_text())
    if (
        digest(pool) != manifest["private_assessment_sha256"]
        or digest(exclusions) != manifest["family_exclusions_sha256"]
    ):
        raise ValueError("protected set hash mismatch")
    answers = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines() if line]
    for item in answers:
        if not item.get("question"):
            raise ValueError("protected question missing")
    expected = [(normalise(row["question"]), shape(row["question"])) for row in answers]
    families = set(json.loads(exclusions.read_text())["forbidden_generator_families"])
    counts: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=False)
    with (output / "flags.jsonl").open("x", encoding="utf-8") as flags:
        for path in files:
            sha_before = digest(path)
            rows = 0
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    family = row.get("problem_family")
                    reasons = []
                    if family in families:
                        reasons.append("excluded_generator_family")
                    # Only exact matches to the frozen family vocabulary are mapped.
                    # A non-empty new taxonomy is unknown, not evidence of separation.
                    if not family or family not in families:
                        counts["unmapped_family_rows"] += 1
                        reasons.append("unknown_or_unmapped_family")
                    content = str(
                        row.get("question")
                        or row.get("original_source_text")
                        or row.get("text")
                        or ""
                    )
                    normalized = normalise(content)
                    if any(q and q in normalized for q, _ in expected):
                        reasons.append("exact_question_text")
                    # Local segments only; adjacent or paraphrased questions can be missed.
                    segments = re.split(r"(?<=[.!?])\s+|\n+", content)
                    for segment in segments:
                        if not 18 <= len(segment) <= 400:
                            continue
                        shape_text = shape(segment)
                        if any(
                            len(target) >= 35
                            and 2
                            * min(len(shape_text), len(target))
                            / (len(shape_text) + len(target))
                            >= 0.88
                            and SequenceMatcher(
                                None, shape_text, target, autojunk=False
                            ).quick_ratio()
                            >= 0.88
                            and SequenceMatcher(None, shape_text, target, autojunk=False).ratio()
                            >= 0.88
                            for _, target in expected
                        ):
                            reasons.append("possible_template_overlap")
                            break
                    if reasons:
                        counts["flagged_rows"] += 1
                        counts.update(reasons)
                        flags.write(
                            json.dumps(
                                {
                                    "input": str(path),
                                    "line": line_no,
                                    "id": row.get("id"),
                                    "reasons": reasons,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            sha_after = digest(path)
            if sha_before != sha_after:
                raise RuntimeError(f"input changed during scan: {path}")
            counts["records_scanned"] += rows
            receipts.append({"input": str(path), "sha256": sha_after, "records": rows})
    report = {
        "counts": counts,
        "inputs": receipts,
        "protected_manifest_sha256": digest(protected / "freeze-manifest.json"),
        "status": "flagging only; no record is approved by this screen",
        "limits": [
            "Family exclusions apply only to correctly labelled synthetic records.",
            "Text and template checks are heuristic; math equivalence, rewording, long multi-part questions, images, translations and all related families are not covered.",
            "Unlabelled source records remain held; zero matches do not establish no prior exposure.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("inputs", type=Path, nargs="+")
    args = p.parse_args()
    report = scan(args.protected, args.inputs, args.output)
    print(json.dumps({"counts": report["counts"], "status": report["status"]}))


if __name__ == "__main__":
    main()
