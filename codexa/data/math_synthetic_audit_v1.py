"""Count native tokenizer usage and exact duplicates in a held synthetic shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


def audit(path: Path, tokenizer_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    grades: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    first: dict[str, str] = {}
    problem_first: dict[tuple[str, tuple[int, ...]], str] = {}
    family_grades: dict[str, set[int]] = {}
    family_counts: Counter[str] = Counter()
    duplicates: list[dict[str, str]] = []
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = " ".join(row["question"].casefold().split())
            if key in first:
                duplicates.append({"id": row["id"], "duplicate_of": first[key]})
            else:
                first[key] = row["id"]
            family = row["problem_family"]
            # Conservative review signal: ordering and prose are intentionally discarded.
            # Equal signatures can be false positives; unequal signatures do not prove safety.
            signature = (family, tuple(sorted(int(n) for n in re.findall(r"\d+", row["question"]))))
            if signature in problem_first:
                duplicates.append(
                    {
                        "id": row["id"],
                        "related_to": problem_first[signature],
                        "reason": "same_family_and_numeric_multiset",
                    }
                )
            else:
                problem_first[signature] = row["id"]
            family_counts[family] += 1
            family_grades.setdefault(family, set()).add(row["grade"])
            count += 1
            grade = str(row["grade"])
            grades[grade] += 1
            topics[row["topic"]] += 1
            payload = row["question"] + "\n" + row["answer"]
            if row["answer_unit"]:
                payload += " " + row["answer_unit"]
            tokens[grade] += len(tokenizer.encode(payload, add_special_tokens=False).ids)
    return {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "token_count_definition": "question + newline + answer + optional space and answer_unit; no special tokens",
        "records": count,
        "records_per_grade": dict(sorted(grades.items(), key=lambda p: int(p[0]))),
        "records_per_topic": dict(sorted(topics.items())),
        "tokens_per_grade": dict(sorted(tokens.items(), key=lambda p: int(p[0]))),
        "tokens_total": sum(tokens.values()),
        "exact_duplicate_questions": sum("duplicate_of" in item for item in duplicates),
        "related_numeric_signatures": sum("related_to" in item for item in duplicates),
        "family_counts": dict(sorted(family_counts.items())),
        "family_grades": {
            family: sorted(grades) for family, grades in sorted(family_grades.items())
        },
        "structural_diversity_limitation": "Mostly numerical and wording variants; no protected-family partition or near-duplicate clearance has been performed.",
        "admission_status": "all held: CAPS mapping, near-duplicate and protected evaluation review pending",
    }, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--duplicates", required=True, type=Path)
    args = parser.parse_args()
    report, duplicates = audit(args.input, args.tokenizer)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with args.duplicates.open("x", encoding="utf-8") as stream:
        for row in duplicates:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
