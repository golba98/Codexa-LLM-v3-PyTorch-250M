"""Combine complete independent ratings conservatively and bind their provenance."""

import argparse
import json
from pathlib import Path
from typing import Any

from codexa.data.production import atomic, digest
from codexa.data.quality_protocol import CHECKS
from scripts.datasets.review_continuation_corpus import adjudicate


def combined(inputs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(inputs) != 2:
        raise ValueError("Exactly two independent reviews required")
    keyed = [{r["id"]: r for r in rows} for rows in inputs]
    if set(keyed[0]) != set(keyed[1]):
        raise ValueError("Independent reviewers must score identical complete packets")
    order = {"not_applicable": 0, "pass": 1, "uncertain": 2, "defect": 3}
    return [
        {
            "id": ident,
            "usable": all(rows[ident]["usable"] for rows in keyed),
            "severe": any(rows[ident]["severe"] for rows in keyed),
            "critical": any(rows[ident]["critical"] for rows in keyed),
            "coverage_complete": all(rows[ident]["coverage_complete"] for rows in keyed),
            "checks": {
                check: max((rows[ident]["checks"][check] for rows in keyed), key=order.__getitem__)
                for check in CHECKS
            },
            "reason": " | ".join(
                f"Reviewer {i + 1}: {rows[ident]['reason']}" for i, rows in enumerate(keyed)
            ),
        }
        for ident in keyed[0]
    ]


def verify_reviewers(review: Path) -> None:
    record = json.loads((review / "independent-reviewers.json").read_text())
    if record["ratings_sha256"] != digest(review / "ratings.jsonl"):
        raise ValueError("Combined review changed")
    if len(record["reviewers"]) != 2 or len({r["reviewer"] for r in record["reviewers"]}) != 2:
        raise ValueError("Two distinct independent reviewers required")
    mapping = [json.loads(x) for x in (review / "mapping.jsonl").read_text().splitlines()]
    protocol = json.loads((review / "protocol.json").read_text())
    inputs = []
    for row in record["reviewers"]:
        path = review / row["file"]
        if path.parent.resolve() != review.resolve() or digest(path) != row["sha256"]:
            raise ValueError("Independent review identity changed")
        ratings = [json.loads(x) for x in path.read_text().splitlines()]
        result = adjudicate(mapping, ratings, protocol)
        if result["unreviewed"]:
            raise ValueError("Independent review is incomplete")
        inputs.append(ratings)
    actual = [json.loads(x) for x in (review / "ratings.jsonl").read_text().splitlines()]
    if actual != combined(inputs):
        raise ValueError("Combined ratings do not match independent ratings")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review", type=Path)
    p.add_argument(
        "--reviewer", action="append", required=True, help="Distinct agent identity:file name"
    )
    args = p.parse_args()
    mapping = [json.loads(x) for x in (args.review / "mapping.jsonl").read_text().splitlines()]
    protocol = json.loads((args.review / "protocol.json").read_text())
    records, inputs = [], []
    for value in args.reviewer:
        identity, name = value.split(":", 1)
        if Path(name).name != name or not identity:
            raise ValueError("Use a local reviewer filename and explicit agent identity")
        path = args.review / name
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        result = adjudicate(mapping, rows, protocol)
        if result["unreviewed"]:
            raise ValueError("Independent review is incomplete")
        inputs.append(rows)
        records.append({"reviewer": identity, "file": name, "sha256": digest(path)})
    with (args.review / "ratings.jsonl").open("x") as out:
        for row in combined(inputs):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic(
        args.review / "independent-reviewers.json",
        {"reviewers": records, "ratings_sha256": digest(args.review / "ratings.jsonl")},
    )
    verify_reviewers(args.review)


if __name__ == "__main__":
    main()
