"""Map recovery-generator family labels to frozen broad exclusions; never release records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ALIASES: dict[str, str | None] = {
    "arithmetic.missing_addend": "addition",
    "arithmetic.equal_groups": "multiplication",
    "arithmetic.equal_sharing": "division",
    "measurement.rectangle_perimeter": "rectangle_perimeter",
    "fraction.same_denominator_sum": "fraction_add",
    "percentage.quantity": "percent",
    "ratio.scale_given_term": "ratio",
    "algebra.substitution": None,
    "equation.linear_isolation": "linear",
    "equation.factored_quadratic_roots": "quadratic_roots",
    "equation.simultaneous_sum_difference": "simultaneous",
    "probability.two_without_replacement": "probability_without_replacement",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--exclusions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    freeze = args.exclusions.parent / "freeze-manifest.json"
    frozen = json.loads(freeze.read_text())
    exclusions_hash = sha(args.exclusions)
    if exclusions_hash != frozen["family_exclusions_sha256"]:
        raise ValueError("frozen family exclusions hash mismatch")
    excluded = set(json.loads(args.exclusions.read_text())["forbidden_generator_families"])
    source_hash = sha(args.input)
    results = []
    family_counts: dict[str, Counter[str]] = {}
    for line in args.input.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        family = row.get("problem_family")
        broad = ALIASES.get(family)
        reason = "excluded_broad_family" if broad in excluded else "ambiguous_or_unmapped_family"
        results.append(
            {
                "id": row["id"],
                "family": family,
                "broad_family": broad,
                "disposition": "held",
                "reason": reason,
            }
        )
        family_counts.setdefault(str(family), Counter())[reason] += 1
    if sha(args.input) != source_hash or sha(args.exclusions) != exclusions_hash:
        raise RuntimeError("input or exclusions changed during mapping")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "family-flags.jsonl").open("x", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input_sha256": source_hash,
        "exclusions_sha256": sha(args.exclusions),
        "family_aliases": ALIASES,
        "family_counts": family_counts,
        "records": len(results),
        "all_held": True,
        "limitation": "Aliases are broad conservative screening labels, not evidence of exact protected overlap or release eligibility. Unmapped families remain held.",
    }
    (args.output / "mapping-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"records": len(results), "dispositions": Counter(x["reason"] for x in results)})
    )


if __name__ == "__main__":
    main()
