"""Generate held, deduplicated, independently checkable mathematics pairs.

Only a frozen family manifest is read; assessment questions are never read by
this module. Actual admission requires a separate assessment and CAPS review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

# Family IDs describe mathematical structures, not numerical parameters.
FAMILIES = {
    1: "arithmetic.missing_addend",
    2: "arithmetic.equal_groups",
    3: "arithmetic.equal_sharing",
    4: "measurement.rectangle_perimeter",
    5: "fraction.same_denominator_sum",
    6: "percentage.quantity",
    7: "ratio.scale_given_term",
    8: "algebra.substitution",
    9: "equation.linear_isolation",
    10: "equation.factored_quadratic_roots",
    11: "equation.simultaneous_sum_difference",
    12: "probability.two_without_replacement",
}
TOPICS = {
    1: "whole numbers",
    2: "whole numbers",
    3: "whole numbers",
    4: "perimeter",
    5: "common fractions",
    6: "percentages",
    7: "ratio",
    8: "algebraic expressions",
    9: "equations",
    10: "equations",
    11: "simultaneous equations",
    12: "probability",
}


def rational(n: int, d: int) -> str:
    result = Fraction(n, d)
    return str(result.numerator) if result.denominator == 1 else str(result)


def build(grade: int, params: dict[str, int], seed: int) -> dict[str, Any]:
    a, b = params["a"], params["b"]
    c = params.get("c", 0)
    unit = None
    if grade == 1:
        question = f"Fill in the missing number: {a} + □ = {a + b}."
        answer, steps = str(b), [f"{a + b} - {a} = {b}"]
    elif grade == 2:
        question = f"There are {a} bags with {b} marbles in each bag. How many marbles are there altogether?"
        answer, unit, steps = str(a * b), "marbles", [f"{a} * {b} = {a * b}"]
    elif grade == 3:
        question = f"Share {a * b} stickers equally among {b} children. How many stickers does each child receive?"
        answer, unit, steps = str(a), "stickers", [f"{a * b} / {b} = {a}"]
    elif grade == 4:
        question = f"A rectangle has length {a} cm and width {b} cm. Find its perimeter."
        answer, unit = str(2 * (a + b)), "cm"
        steps = [f"2 * ({a} + {b}) = {2 * (a + b)}"]
    elif grade == 5:
        value = rational(a + b, c)
        question = f"Calculate {a}/{c} + {b}/{c}. Give the answer in simplest form."
        answer, steps = value, [f"({a} + {b}) / {c} = {value}"]
    elif grade == 6:
        question = f"A school has {b} books. If {a}% are storybooks, how many are storybooks?"
        answer, unit = str(a * b // 100), "books"
        steps = [f"{a} / 100 * {b} = {answer}"]
    elif grade == 7:
        question = f"The ratio of red to blue beads is {a}:{b}. If there are {b * c} blue beads, how many red beads are there?"
        answer, unit = str(a * c), "red beads"
        steps = [f"{b * c} / {b} = {c}", f"{a} * {c} = {answer}"]
    elif grade == 8:
        question = f"Evaluate {a}x + {b} when x = {c}."
        answer = str(a * c + b)
        steps = [f"{a} * {c} + {b} = {answer}"]
    elif grade == 9:
        question = f"Solve for x: {a}x + {b} = {a * c + b}."
        answer = str(c)
        steps = [f"({a * c + b} - {b}) / {a} = {c}"]
    elif grade == 10:
        question = f"Find all real solutions of (x - {a})(x - {b}) = 0."
        answer = f"{a}, {b}"
        steps = [f"{a} - {a} = 0", f"{b} - {b} = 0"]
    elif grade == 11:
        question = f"Solve simultaneously: x + y = {a + b} and x - y = {a - b}. Give (x, y)."
        answer = f"({a}, {b})"
        steps = [f"({a + b} + ({a - b})) / 2 = {a}", f"{a + b} - {a} = {b}"]
    else:
        question = f"A bag contains {a} red and {b} blue counters. Two are drawn at random without replacement. Find the probability that both are red."
        answer = rational(a * (a - 1), (a + b) * (a + b - 1))
        steps = [f"{a} / {a + b} * {a - 1} / {a + b - 1} = {answer}"]
    canonical = " ".join(question.casefold().split())
    return {
        "id": hashlib.sha256(canonical.encode()).hexdigest(),
        "grade": grade,
        "topic": TOPICS[grade],
        "problem_family": FAMILIES[grade],
        "params": params,
        "question": question,
        "answer": answer,
        "answer_unit": unit,
        "solution_steps": steps,
        "seed": seed,
        "origin": "original_synthetic",
        "licence": "CC0-1.0",
        "record_type": "problem_solution_pair",
        "grade_mapping_status": "provisional_caps_review_required",
        "verification_status": "generated_unchecked",
        "admission_status": "held",
    }


def generate(seed: int, per_family: int, freeze: dict[str, Any]) -> list[dict[str, Any]]:
    if per_family < 1:
        raise ValueError("per-family must be positive")
    reserved = set(freeze["reserved_families"])
    if reserved.intersection(FAMILIES.values()):
        raise ValueError("generated family is reserved for protected evaluation")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for grade in range(1, 13):
        attempts = 0
        while sum(row["grade"] == grade for row in rows) < per_family:
            attempts += 1
            if attempts > per_family * 200:
                raise RuntimeError("unable to generate unique questions")
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c = rng.randint(2, 9)
            if grade == 1:
                a, b = rng.randint(1, 9), rng.randint(1, 10)
                if a + b > 10:
                    continue
            if grade == 5:
                c = rng.randint(4, 12)
                a, b = rng.randint(1, c - 1), rng.randint(1, c - 1)
            if grade == 6:
                a, b = rng.choice([5, 10, 20, 25, 50]), rng.randint(2, 20) * 100
            if grade == 10:
                if a == b:
                    continue
                a, b = sorted((a, b))
            params = {"a": a, "b": b, "c": c}
            row = build(grade, params, seed)
            if row["id"] not in seen:
                rows.append(row)
                seen.add(row["id"])
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-family", type=int, required=True)
    p.add_argument("--family-freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    freeze = json.loads(args.family_freeze.read_text(encoding="utf-8"))
    rows = generate(args.seed, args.per_family, freeze)
    with args.output.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "generated": len(rows),
                "family_count": len(FAMILIES),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "status": "held_pending_independent_checks_and_full_evaluation_boundary",
            }
        )
    )


if __name__ == "__main__":
    main()
