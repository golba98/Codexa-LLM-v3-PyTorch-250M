"""Reproducible, held-for-review mathematics candidates; never a training admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

VERSION = "math-synthetic-1"
# Grade mappings are provisional until compared with the official CAPS subtopics.
FAMILIES: dict[int, tuple[str, ...]] = {
    1: ("addition", "subtraction"),
    2: ("addition", "subtraction", "multiplication"),
    3: ("addition", "subtraction", "multiplication", "division"),
    4: ("multiplication", "division", "fraction_add"),
    5: ("fraction_add", "fraction_multiply", "division"),
    6: ("fraction_add", "fraction_multiply", "percent"),
    7: ("percent", "ratio", "linear"),
    8: ("linear", "ratio", "fraction_multiply"),
    9: ("linear", "simultaneous", "gradient"),
    10: ("quadratic_roots", "simultaneous", "gradient"),
    11: ("quadratic_roots", "gradient", "polynomial"),
    12: ("quadratic_roots", "polynomial", "derivative"),
}


def _fraction(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def build(seed: int, grade: int, family: str, index: int) -> dict[str, Any]:
    """Generate one question from its independently reproducible coordinates."""
    if grade not in FAMILIES or family not in FAMILIES[grade] or index < 0:
        raise ValueError("invalid grade, family or index")
    key = f"{VERSION}:{seed}:{grade}:{family}:{index}"
    r = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
    a, b, c, d = (r.randint(2, 20) for _ in range(4))
    variant = r.randrange(3)
    answer_unit = None
    x = sp.Symbol("x")
    if family == "addition":
        if grade == 1:
            a, b = r.randint(1, 9), r.randint(1, 10)
        q = (f"Calculate {a} + {b}.", f"Add {a} and {b}.", f"What is the sum of {a} and {b}?")[
            variant
        ]
        ans = str(a + b)
        reference = sp.Integer(a) + sp.Integer(b)
        topic = "whole numbers"
    elif family == "subtraction":
        if grade == 1:
            a, b = r.randint(1, 9), r.randint(1, 10)
        a, b = max(a, b), min(a, b)
        q = (
            f"Calculate {a} - {b}.",
            f"Subtract {b} from {a}.",
            f"What is the difference between {a} and {b}?",
        )[variant]
        ans = str(a - b)
        reference = sp.Integer(a) - sp.Integer(b)
        topic = "whole numbers"
    elif family == "multiplication":
        if grade <= 3:
            a, b = r.randint(1, 5), r.randint(1, 10)
        q = (
            f"Calculate {a} × {b}.",
            f"Multiply {a} by {b}.",
            f"What is the product of {a} and {b}?",
        )[variant]
        ans = str(a * b)
        reference = sp.Mul(sp.Integer(a), sp.Integer(b))
        topic = "whole numbers"
    elif family == "division":
        q = (
            f"Calculate {a * b} ÷ {b}.",
            f"Share {a * b} items equally among {b} groups. How many items are in each group?",
            f"How many groups of {b} are in {a * b}?",
        )[variant]
        answer_unit = "items" if variant == 1 else "groups" if variant == 2 else None
        ans = str(a)
        reference = sp.Rational(a * b, b)
        topic = "whole numbers"
    elif family == "fraction_add":
        # Equal denominators avoid assigning an unsupported skill to a younger grade.
        q = (
            f"Calculate {a}/{c} + {b}/{c}. Give a simplified fraction.",
            f"Find the sum of {a}/{c} and {b}/{c}. Simplify your answer.",
            f"Add {a}/{c} to {b}/{c}, and simplify.",
        )[variant]
        ans = _fraction(Fraction(a, c) + Fraction(b, c))
        reference = sp.Rational(a, c) + sp.Rational(b, c)
        topic = "common fractions"
    elif family == "fraction_multiply":
        q = (
            f"Calculate {a}/{c} × {b}/{d}. Give a simplified fraction.",
            f"Find the product of {a}/{c} and {b}/{d}. Simplify your answer.",
            f"Multiply {a}/{c} by {b}/{d}, and simplify.",
        )[variant]
        ans = _fraction(Fraction(a, c) * Fraction(b, d))
        reference = sp.Rational(a, c) * sp.Rational(b, d)
        topic = "common fractions"
    elif family == "percent":
        a, b = r.randint(1, 50), r.randint(1, 50) * 100
        q = (f"Calculate {a}% of {b}.", f"Find {a} percent of {b}.", f"What is {a}% of {b}?")[
            variant
        ]
        ans = str(a * b // 100)
        reference = sp.Rational(a, 100) * b
        topic = "percentages"
    elif family == "ratio":
        q = (
            f"A recipe uses {a} cups of flour per {b} cups of water. How many cups of flour are needed for {b * c} cups of water at the same ratio?",
            f"The ratio of red to blue counters is {a}:{b}. If there are {b * c} blue counters, how many red counters are there?",
            f"A mixture uses {a} litres of A for every {b} litres of B. How many litres of A are needed for {b * c} litres of B?",
        )[variant]
        answer_unit = ("cups of flour", "red counters", "litres of A")[variant]
        ans = str(a * c)
        reference = sp.Rational(a, b) * (b * c)
        topic = "ratio and rate"
    elif family == "linear":
        # The construction and verification use different operations.
        q = (
            f"Solve for x: {a}x + {b} = {a * c + b}.",
            f"Find x if {a}x + {b} equals {a * c + b}.",
            f"What value of x satisfies {a}x + {b} = {a * c + b}?",
        )[variant]
        ans = str(c)
        reference = sp.solve(sp.Eq(a * x + b, a * c + b), x)[0]
        topic = "equations"
    elif family == "simultaneous":
        y = sp.Symbol("y")
        q = (
            f"Solve simultaneously: x + y = {a + b} and x - y = {a - b}.",
            f"Find x and y if x + y = {a + b} and x - y = {a - b}.",
            f"Determine the ordered pair (x, y) satisfying x + y = {a + b} and x - y = {a - b}.",
        )[variant]
        ans = f"({a}, {b})"
        solution = sp.solve((x + y - (a + b), x - y - (a - b)), (x, y))
        reference = (solution[x], solution[y])
        topic = "simultaneous equations"
    elif family == "gradient":
        q = (
            f"Find the gradient of the line through (0, {a}) and ({b}, {a + b * c}).",
            f"Calculate the slope of the line joining (0, {a}) to ({b}, {a + b * c}).",
            f"What is the gradient between the points (0, {a}) and ({b}, {a + b * c})?",
        )[variant]
        ans = str(c)
        reference = sp.Rational((a + b * c) - a, b)
        topic = "analytical geometry"
    elif family == "quadratic_roots":
        a, b = sorted((a, b))
        q = (
            f"Find all real roots of (x - {a})(x - {b}) = 0.",
            f"Solve (x - {a})(x - {b}) = 0 over the reals.",
            f"Which real values of x satisfy (x - {a})(x - {b}) = 0?",
        )[variant]
        roots = sorted({a, b})
        ans = ", ".join(map(str, roots))
        reference = sorted(sp.solve((x - a) * (x - b), x))
        topic = "quadratic equations"
    elif family == "polynomial":
        q = (
            f"Expand and simplify (x + {a})(x + {b}).",
            f"Write (x + {a})(x + {b}) in expanded form.",
            f"Find the expanded polynomial for (x + {a})(x + {b}).",
        )[variant]
        ans = f"x**2 + {a + b}*x + {a * b}"
        reference = sp.expand((x + a) * (x + b))
        topic = "algebraic expressions"
    else:
        q = (
            f"Differentiate {a}x² + {b}x + {c} with respect to x.",
            f"Find the derivative of f(x) = {a}x² + {b}x + {c}.",
            f"If f(x) = {a}x² + {b}x + {c}, find f'(x).",
        )[variant]
        ans = f"{2 * a}*x + {b}"
        reference = sp.diff(a * x**2 + b * x + c, x)
        topic = "differential calculus"
    if family == "simultaneous":
        valid = reference == (a, b)
    elif family == "quadratic_roots":
        valid = reference == roots
    else:
        valid = sp.simplify(sp.sympify(ans, locals={"x": x}) - reference) == 0
    if not valid:
        raise ArithmeticError(f"independent verification failed: {key}")
    return {
        "id": hashlib.sha256(key.encode()).hexdigest(),
        "generator_version": VERSION,
        "seed": seed,
        "index": index,
        "grade": grade,
        "grade_band": "1-3"
        if grade <= 3
        else "4-7"
        if grade <= 7
        else "8-9"
        if grade <= 9
        else "10-12",
        "topic": topic,
        "subtopic": family,
        "grade_mapping_status": "provisional_caps_review_required",
        "problem_family": family,
        "question": q,
        "answer": ans,
        "answer_unit": answer_unit,
        "explanation": None,
        "content_origin": "synthetic",
        "record_type": "generated_problem",
        "answer_verification": "passed_sympy_independent_recalculation",
        "explanation_verification": "not_applicable",
        "assumptions": "real numbers"
        if family in {"linear", "simultaneous", "quadratic_roots", "polynomial", "derivative"}
        else "positive integer quantities and nonzero denominators",
        "training_eligibility": "held_pending_caps_and_protected_evaluation_review",
    }


def generate(seed: int, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        build(seed, grade, family, i)
        for grade, families in FAMILIES.items()
        for family in families
        for i in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output JSONL path; never overwritten"
    )
    args = parser.parse_args()
    rows = generate(args.seed, args.per_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "records": len(rows),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "status": "all held pending review",
            }
        )
    )


if __name__ == "__main__":
    main()
