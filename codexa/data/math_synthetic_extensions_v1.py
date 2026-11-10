"""Additional held mathematics candidates. No training admission is performed."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

VERSION = "math-synthetic-extension-1"
CASES: dict[int, tuple[str, ...]] = {
    9: ("probability_single", "statistics_median", "statistics_range"),
    10: ("function_evaluate", "function_intercept", "statistics_mean"),
    11: ("probability_without_replacement", "trig_sin", "trig_cos", "trig_tan"),
    12: ("probability_without_replacement", "function_inverse", "function_quadratic"),
}


def _format_fraction(n: Fraction) -> str:
    return str(n.numerator) if n.denominator == 1 else f"{n.numerator}/{n.denominator}"


def build(seed: int, grade: int, family: str, index: int) -> dict[str, Any]:
    if grade not in CASES or family not in CASES[grade] or index < 0:
        raise ValueError("invalid grade, family or index")
    key = f"{VERSION}:{seed}:{grade}:{family}:{index}"
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
    a, b, c = (rng.randint(2, 15) for _ in range(3))
    x = sp.Symbol("x", real=True)
    answer_unit = None
    if family == "probability_single":
        question = f"A bag contains {a} red counters and {b} blue counters. One counter is chosen at random. What is the probability that it is red?"
        answer = _format_fraction(Fraction(a, a + b))
        check = sp.Rational(a, a + b)
        topic = "probability"
        assumptions = "All individual counters are equally likely to be selected."
    elif family == "probability_without_replacement":
        question = f"A bag contains {a} red counters and {b} blue counters. Two counters are chosen uniformly without replacement. What is the probability that both are red?"
        answer = _format_fraction(Fraction(a * (a - 1), (a + b) * (a + b - 1)))
        check = sp.binomial(a, 2) / sp.binomial(a + b, 2)
        topic = "probability"
        assumptions = (
            "All individual counters are equally likely; there are at least two red counters."
        )
    elif family in {"statistics_mean", "statistics_median", "statistics_range"}:
        values = [rng.randint(0, 40) for _ in range(5)]
        sequence = ", ".join(map(str, values))
        label = family.removeprefix("statistics_")
        question = f"Find the {label} of this data set: {sequence}."
        if label == "mean":
            answer = _format_fraction(Fraction(sum(values), len(values)))
            check = sum(map(sp.Integer, values)) / sp.Integer(len(values))
        elif label == "median":
            answer = str(statistics.median(values))
            check = sp.Integer(sorted(map(sp.Integer, values))[2])
        else:
            answer = str(max(values) - min(values))
            check = sp.Max(*values) - sp.Min(*values)
        topic = "statistics"
        assumptions = "All five displayed values are observations; no units are specified."
    elif family == "function_evaluate":
        question = f"Let f(x) = {a}x + {b}. Find f({c})."
        answer = str(a * c + b)
        check = (a * x + b).subs(x, c)
        topic = "functions"
        assumptions = "The function has real domain."
    elif family == "function_quadratic":
        question = f"Let f(x) = x² - {a}x + {b}. Find f({c})."
        answer = str(c * c - a * c + b)
        check = (x**2 - a * x + b).subs(x, c)
        topic = "functions"
        assumptions = "The function has real domain."
    elif family == "function_intercept":
        question = f"The straight line y = {a}x + {b} crosses the y-axis at (0, k). Find k."
        answer = str(b)
        check = (a * x + b).subs(x, 0)
        topic = "functions"
        assumptions = "The variables are real and x is the horizontal coordinate."
    elif family == "function_inverse":
        question = f"Let f(x) = {a}x + {b}, for real x. Find f⁻¹(x)."
        answer = f"(x - {b})/{a}"
        # Solve f(t) = x independently rather than expanding the constructed inverse.
        t = sp.Symbol("t", real=True)
        check = sp.solve(sp.Eq(a * t + b, x), t)[0]
        topic = "inverse functions"
        assumptions = "The domain and range are real numbers; the slope is nonzero."
    else:
        # Name opposite and adjacent sides explicitly, avoiding an implicit diagram.
        opposite, adjacent, hypotenuse = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17)])
        scale = rng.randint(1, 5)
        opposite *= scale
        adjacent *= scale
        hypotenuse *= scale
        ratio = family.removeprefix("trig_")
        question = f"A right triangle has an acute angle θ. Relative to θ, the opposite side is {opposite} cm and the adjacent side is {adjacent} cm. Find {ratio} θ exactly."
        numerator, denominator = {
            "sin": (opposite, hypotenuse),
            "cos": (adjacent, hypotenuse),
            "tan": (opposite, adjacent),
        }[ratio]
        answer = _format_fraction(Fraction(numerator, denominator))
        angle = sp.atan(sp.Rational(opposite, adjacent))
        check = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan}[ratio](angle).simplify()
        topic = "trigonometry"
        assumptions = "The triangle is right-angled; θ is acute and the named lengths are positive."
    if sp.simplify(sp.sympify(answer, locals={"x": x}) - check) != 0:
        raise ArithmeticError(f"independent answer check failed for {key}")
    return {
        "id": hashlib.sha256(key.encode()).hexdigest(),
        "generator_version": VERSION,
        "seed": seed,
        "index": index,
        "grade": grade,
        "grade_band": "8-9" if grade == 9 else "10-12",
        "topic": topic,
        "subtopic": family,
        "grade_mapping_status": "provisional_caps_review_required",
        "problem_family": family,
        "question": question,
        "answer": answer,
        "answer_unit": answer_unit,
        "explanation": None,
        "content_origin": "synthetic",
        "record_type": "generated_problem",
        "answer_verification": "passed_sympy_independent_recalculation",
        "explanation_verification": "not_applicable",
        "assumptions": assumptions,
        "training_eligibility": "held_pending_caps_and_protected_evaluation_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.per_family <= 0:
        parser.error("--per-family must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("x", encoding="utf-8") as stream:
        for grade, families in CASES.items():
            for family in families:
                for index in range(args.per_family):
                    row = build(args.seed, grade, family, index)
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                    count += 1
    print(
        json.dumps(
            {"records": count, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
