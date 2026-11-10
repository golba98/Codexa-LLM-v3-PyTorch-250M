"""Independent answer/step verification from the displayed synthetic question.

Checks are necessary, not sufficient for curriculum approval or contamination
clearance. A successful calculation never marks final training admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

PATTERNS = {
    1: r"Fill in the missing number: (\d+) \+ □ = (\d+)\.",
    2: r"There are (\d+) bags with (\d+) marbles in each bag\. How many marbles are there altogether\?",
    3: r"Share (\d+) stickers equally among (\d+) children\. How many stickers does each child receive\?",
    4: r"A rectangle has length (\d+) cm and width (\d+) cm\. Find its perimeter\.",
    5: r"Calculate (\d+)/(\d+) \+ (\d+)/(\d+)\. Give the answer in simplest form\.",
    6: r"A school has (\d+) books\. If (\d+)% are storybooks, how many are storybooks\?",
    7: r"The ratio of red to blue beads is (\d+):(\d+)\. If there are (\d+) blue beads, how many red beads are there\?",
    8: r"Evaluate (\d+)x \+ (\d+) when x = (\d+)\.",
    9: r"Solve for x: (\d+)x \+ (\d+) = (\d+)\.",
    10: r"Find all real solutions of \(x - (\d+)\)\(x - (\d+)\) = 0\.",
    11: r"Solve simultaneously: x \+ y = (\d+) and x - y = (-?\d+)\. Give \(x, y\)\.",
    12: r"A bag contains (\d+) red and (\d+) blue counters\. Two are drawn at random without replacement\. Find the probability that both are red\.",
}
UNITS = {2: "marbles", 3: "stickers", 4: "cm", 6: "books", 7: "red beads"}


def verify(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    grade = row.get("grade")
    if not isinstance(grade, int):
        return ["invalid grade"]
    match = re.fullmatch(PATTERNS.get(grade, r"(?!)"), str(row.get("question", "")))
    if not match:
        return ["question does not match independently parsed structure"]
    nums = [int(n) for n in match.groups()]
    x, y = sp.symbols("x y", real=True)
    answer: Any
    if grade == 1:
        answer = nums[1] - nums[0]
    elif grade == 2:
        answer = nums[0] * nums[1]
    elif grade == 3:
        answer = sp.Rational(nums[0], nums[1])
    elif grade == 4:
        answer = 2 * nums[0] + 2 * nums[1]
    elif grade == 5:
        answer = sp.Rational(nums[0], nums[1]) + sp.Rational(nums[2], nums[3])
    elif grade == 6:
        answer = sp.Rational(nums[0] * nums[1], 100)
    elif grade == 7:
        answer = sp.Rational(nums[0] * nums[2], nums[1])
    elif grade == 8:
        answer = (nums[0] * x + nums[1]).subs(x, nums[2])
    elif grade == 9:
        answer = sp.solve(sp.Eq(nums[0] * x + nums[1], nums[2]), x)[0]
    elif grade == 10:
        answer = sp.solve((x - nums[0]) * (x - nums[1]), x)
    elif grade == 11:
        solution = sp.solve((x + y - nums[0], x - y - nums[1]), (x, y))
        answer = (solution[x], solution[y])
    else:
        answer = sp.binomial(nums[0], 2) / sp.binomial(sum(nums), 2)
    try:
        given = row.get("answer", "")
        if grade == 10:
            valid = [sp.Rational(part.strip()) for part in given.split(",")] == answer
        elif grade == 11:
            pairs = re.fullmatch(r"\((-?\d+),\s*(-?\d+)\)", given)
            valid = pairs is not None and tuple(int(v) for v in pairs.groups()) == answer
        else:
            valid = sp.Rational(given) == answer
            if grade == 5:
                valid = valid and str(Fraction(given)) == given
        if not valid:
            errors.append("answer mismatch")
    except (ValueError, TypeError, ZeroDivisionError):
        errors.append("invalid answer")
    if row.get("answer_unit") != UNITS.get(grade):
        errors.append("answer unit mismatch")
    expected_step_values = (
        [sp.Rational(nums[2], nums[1]), answer]
        if grade == 7
        else list(answer)
        if grade == 11
        else [0, 0]
        if grade == 10
        else [answer]
    )
    # These forms tie each displayed arithmetic step back to parsed quantities.
    expected_lhs = {
        1: [f"{nums[-1]} - {nums[0]}"],
        2: [f"{nums[0]} * {nums[1]}"],
        3: [f"{nums[0]} / {nums[1]}"],
        4: [f"2 * ({nums[0]} + {nums[1]})"],
        5: [f"({nums[0]} + {nums[2]}) / {nums[1]}"] if grade == 5 else [],
        6: [f"{nums[1]} / 100 * {nums[0]}"],
        7: [f"{nums[2]} / {nums[1]}", f"{nums[0]} * {sp.Rational(nums[2], nums[1])}"]
        if grade == 7
        else [],
        8: [f"{nums[0]} * {nums[2]} + {nums[1]}"] if grade == 8 else [],
        9: [f"({nums[2]} - {nums[1]}) / {nums[0]}"] if grade == 9 else [],
        10: [f"{nums[0]} - {nums[0]}", f"{nums[1]} - {nums[1]}"],
        11: [f"({nums[0]} + ({nums[1]})) / 2", f"{nums[0]} - {answer[0]}"] if grade == 11 else [],
        12: [f"{nums[0]} / {nums[0] + nums[1]} * {nums[0] - 1} / {nums[0] + nums[1] - 1}"],
    }[grade]
    if grade == 5 and nums[1] != nums[3]:
        errors.append("fraction denominators differ")
    steps = row.get("solution_steps", [])
    if len(steps) != len(expected_step_values):
        errors.append("incorrect number of steps")
    for index, step in enumerate(steps):
        pieces = step.split("=")
        # Only numeric arithmetic tokens can be passed to SymPy.
        if len(pieces) != 2 or not all(re.fullmatch(r"[\d +*/().-]+", s) for s in pieces):
            errors.append("unsupported step syntax")
            continue
        if index < len(expected_lhs) and pieces[0].replace(" ", "") != expected_lhs[index].replace(
            " ", ""
        ):
            errors.append("step is not linked to the question")
        try:
            lhs, rhs = sp.sympify(pieces[0]), sp.sympify(pieces[1])
            if sp.simplify(lhs - rhs) != 0:
                errors.append("incorrect step")
            if index < len(expected_step_values) and rhs != expected_step_values[index]:
                errors.append("step does not establish required intermediate value")
            if grade == 10 and index < 2:
                factors = [int(v) for v in re.findall(r"\d+", pieces[0])]
                if factors != [nums[index], nums[index]]:
                    errors.append("root step is not linked to the question")
        except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
            errors.append("unparseable step")
    if not row.get("solution_steps"):
        errors.append("missing steps")
    expected_id = hashlib.sha256(" ".join(row["question"].casefold().split()).encode()).hexdigest()
    if row.get("id") != expected_id:
        errors.append("question identity mismatch")
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            errors = verify(row)
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "errors": errors,
                        "answer_and_steps": "passed" if not errors else "failed",
                        "method": "independent parsing of displayed question, SymPy solution and numeric step equalities",
                        "caps_and_protected_status": "pending; held",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        json.dumps(
            {"records": len(rows), "passed": sum(not verify(r) for r in rows), "admitted": 0}
        )
    )


if __name__ == "__main__":
    main()
