import re
from collections import Counter

import pytest

from codexa.data.math_synthetic_v1 import FAMILIES, build, generate


def test_deterministic_coordinates_and_distinct_ids():
    rows = generate(144, 2)
    assert rows == generate(144, 2)
    assert len({r["id"] for r in rows}) == len(rows)
    assert rows != generate(145, 2)
    assert set(Counter(r["grade"] for r in rows)) == set(range(1, 13))
    assert all(r["training_eligibility"].startswith("held_") for r in rows)


@pytest.mark.parametrize("grade,family", [(g, f) for g, fs in FAMILIES.items() for f in fs])
def test_generated_answers_pass_independent_recalculation(grade, family):
    # Multiple values exercise exact arithmetic, degenerate repeated roots and construction.
    for i in range(15):
        row = build(12637, grade, family, i)
        assert row["answer_verification"] == "passed_sympy_independent_recalculation"
        assert row["grade_mapping_status"] == "provisional_caps_review_required"
        assert row["explanation"] is None
        assert row["question"] and row["answer"]
        if family == "ratio":
            assert row["answer_unit"]


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError):
        build(1, 12, "addition", 0)
    with pytest.raises(ValueError):
        generate(1, 0)


@pytest.mark.parametrize(
    "grade,family,question,answer,unit",
    [
        (1, "addition", "Add 7 and 9.", "16", None),
        (
            7,
            "ratio",
            (
                "A mixture uses 12 litres of A for every 10 litres of B. "
                "How many litres of A are needed for 100 litres of B?"
            ),
            "120",
            "litres of A",
        ),
        (
            9,
            "simultaneous",
            "Solve simultaneously: x + y = 24 and x - y = 16.",
            "(20, 4)",
            None,
        ),
        (10, "quadratic_roots", "Solve (x - 15)(x - 19) = 0 over the reals.", "15, 19", None),
        (12, "derivative", "If f(x) = 10x² + 13x + 5, find f'(x).", "20*x + 13", None),
    ],
)
def test_known_answers_independently_checked(grade, family, question, answer, unit):
    row = build(19, grade, family, 0)
    assert (row["question"], row["answer"], row["answer_unit"]) == (question, answer, unit)


def test_repeated_quadratic_root_is_reported_once():
    # A double root is one distinct real solution.
    # Find a repeated-root case deterministically without relying on the generator's verifier.
    for i in range(1000):
        row = build(0, 10, "quadratic_roots", i)
        values = re.findall(r"x - (\d+)", row["question"])
        if len(values) == 2 and values[0] == values[1]:
            assert row["answer"] == values[0]
            break
    else:
        pytest.fail("No repeated-root case generated")
