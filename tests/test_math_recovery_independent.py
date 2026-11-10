"""Checks use fixed independently calculated answers and deliberate corruption."""

import pytest

from codexa.data.math_recovery_check import verify
from codexa.data.math_recovery_generate import build, generate


@pytest.mark.parametrize(
    ("grade", "params", "expected"),
    [
        (1, {"a": 3, "b": 5}, "5"),
        (2, {"a": 4, "b": 3}, "12"),
        (3, {"a": 7, "b": 3}, "7"),
        (4, {"a": 7, "b": 4}, "22"),
        (5, {"a": 1, "b": 3, "c": 8}, "1/2"),
        (6, {"a": 25, "b": 400}, "100"),
        (7, {"a": 3, "b": 5, "c": 4}, "12"),
        (8, {"a": 4, "b": 3, "c": 2}, "11"),
        (9, {"a": 4, "b": 3, "c": 2}, "2"),
        (10, {"a": 2, "b": 7}, "2, 7"),
        (11, {"a": 4, "b": 7}, "(4, 7)"),
        (12, {"a": 3, "b": 2}, "3/10"),
    ],
)
def test_independent_fixed_answers(grade, params, expected):
    row = build(grade, params, 7)
    assert row["answer"] == expected
    assert verify(row) == []
    row["answer"] = "999"
    assert verify(row)


def test_step_and_units_corruption_detected():
    row = build(4, {"a": 4, "b": 7}, 2)
    row["solution_steps"] = ["1 + 1 = 2"]
    row["answer_unit"] = "m"
    assert len(verify(row)) >= 2


def test_families_and_identity_are_stable_and_reserved_fails():
    freeze = {"reserved_families": ["comparison.order"]}
    first = generate(7, 3, freeze)
    assert first == generate(7, 3, freeze)
    assert len({r["id"] for r in first}) == 36
    assert len({r["problem_family"] for r in first}) == 12
    with pytest.raises(ValueError, match="protected"):
        generate(7, 3, {"reserved_families": [first[0]["problem_family"]]})
