"""Independent worked values for synthetic extension families."""

import pytest

from codexa.data.math_synthetic_extensions_v1 import CASES, build


@pytest.mark.parametrize(
    "grade,family,question,answer",
    [
        (
            9,
            "probability_single",
            "A bag contains 5 red counters and 9 blue counters. One counter is chosen at random. What is the probability that it is red?",
            "5/14",
        ),
        (9, "statistics_median", "Find the median of this data set: 5, 4, 30, 38, 40.", "30"),
        (9, "statistics_range", "Find the range of this data set: 24, 32, 17, 13, 18.", "19"),
        (10, "function_evaluate", "Let f(x) = 5x + 4. Find f(3).", "19"),
        (
            10,
            "function_intercept",
            "The straight line y = 11x + 6 crosses the y-axis at (0, k). Find k.",
            "6",
        ),
        (10, "statistics_mean", "Find the mean of this data set: 11, 23, 15, 4, 37.", "18"),
        (
            11,
            "probability_without_replacement",
            "A bag contains 2 red counters and 13 blue counters. Two counters are chosen uniformly without replacement. What is the probability that both are red?",
            "1/105",
        ),
        (
            11,
            "trig_sin",
            "A right triangle has an acute angle θ. Relative to θ, the opposite side is 15 cm and the adjacent side is 20 cm. Find sin θ exactly.",
            "3/5",
        ),
        (
            11,
            "trig_cos",
            "A right triangle has an acute angle θ. Relative to θ, the opposite side is 16 cm and the adjacent side is 30 cm. Find cos θ exactly.",
            "15/17",
        ),
        (
            11,
            "trig_tan",
            "A right triangle has an acute angle θ. Relative to θ, the opposite side is 9 cm and the adjacent side is 12 cm. Find tan θ exactly.",
            "3/4",
        ),
        (12, "function_inverse", "Let f(x) = 10x + 8, for real x. Find f⁻¹(x).", "(x - 8)/10"),
        (12, "function_quadratic", "Let f(x) = x² - 12x + 10. Find f(14).", "38"),
    ],
)
def test_known_worked_answers(grade, family, question, answer):
    row = build(8, grade, family, 0)
    assert row["question"] == question
    assert row["answer"] == answer
    assert row["training_eligibility"].startswith("held_")
    assert row["grade_mapping_status"] == "provisional_caps_review_required"


@pytest.mark.parametrize("grade,family", [(g, f) for g, fs in CASES.items() for f in fs])
def test_many_values_deterministic(grade, family):
    rows = [build(901, grade, family, i) for i in range(20)]
    assert rows == [build(901, grade, family, i) for i in range(20)]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["assumptions"] and r["explanation"] is None for r in rows)


def test_bad_coordinates_rejected():
    with pytest.raises(ValueError):
        build(1, 1, "trig_sin", 0)
    with pytest.raises(ValueError):
        build(1, 11, "trig_sin", -1)
