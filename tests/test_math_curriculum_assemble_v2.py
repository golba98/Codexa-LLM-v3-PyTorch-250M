from codexa.data.math_curriculum_assemble_v2 import canonical_key, text_and_kind


def test_duplicate_policy_preserves_different_source_identity_and_context() -> None:
    assert canonical_key("x + 1", "exercise", "source-one") == canonical_key(
        "x + 1", "exercise", "source-one"
    )
    assert canonical_key("x + 1", "exercise", "source-one") != canonical_key(
        "x + 1", "exercise", "source-two"
    )
    assert canonical_key("x + 1", "exercise", "source-one") != canonical_key(
        "x + 1", "explanation", "source-one"
    )
    assert canonical_key("x + 1", "exercise", "source-one") != canonical_key(
        "X + 1", "exercise", "source-one"
    )


def test_paired_source_exercise_retains_context_and_unverified_solution() -> None:
    source = {
        "record_type": "exercise_with_source_solution",
        "question": "Find the value.",
        "context": ["Use the diagram."],
        "source_solution": ["First step.", "[SOURCE-IMAGE:equation.png]"],
    }
    text, kind = text_and_kind(source)
    assert text == "Use the diagram.\nFind the value.\nFirst step.\n[SOURCE-IMAGE:equation.png]"
    assert kind == "exercise_with_source_solution"
