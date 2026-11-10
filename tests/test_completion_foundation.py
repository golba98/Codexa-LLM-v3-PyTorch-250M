from scripts.math_completion_workers.foundation import candidates


def base(text):
    return {
        "source_id": "bala-teacher",
        "original_source_text": text,
        "grade": 2,
        "source_file_sha256": "abc",
        "page_or_section": "PDF page 1",
        "id": "parent",
    }


def test_extracts_bounded_sections_and_preserves_text():
    mental = "Learners add the two given numbers, then explain how to check their answers using concrete counters."
    watch = "Learners may mistake the order of the terms, so ask them to describe each step clearly (and check)."
    text = (
        "Mental Maths\n"
        + mental
        + "\nGame\nGame instructions\nConcept development\nConcept text\nWhat to look out for this week\n"
        + watch
    )
    rows = candidates(base(text))
    assert [r["record_type"] for r in rows] == [
        "mental_mathematics_guidance",
        "teacher_diagnostic_guidance",
    ]
    assert [r["original_text"] for r in rows] == [mental, watch]
    assert all(
        r["admission_status"] == "blocked" and r["source_fidelity"] == "pending" for r in rows
    )
    assert all(
        text[r["source_span"]["start"] : r["source_span"]["end"]].strip() == r["original_text"]
        for r in rows
    )


def test_ambiguous_or_truncated_sections_are_not_emitted():
    text = (
        "Mental Maths\n"
        + ("a" * 100)
        + ".\nConcept development\nx\nWhat to look out for this week\n"
        + ("b" * 100)
    )
    rows = candidates(base(text))
    assert [r["record_type"] for r in rows] == ["mental_mathematics_guidance"]
    assert candidates(base("Mental Maths\n" + text)) == []
