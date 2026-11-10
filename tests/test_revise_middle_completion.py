"""Ensure answer corrections preserve prior evidence and remain held."""

from __future__ import annotations

from scripts.revise_middle_completion import CORRECTIONS, revise


def test_corrected_answers_keep_lineage_and_original_unchanged() -> None:
    for record_id, (answer, _) in CORRECTIONS.items():
        source = {
            "record_id": record_id,
            "record_type": "worked_example",
            "answer": "old answer",
            "admission": "blocked",
        }
        revised = revise(source)
        assert revised["answer"] == answer
        assert revised["previous_answer"] == "old answer"
        assert revised["previous_record_id"] == record_id
        assert revised["record_id"] != record_id
        assert revised["admission"] == "blocked"
        assert source["answer"] == "old answer"


def test_explanations_do_not_get_answer_review_labels() -> None:
    source = {"record_id": "explanation", "record_type": "concept_explanation"}
    assert "independent_question_answer_review" not in revise(source)
