import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "screen_reserved_math",
    Path(__file__).parents[1] / "scripts" / "datasets" / "mathematics" / "screen_reserved_math.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_screen_redacts_questions_and_flags_families(tmp_path):
    protected = tmp_path / "private"
    protected.mkdir()
    pool = protected / "private-assessment.jsonl"
    pool.write_text(json.dumps({"question": "Example confidential phrase with numbers 321"}) + "\n")
    families = protected / "family-exclusions.json"
    families.write_text(json.dumps({"forbidden_generator_families": ["restricted"]}))
    (protected / "freeze-manifest.json").write_text(
        json.dumps(
            {
                "private_assessment_sha256": module.digest(pool),
                "family_exclusions_sha256": module.digest(families),
            }
        )
    )
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {
                    "id": "a",
                    "question": "Example confidential phrase with numbers 321",
                    "problem_family": "other",
                },
                {"id": "b", "question": "Different question", "problem_family": "restricted"},
                {"id": "c", "original_source_text": "unrelated paragraph"},
            ]
        )
        + "\n"
    )
    output = tmp_path / "output"
    result = module.scan(protected, [source], output)
    assert result["counts"]["exact_question_text"] == 1
    assert result["counts"]["excluded_generator_family"] == 1
    assert result["counts"]["unmapped_family_rows"] == 2
    assert result["counts"]["unknown_or_unmapped_family"] == 2
    assert "confidential" not in (output / "flags.jsonl").read_text()
    pool.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.scan(protected, [source], tmp_path / "again")
