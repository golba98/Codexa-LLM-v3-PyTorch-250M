"""Protected language fixtures and paired scoring fail closed on changed evidence."""

import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

from codexa.data.production import atomic, digest
from codexa.evaluation.language import (
    DIMENSIONS,
    SETTINGS,
    TASKS,
    _evaluate,
    compare,
    fixture_rows,
    freeze,
    load_fixtures,
    protected_texts,
    review_packet,
    score_reviews,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def output_fixture(path: Path, fixtures: Path, sha: str, loop: bool = False) -> None:
    path.mkdir()
    atomic(
        path / "identity.json",
        {
            "checkpoint_sha256": sha,
            "fixture_manifest_sha256": digest(fixtures / "manifest.json"),
            "settings": SETTINGS,
            "device": "cpu",
        },
    )
    item = fixture_rows()[0]
    write_rows(
        path / "outputs.jsonl",
        [
            {
                "key": item["id"] + ":greedy",
                "item_id": item["id"],
                "split": "dev",
                "task": "reading",
                "setting": "greedy",
                "text": "The staff added a second box.",
                "ids": [5, 3],
                "truncated": False,
                "severe_loop": loop,
                "repeated_token_4gram_fraction": float(loop),
            }
        ],
    )

    verify_output(path)


def verify_output(path: Path) -> None:
    identity = json.loads((path / "identity.json").read_text())
    rows = (path / "outputs.jsonl").read_text().splitlines()
    atomic(
        path / "verification.json",
        identity
        | {
            "outputs_sha256": digest(path / "outputs.jsonl"),
            "records": len(rows),
            "weights_unchanged": True,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "forbidden_attempts": [],
        },
    )


def test_freeze_protects_all_authored_fields_and_discloses_correlation(tmp_path):
    fixtures = tmp_path / "fixtures"
    manifest = freeze(fixtures)
    rows = load_fixtures(fixtures)
    assert len(rows) == 200
    for split in ("dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        assert Counter(row["task"] for row in selected) == dict.fromkeys(TASKS, 20)
        assert len({row["family"] for row in selected}) == 20
    assert {row["passage"] for row in rows if row["split"] == "dev"}.isdisjoint(
        {row["passage"] for row in rows if row["split"] == "test"}
    )
    protected = set(protected_texts(fixtures))
    assert all(row["passage"] in protected and row["reference"] in protected for row in rows)
    assert manifest["contexts_per_split"] == 20
    with pytest.raises(FileExistsError):
        freeze(fixtures)


def test_fixture_changes_and_rehashed_spec_changes_are_rejected(tmp_path):
    fixtures = tmp_path / "fixtures"
    freeze(fixtures)
    path = fixtures / "dev.jsonl"
    path.write_text(path.read_text().replace("second box", "third box"))
    with pytest.raises(ValueError, match="changed"):
        load_fixtures(fixtures)
    manifest = json.loads((fixtures / "manifest.json").read_text())
    manifest["files"]["dev.jsonl"] = digest(path)
    atomic(fixtures / "manifest.json", manifest)
    with pytest.raises(ValueError, match="versioned"):
        load_fixtures(fixtures)


def test_independent_review_requires_complete_ratings_and_keeps_disagreement(tmp_path):
    fixtures = tmp_path / "fixtures"
    freeze(fixtures)
    before, after = tmp_path / "before", tmp_path / "after"
    output_fixture(before, fixtures, "before", loop=True)
    output_fixture(after, fixtures, "after")
    packet = tmp_path / "packet"
    review_packet(fixtures, [before, after], packet)
    blind = [json.loads(line) for line in (packet / "blind.jsonl").read_text().splitlines()]
    assert all("checkpoint_sha256" not in row and "output" not in row for row in blind)
    first = [
        {
            "id": row["id"],
            "scores": dict.fromkeys(DIMENSIONS, 2),
            "uncertain": False,
            "reason": "Read full response and compared against passage.",
        }
        for row in blind
    ]
    second = json.loads(json.dumps(first))
    second[0]["scores"]["coherence"] = 0
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_rows(a, first)
    write_rows(b, second[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        score_reviews(packet, {"one": a, "two": b})
    write_rows(b, second)
    result = score_reviews(packet, {"one": a, "two": b})
    assert sum(row["disagreement"] for row in result["records"]) == 1
    assert any(row["scores"]["coherence"] == 0 for row in result["records"])
    paired = compare(before, after, result)
    assert paired["pairs"] == 1
    assert paired["metrics"]["dev:reading:greedy"]["severe_loops_before"] == 1
    assert paired["metrics"]["dev:reading:greedy"]["severe_loops_after"] == 0
    with pytest.raises(ValueError, match="distinct"):
        score_reviews(packet, {"one": a, "two": a})
    (packet / "blind.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        score_reviews(packet, {"one": a, "two": b})


def test_paired_settings_must_match_and_missing_cases_are_disclosed(tmp_path):
    fixtures = tmp_path / "fixtures"
    freeze(fixtures)
    before, after = tmp_path / "before", tmp_path / "after"
    output_fixture(before, fixtures, "before")
    output_fixture(after, fixtures, "after")
    write_rows(after / "outputs.jsonl", [])
    verify_output(after)
    result = compare(before, after)
    assert result["pairs"] == 0 and len(result["before_only"]) == 1
    identity = json.loads((after / "identity.json").read_text())
    identity["settings"] = {"max_tokens": 2}
    atomic(after / "identity.json", identity)
    with pytest.raises(ValueError, match="evidence"):
        compare(before, after)


def test_evaluation_budget_is_combined_and_failed_load_is_charged(tmp_path, monkeypatch):
    root = tmp_path
    fixtures = root / "benchmarks/fixtures"
    freeze(fixtures)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_text("fixture")
    budget_path = root / "benchmarks/language-pair-budget.json"
    atomic(budget_path, {"limit_seconds": 14400, "used_seconds": 14400})
    with pytest.raises(RuntimeError, match="exhausted"):
        _evaluate(root, fixtures, checkpoint, root / "benchmarks/output", 1, "cpu")
    atomic(budget_path, {"limit_seconds": 14400, "used_seconds": 0.0})
    benchmark = importlib.import_module("codexa.evaluation.benchmark")

    def fail(*args):
        raise RuntimeError("test failed model loading")

    monkeypatch.setattr(benchmark, "load_frozen", fail)
    with pytest.raises(RuntimeError, match="test failed"):
        _evaluate(root, fixtures, checkpoint, root / "benchmarks/output", 1, "cpu")
    budget = json.loads(budget_path.read_text())
    assert budget["used_seconds"] > 0 and "active_since" not in budget
    verification = json.loads((root / "benchmarks/output/verification.json").read_text())
    assert verification["complete"] is False and verification["weights_unchanged"] is False


def test_generation_tampering_cannot_be_used_for_blinded_review(tmp_path):
    fixtures = tmp_path / "fixtures"
    freeze(fixtures)
    output = tmp_path / "output"
    output_fixture(output, fixtures, "model")
    path = output / "outputs.jsonl"
    path.write_text(path.read_text().replace("second box", "third box"))
    with pytest.raises(ValueError, match="evidence"):
        review_packet(fixtures, [output], tmp_path / "packet")


def test_review_filters_are_fixed_and_unselected_outputs_stay_unreviewed(tmp_path):
    fixtures = tmp_path / "fixtures"
    freeze(fixtures)
    output = tmp_path / "output"
    output_fixture(output, fixtures, "model")
    rows = [json.loads(line) for line in (output / "outputs.jsonl").read_text().splitlines()]
    selected = next(row for row in fixture_rows() if row["split"] == "test")
    rows.append(
        rows[0]
        | {
            "item_id": selected["id"],
            "split": "test",
            "setting": "sample",
            "key": selected["id"] + ":sample",
        }
    )
    write_rows(output / "outputs.jsonl", rows)
    verify_output(output)
    manifest = review_packet(fixtures, [output], tmp_path / "panel", split="test", setting="sample")
    assert manifest["records"] == 1
    assert manifest["selection"]["split"] == "test"
    assert len((output / "outputs.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="Unknown"):
        review_packet(fixtures, [output], tmp_path / "bad", setting="other")
    with pytest.raises(ValueError, match="No generation"):
        review_packet(fixtures, [output], tmp_path / "empty", split="dev", setting="sample")


def test_evaluation_resume_rejects_changed_saved_text(tmp_path):
    fixtures = tmp_path / "benchmarks/fixtures"
    freeze(fixtures)
    output = tmp_path / "benchmarks/output"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint")
    output_fixture(output, fixtures, digest(checkpoint))
    path = output / "outputs.jsonl"
    path.write_text(path.read_text().replace("second box", "third box"))
    with pytest.raises(ValueError, match="evidence"):
        _evaluate(tmp_path, fixtures, checkpoint, output, 1, "cpu")


def test_historical_paragraph_selection_cannot_mix_with_other_outputs(tmp_path, monkeypatch):
    from codexa.evaluation import benchmark

    fixtures = tmp_path / "benchmarks/v3"
    fixtures.mkdir(parents=True)
    atomic(fixtures / "manifest.json", {"files": {}})
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_text("Test identity only; no model loading expected.")
    output = tmp_path / "benchmarks/selected"
    output.mkdir()

    class Disk:
        def check(self, allocation):
            pass

    monkeypatch.setattr(benchmark.Storage, "load", lambda root: Disk())
    (output / "outputs.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="unfiltered"):
        benchmark._evaluate(tmp_path, checkpoint, 1, output, kinds={"paragraph"})
    (output / "outputs.jsonl").unlink()
    atomic(
        output / "selection.json",
        {"kinds": ["math"], "fixture_manifest_sha256": digest(fixtures / "manifest.json")},
    )
    with pytest.raises(ValueError, match="selection changed"):
        benchmark._evaluate(tmp_path, checkpoint, 1, output, kinds={"paragraph"})
