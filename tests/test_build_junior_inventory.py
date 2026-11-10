"""Regression tests for Grade 8–9 evidence inventory and fail-closed state."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "junior_inventory", Path(__file__).parents[1] / "scripts" / "build_junior_inventory.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bullets_keep_page_text_and_do_not_claim_objectives():
    result = module.split_bullets(
        "topic\n Multiply with integers\n including negatives\n− test a case\n"
    )
    assert result == [(2, " Multiply with integers including negatives"), (4, "− test a case")]


def test_inputs_provenance_and_no_approval(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw").mkdir()
    raw = b"<p>Example</p>"
    (source / "raw" / "chapter.html").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    item = {
        "id": "unit",
        "grade": 9,
        "raw_file": "raw/chapter.html",
        "source_file_sha256": digest,
        "source_id": "src",
        "source_url": "https://a.test/chapter",
        "diagram_dependency": True,
    }
    registry = {"source_id": "src", "sha256": digest}
    section = {
        "url": "https://a.test/chapter",
        "source_id": "src",
        "sha256": digest,
        "status": "parsed",
    }
    toc = {
        "grade": 9,
        "label": "Chapter 1: Whole numbers",
        "source_toc_url": "https://a.test/g9",
        "url": "https://a.test/chapter",
        "document_url": "https://a.test/chapter",
    }
    for name, row in [
        ("records", item),
        ("sources", registry),
        ("inventory", section),
        ("toc-items", toc),
    ]:
        (source / (name + ".jsonl")).write_text(json.dumps(row) + "\n")
    policy = tmp_path / "policy"
    policy.mkdir()
    entries = []
    for grade in (8, 9):
        data = b"%PDF-FAKE"
        (policy / f"grade{grade}-atp.pdf").write_bytes(data)
        entries.append(
            {
                "name": f"grade{grade}-atp",
                "status": "acquired",
                "sha256": hashlib.sha256(data).hexdigest(),
                "url": "https://dbe.test",
            }
        )
    (policy / "acquisition.json").write_text(json.dumps(entries))
    monkeypatch.setattr(module, "policy_items", lambda *args: ([], []))
    out = tmp_path / "output"
    report = module.build(source, policy, out)
    assert report["record_count"] == 1 and report["final_admitted"] == 0
    row = json.loads((out / "records-source.jsonl").read_text().strip())
    assert row["extraction"] == "visual-dependent" and row["rights"].startswith("unresolved")
    assert row["objective_ids"] == [] and row["admission"] == "not_requested"
    with pytest.raises(FileExistsError):
        module.build(source, policy, out)
    (source / "raw" / "chapter.html").write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw hash mismatch"):
        module.build(source, policy, tmp_path / "other")
