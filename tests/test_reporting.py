"""Report output must preserve evidence and keep launch decisions independent."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from codexa.reporting import evidence_summary, render_report, write_report
from codexa.repository import check_archive, check_documentation
from scripts.maintenance.render_reports import render_tree


def readiness():
    return json.loads(
        Path("reports/language/evidence/continuation-readiness-report.json").read_text()
    )


def test_reviewed_reports_regenerate_exactly():
    render_tree(Path("reports"), check=True)
    check_documentation()
    check_archive(
        Path("reports/archive/research-2026-11-10.tar.gz"), Path("reports/archive/index.json")
    )


def test_quality_pass_cannot_authorize_execution():
    record = readiness()
    assert record["training"] == {
        "authorized": True,
        "execution_permitted": False,
        "started": False,
    }
    assert record["gates"]["quality"] == "PASS"
    record["status"] = "PASS"
    with pytest.raises(ValueError, match="Source approval"):
        render_report(record)


@pytest.mark.parametrize(
    "field", ["date", "model_version", "methodology", "limitations", "evidence"]
)
def test_missing_evidence_or_context_is_rejected(field):
    record = readiness()
    del record[field]
    with pytest.raises(ValueError):
        render_report(record)


def test_summary_preserves_raw_evidence_and_escapes_tables(tmp_path):
    source = tmp_path / "original.json"
    payload = b'{"raw": "retained evidence"}\n'
    source.write_bytes(payload)
    record = evidence_summary(
        title="A bounded measurement",
        evidence=source,
        metrics=[{"name": "Count | scope", "value": 3, "scope": "Measured\non fixture"}],
        limitations=["No model capability is established."],
    )
    before = copy.deepcopy(record)
    report = tmp_path / "report.md"
    write_report(record, report)
    assert source.read_bytes() == payload
    assert record == before
    assert record["evidence"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert "Count \\| scope" in report.read_text()
    assert "Measured on fixture" in report.read_text()
    with pytest.raises(FileExistsError):
        write_report(record, report)


@pytest.mark.parametrize(
    "private", ["/home/person/data", "/Users/person/run", "private-assessment.jsonl"]
)
def test_private_workstation_and_assessment_details_are_rejected(private):
    record = readiness()
    record["limitations"].append(private)
    with pytest.raises(ValueError, match="Private"):
        render_report(record)


def test_changed_evidence_hash_is_rejected():
    record = readiness()
    record["evidence"][0]["sha256"] = "unknown"
    with pytest.raises(ValueError, match="SHA256"):
        render_report(record)


@pytest.mark.parametrize(
    "link", ["[missing](#absent)", "[missing][ref]\n\n[ref]: missing.md", "![image](missing.png)"]
)
def test_broken_documentation_targets_are_rejected(tmp_path, monkeypatch, link):
    from codexa import repository

    doc = tmp_path / "guide.md"
    doc.write_text("# Present\n\n" + link + "\n")
    monkeypatch.setattr(
        repository.subprocess, "check_output", lambda *args, **kwargs: str(doc) + "\n"
    )
    with pytest.raises(ValueError, match="Broken"):
        repository.check_documentation()
