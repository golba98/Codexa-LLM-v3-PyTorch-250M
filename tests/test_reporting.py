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


def test_historical_presentation_preserves_original_decisions_and_evidence(tmp_path):
    record = readiness()
    before = copy.deepcopy(record)
    rendered = render_report(record)
    assert "Historical snapshot" in rendered
    assert "500M targets" in rendered
    assert "Actions recorded at the time" in rendered
    assert "1,000,000,000" in rendered
    assert "| Training started | NO |" in rendered
    for evidence in record["evidence"]:
        assert evidence["location"] in rendered
        assert evidence["sha256"] in rendered
    assert record == before

    root = tmp_path / "reports"
    source = root / "language" / "evidence" / "readiness-report.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record))
    current = copy.deepcopy(record)
    current.update(report_id="current-assessment", historical=False, takeaway="Current | finding")
    source.with_name("current-report.json").write_text(json.dumps(current))
    render_tree(root)
    render_tree(root, check=True)
    index = (root / "README.md").read_text()
    results, history = index.split("## Historical decisions")
    assert "(language/current-assessment.md)" in results
    assert "(language/continuation-readiness.md)" in history
    assert "Current \\| finding" in results


def test_legacy_report_defaults_need_no_presentation_metadata(tmp_path):
    record = readiness()
    record.pop("historical", None)
    record.pop("takeaway", None)
    rendered = render_report(record)
    assert record["summary"][0] in rendered
    assert "Historical snapshot" not in rendered
    assert "## Next steps" in rendered
    source = tmp_path / "language" / "evidence" / "legacy-report.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record))
    render_tree(tmp_path)
    assert "## Historical decisions" not in (tmp_path / "README.md").read_text()


@pytest.mark.parametrize("metadata", [{"historical": "false"}, {"takeaway": ""}, {"takeaway": 3}])
def test_invalid_presentation_metadata_is_rejected(metadata):
    record = readiness()
    record.update(metadata)
    with pytest.raises(ValueError):
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
