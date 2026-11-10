"""Explicit positive semantic dispositions; unchecked affected records are never admitted."""

import json
from pathlib import Path
from typing import Any

from codexa.data.production import digest
from codexa.data.quality_protocol import CHECKS, STATUSES

SOURCES = {"finemath-4plus", "fineweb-edu", "openwebmath"}


def assessed(rows: list[dict[str, Any]], expected: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if row["id"] in result or row["id"] not in expected or not row.get("reason"):
            raise ValueError("Invalid admission identity or rationale")
        if any(
            type(row.get(k)) is not bool
            for k in ("usable", "severe", "critical", "coverage_complete")
        ):
            raise ValueError("Explicit admission labels required")
        if not row["coverage_complete"]:
            raise ValueError("Complete each candidate before finalizing its census")
        if set(row.get("checks", {})) != set(CHECKS) or any(
            v not in STATUSES for v in row["checks"].values()
        ):
            raise ValueError("Seven explicit admission checks required")
        if any(not row.get(k) for k in ("judge_id", "topic", "level", "format")):
            raise ValueError("Judge identity and diversity annotations required")
        result[row["id"]] = row
    if set(result) != expected:
        raise ValueError("Complete the entire preregistered candidate census")
    return result


def cleared(row: dict[str, Any], source: str) -> bool:
    checks = row["checks"]
    return (
        row["usable"]
        and not row["severe"]
        and not row["critical"]
        and row["coverage_complete"]
        and all(
            checks[k] == "pass" for k in ("completeness", "education", "coherence", "contamination")
        )
        and all(value not in {"defect", "uncertain"} for value in checks.values())
        and (source != "finemath-4plus" or checks["correctness"] == "pass")
    )


def require_primary_evidence(row: dict[str, Any]) -> None:
    evidence = row.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, dict)
            or any(not item.get(k) for k in ("claim", "source_url", "finding", "limitation"))
            or not str(item["source_url"]).startswith("https://")
            for item in evidence
        )
    ):
        raise ValueError("Clearing factual uncertainty requires explicit primary evidence")


def primary_evidence_ids(protocol: dict[str, Any], expected: set[str]) -> set[str]:
    identifiers = protocol.get("primary_evidence_required_ids", [])
    flag = protocol.get("require_primary_evidence", False)
    if (
        type(flag) is not bool
        or not isinstance(identifiers, list)
        or any(not isinstance(ident, str) for ident in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or not set(identifiers) <= expected
    ):
        raise ValueError("Invalid primary-evidence requirements")
    return expected if flag else set(identifiers)


def verify_candidate_input(directory: Path, parent_sha256: str, count: int) -> str:
    path = directory / "token-verification.json"
    proof = json.loads(path.read_text())
    if (
        proof.get("status") != "passed"
        or proof.get("manifest_sha256") != parent_sha256
        or proof.get("documents") != count
        or proof.get("packet_integrity_sha256") != digest(directory / "packet-integrity.json")
    ):
        raise ValueError("Census text-to-token proof is missing or stale")
    return digest(path)


def load_admission(directory: Path, parent_sha256: str) -> dict[str, dict[str, Any]]:
    decision = json.loads((directory / "decision.json").read_text())
    if decision.get("parent_manifest_sha256") != parent_sha256 or decision.get("unreviewed") != 0:
        raise ValueError("Semantic admission is incomplete or belongs to another parent")
    integrity = json.loads((directory / "packet-integrity.json").read_text())
    if digest(directory / "packet-integrity.json") != decision["packet_integrity_sha256"]:
        raise ValueError("Semantic packet identity changed")
    for name, sha in integrity.items():
        if Path(name).name != name or digest(directory / name) != sha:
            raise ValueError("Semantic packet changed")
    for name, sha in decision["reviewer_hashes"].items():
        if Path(name).name != name or digest(directory / name) != sha:
            raise ValueError("Semantic judgments changed")
    if set(decision["reviewer_hashes"]) != {"curator-a.jsonl", "curator-b.jsonl"}:
        raise ValueError("Both independent semantic judgments must be bound")
    if digest(directory / "admission.jsonl") != decision["admission_sha256"]:
        raise ValueError("Semantic dispositions changed")
    mapping = [json.loads(x) for x in (directory / "mapping.jsonl").read_text().splitlines()]
    expected = {r["id"] for r in mapping}
    if len(expected) != len(mapping):
        raise ValueError("Duplicate census identity")
    if verify_candidate_input(directory, parent_sha256, len(mapping)) != decision.get(
        "token_verification_sha256"
    ):
        raise ValueError("Census text-to-token proof changed")
    judges = [
        assessed([json.loads(x) for x in (directory / name).read_text().splitlines()], expected)
        for name in ("curator-a.jsonl", "curator-b.jsonl")
    ]
    require_evidence = primary_evidence_ids(
        json.loads((directory / "protocol.json").read_text()), expected
    )
    ledger_rows = [json.loads(x) for x in (directory / "admission.jsonl").read_text().splitlines()]
    ledger = {row["id"]: row for row in ledger_rows}
    if set(ledger) != expected or len(ledger_rows) != len(expected):
        raise ValueError("Admission ledger differs from its complete census")
    result = {}
    for record in mapping:
        reviews = [judge[record["id"]] for judge in judges]
        if reviews[0]["judge_id"] == reviews[1]["judge_id"]:
            raise ValueError("A document needs two different independent semantic judges")
        admitted = all(cleared(r, record["source"]) for r in reviews)
        if record["id"] in require_evidence:
            for review in reviews:
                if cleared(review, record["source"]):
                    require_primary_evidence(review)
        row = ledger[record["id"]]
        if (
            any(row.get(k) != value for k, value in record.items())
            or (row["status"] == "admitted") != admitted
        ):
            raise ValueError("Semantic disposition differs from independent judgments")
        if record["document_id"] in result:
            raise ValueError("Duplicate semantic document")
        result[record["document_id"]] = row
    return result
