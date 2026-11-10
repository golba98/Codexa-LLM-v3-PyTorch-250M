"""Small, deterministic research summaries. Rendering never runs an experiment."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATUSES = {"PASS", "FAIL", "BLOCKED", "IN PROGRESS"}
SECTIONS = ("summary", "methodology", "results", "limitations", "next_actions")


def validate_report(record: dict[str, Any]) -> None:
    if record.get("schema_version") != 1 or record.get("status") not in STATUSES:
        raise ValueError("Unsupported report schema or status")
    for field in (
        "report_id",
        "title",
        "purpose",
        "date",
        "run_id",
        "model_version",
        "dataset_version",
    ):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError("Missing report metadata: " + field)
    ident = record["report_id"]
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in ident):
        raise ValueError("Report identifier must be lowercase kebab-case")
    datetime.fromisoformat(record["date"])
    for field in SECTIONS:
        if not isinstance(record.get(field), list) or not record[field]:
            raise ValueError("Missing report section: " + field)
        if not all(isinstance(item, str) and item.strip() for item in record[field]):
            raise ValueError("Report sections require nonempty text")
    for field in ("metrics", "evidence"):
        if not isinstance(record.get(field), list) or not record[field]:
            raise ValueError("Missing report section: " + field)
    for metric in record["metrics"]:
        if not all(k in metric for k in ("name", "value", "scope")):
            raise ValueError("Metrics require a name, value and measured/estimated scope")
    for evidence in record["evidence"]:
        if not all(evidence.get(k) for k in ("label", "location", "sha256")):
            raise ValueError("Evidence requires a location and original SHA256")
        sha = evidence["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid evidence SHA256")
    if record.get("kind") == "continuation-readiness":
        training = record.get("training", {})
        if not all(
            type(training.get(k)) is bool for k in ("authorized", "execution_permitted", "started")
        ):
            raise ValueError("Readiness requires separate authorization/execution/start decisions")
        gates = record.get("gates", {})
        if record["status"] == "PASS" and (
            not gates
            or any(value != "PASS" for value in gates.values())
            or not training["authorized"]
            or not training["execution_permitted"]
        ):
            raise ValueError("Source approval alone cannot establish launch readiness")
    # Public records contain portable artifact locators, never workstation home paths.
    text = json.dumps(record)
    if "/home/" in text or "/Users/" in text or "private-assessment.jsonl" in text:
        raise ValueError("Private workstation or assessment details in report")


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(record: dict[str, Any]) -> str:
    validate_report(record)
    lines = [
        "<!-- Generated from the adjacent evidence record; edit the JSON and regenerate. -->",
        "# " + record["title"],
        "",
        record["purpose"],
        "",
        "| Date | Run | Model | Dataset | Status |",
        "| --- | --- | --- | --- | --- |",
        "| "
        + " | ".join(
            cell(record[k])
            for k in ("date", "run_id", "model_version", "dataset_version", "status")
        )
        + " |",
        "",
    ]
    if record.get("kind") == "continuation-readiness":
        lines += ["## Readiness decisions", "", "| Gate / decision | State |", "| --- | --- |"]
        for name, state in record["gates"].items():
            lines.append(f"| {cell(name.replace('_', ' '))} | {cell(state)} |")
        training = record["training"]
        for name in ("authorized", "execution_permitted", "started"):
            lines.append(
                f"| Training {name.replace('_', ' ')} | {'YES' if training[name] else 'NO'} |"
            )
        lines.append("")
    titles = {
        "summary": "Executive summary",
        "methodology": "Methodology",
        "results": "Results",
        "limitations": "Failures and limitations",
        "next_actions": "Required next actions",
    }
    for section in (
        "summary",
        "metrics",
        "methodology",
        "results",
        "limitations",
        "evidence",
        "next_actions",
    ):
        lines += [
            "## "
            + titles.get(
                section,
                {"metrics": "Key metrics", "evidence": "Supporting evidence"}.get(section, section),
            ),
            "",
        ]
        if section == "metrics":
            lines += ["| Metric | Value | Scope |", "| --- | --- | --- |"]
            lines += [
                "| " + " | ".join(cell(m[k]) for k in ("name", "value", "scope")) + " |"
                for m in record[section]
            ]
        elif section == "evidence":
            lines += ["| Evidence | Artifact location | SHA256 |", "| --- | --- | --- |"]
            lines += [
                f"| {cell(e['label'])} | `{cell(e['location'])}` | `{e['sha256']}` |"
                for e in record[section]
            ]
            lines += [
                "",
                "`external:` locations are relative to the private artifact root; payloads are not distributed here.",
            ]
        else:
            lines += ["- " + item for item in record[section]]
        lines.append("")
    return "\n".join(lines)


def write_report(record: dict[str, Any], output: Path, *, replace: bool = False) -> None:
    content = render_report(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if replace else "x", encoding="utf-8") as stream:
        stream.write(content)


def evidence_summary(
    *, title: str, evidence: Path, metrics: list[dict[str, Any]], limitations: list[str]
) -> dict[str, Any]:
    """Adapter for local tool receipts; never copies raw corpus text or review examples."""
    return {
        "schema_version": 1,
        "report_id": "tool-summary",
        "title": title,
        "purpose": "Summarize an existing machine-readable receipt without executing a model.",
        "date": datetime.fromtimestamp(evidence.stat().st_mtime, UTC).date().isoformat(),
        "run_id": evidence.parent.name,
        "model_version": "See original receipt",
        "dataset_version": "See original receipt",
        "status": "IN PROGRESS",
        "summary": [
            "This summary records tool output; it does not confer dataset approval or training authorization."
        ],
        "metrics": metrics,
        "methodology": ["Read the original JSON receipt; retain its format and SHA256."],
        "results": ["Measured values below retain the tool's original scope."],
        "limitations": limitations,
        "evidence": [
            {
                "label": "Original receipt",
                "location": evidence.name,
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "next_actions": ["Review the receipt and applicable gates before promoting any result."],
    }
