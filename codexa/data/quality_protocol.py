"""Versioned, fail-closed quality gates; historical decisions remain historical."""

from collections import Counter
from typing import Any

VERSION = "codexa-quality-4"
SOURCES = {"finemath-4plus", "fineweb-edu", "openwebmath", "tinystories-gpt4", "wikipedia-en"}
CHECKS = (
    "notation",
    "completeness",
    "visuals",
    "correctness",
    "education",
    "coherence",
    "contamination",
)
STATUSES = {"pass", "defect", "uncertain", "not_applicable"}


def specification(kind: str = "primary", attempt: int = 0) -> dict[str, Any]:
    if kind not in {"primary", "extended"} or attempt < 0:
        raise ValueError("Unknown review kind or negative attempt")
    return {
        "version": VERSION,
        "kind": kind,
        "seed": (2026110601 if kind == "primary" else 2026110602) + 1000 * attempt,
        "attempt": attempt,
        "windows": 300 if kind == "primary" else 0,
        "window_tokens": 1024,
        "documents_per_source": 40 if kind == "primary" else 200,
        "sources": sorted(SOURCES if kind == "primary" else {"finemath-4plus", "fineweb-edu"}),
        "maximum_full_document_tokens": None,
        "complete_required": True,
        "early_stop": False,
        "usable_window_minimum": 270,
        "severe_window_maximum": 15,
        "usable_per_source_minimum": 34 if kind == "primary" else 170,
        "severe_per_source_maximum": 2,
        "finemath_severe_maximum": 0,
        "finemath_prohibited_maximum": 0,
        "fineweb_usable_minimum": 40 if kind == "primary" else 200,
        "fineweb_severe_maximum": 0,
        "critical_maximum": 0,
        "independent_reviewers_required": 2,
        "required_checks": list(CHECKS),
        "rubric": (
            "Read every complete document. Usable means meaningful, coherent content, not merely "
            "educational keywords. Assess notation, completeness, essential visual inputs, solution "
            "correctness, educational substance, coherence and contamination separately. Mark "
            "uncertain when evidence cannot establish a check; never guess. Corrected learner "
            "mistakes are not broken solutions. Optional illustrations do not imply missing inputs. "
            "FineMath prohibits broken solutions, missing required diagrams, incomprehensible "
            "notation, unrelated merged exercises and advertising/navigation in answers. "
            "FineWeb-Edu requires substantive educational content, complete intelligible extraction "
            "and no severe corruption, nonsensical answers, resource-only advertising or misleading "
            "formatting. Random-window edge crops and actual BOS/EOS transitions are not defects. "
            "Critical means an observed protected-evaluation leak or sensitive identifier. Public "
            "biographical names alone are not critical. Source style may compromise blinding."
        ),
        "disagreement_policy": "Preserve both independent ratings; unresolved concerns fail. A third blinded reviewer may resolve with explicit evidence before unblinding.",
        "exclusion_policy": "Exclude all prior panel document identities and verified 64-token review passages from both document and window samples; primary and extended panels are disjoint.",
    }


def validate_specification(protocol: dict[str, Any]) -> None:
    expected = specification(protocol.get("kind", ""), protocol.get("attempt", -1))
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Strict quality protocol differs from its frozen version")


def adjudicate(
    mapping: list[dict[str, Any]], ratings: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    validate_specification(protocol)
    by_id = {row["id"]: row for row in mapping}
    if len(by_id) != len(mapping):
        raise ValueError("Duplicate sample identity")
    expected = Counter("windows" if r["kind"] == "window" else r["source"] for r in mapping)
    dimensions = {s: protocol["documents_per_source"] for s in protocol["sources"]}
    if protocol["windows"]:
        dimensions["windows"] = protocol["windows"]
    if dict(expected) != dimensions:
        raise ValueError("Review panel dimensions differ from preregistration")
    seen: set[str] = set()
    panels: dict[str, Counter[str]] = {}
    blockers = []
    for row in ratings:
        ident = row["id"]
        if ident not in by_id or ident in seen or not str(row.get("reason", "")).strip():
            raise ValueError("Invalid, duplicate or unexplained rating")
        if any(
            type(row.get(k)) is not bool
            for k in ("usable", "severe", "critical", "coverage_complete")
        ):
            raise ValueError("Explicit boolean labels and coverage required")
        checks = row.get("checks", {})
        if set(checks) != set(CHECKS) or any(v not in STATUSES for v in checks.values()):
            raise ValueError("All structured quality checks are required")
        seen.add(ident)
        sample = by_id[ident]
        panel = "windows" if sample["kind"] == "window" else sample["source"]
        c = panels.setdefault(panel, Counter())
        c["reviewed"] += 1
        usable = row["usable"]
        if not row["coverage_complete"]:
            blockers.append(ident + ": incomplete review coverage")
        if sample["kind"] == "document" and sample.get("coverage") != "full document":
            raise ValueError("Strict document review requires complete source text")
        if panel == "finemath-4plus":
            prohibited = any(
                checks[k] in {"defect", "uncertain"}
                for k in ("notation", "completeness", "visuals", "correctness", "contamination")
            )
            c["prohibited"] += prohibited
            usable = usable and not prohibited
        if panel == "fineweb-edu":
            acceptable = (
                checks["education"] == "pass"
                and checks["coherence"] == "pass"
                and all(
                    checks[k] not in {"defect", "uncertain"}
                    for k in ("notation", "completeness", "visuals", "contamination")
                )
            )
            usable = usable and acceptable
        c["usable"] += usable
        c["failed"] += not usable
        c["severe"] += row["severe"]
        if row["critical"]:
            blockers.append(ident + ": critical finding")
    for panel, total in expected.items():
        c = panels.get(panel, Counter())
        required = (
            protocol["usable_window_minimum"]
            if panel == "windows"
            else protocol["fineweb_usable_minimum"]
            if panel == "fineweb-edu"
            else protocol["usable_per_source_minimum"]
        )
        severe = (
            protocol["severe_window_maximum"]
            if panel == "windows"
            else 0
            if panel in {"finemath-4plus", "fineweb-edu"}
            else protocol["severe_per_source_maximum"]
        )
        if c["failed"] > total - required:
            blockers.append(panel + ": usable threshold exceeded")
        if c["severe"] > severe:
            blockers.append(panel + ": severe-defect threshold exceeded")
        if c["prohibited"]:
            blockers.append(panel + ": prohibited or unresolved defect")
    complete = len(seen) == len(mapping)
    return {
        "status": "failed" if blockers else "passed" if complete else "incomplete",
        "training_allowed": complete and not blockers,
        "reviewed": len(seen),
        "unreviewed": len(mapping) - len(seen),
        "blockers": blockers,
        "panels": {k: dict(v) for k, v in panels.items()},
        "protocol_version": VERSION,
        "reviewer": "source-blinded evaluation agents; not independent human approval",
    }
