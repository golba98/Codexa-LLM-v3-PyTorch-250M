"""Authorized language-only successor; never reinterprets historical decisions."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from codexa.data.production import digest
from codexa.data.quality_protocol import CHECKS, STATUSES

VERSION = "codexa-language-quality-1"
V2_VERSION = "codexa-language-quality-2"
V3_VERSION = "codexa-language-quality-3"
V4_VERSION = "codexa-language-quality-4"
V5_VERSION = "codexa-language-quality-5"
V6_VERSION = "codexa-language-quality-6"
V7_VERSION = "codexa-language-quality-7"
GENERAL_RUBRIC_SHA256 = "5ef8a3c844329164a649e123c43ba1f56d21ea751a9813372b41b8b3b0031454"
GENERAL_PANEL_SHA256 = "62d9aedb1f4ef4ad53eb9696a3f53e7f952c025a7e61d1efea362a957ff29f7d"
GENERAL_DECISIONS = {
    "U0": "usable",
    "U1": "usable",
    "R2": "reject",
    "H": "hold",
    "X": "out_of_scope",
    "C": "reject",
}
SUPPORTED_VERSIONS = frozenset(
    {VERSION, V2_VERSION, V3_VERSION, V4_VERSION, V5_VERSION, V6_VERSION, V7_VERSION}
)
V3_SOURCES = {
    "dclm-edu": {
        "repo": "HuggingFaceTB/dclm-edu",
        "revision": "dbad8ad71224482740cd9c9d353591adbf62fe04",
    },
    "pressbooks-filtered": {
        "repo": "common-pile/pressbooks_filtered",
        "revision": "1a1d3b50d77f834370f8eb4c0d174668dd1676bb",
    },
}
V4_SOURCES = {
    "tinystories-gpt4": {
        "repo": "roneneldan/TinyStories",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "path": "TinyStoriesV2-GPT4-train.txt",
        "sha256": "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
        "bytes": 2227753162,
    }
}
DEFERRED_SOURCES = {"finemath-4plus", "fineweb-edu", "openwebmath", "fineweb"}


def specification(
    sources: list[str], attempt: int = 0, *, version: str = VERSION
) -> dict[str, Any]:
    if (
        version not in SUPPORTED_VERSIONS
        or not sources
        or len(set(sources)) != len(sources)
        or any(not isinstance(s, str) or not s.strip() for s in sources)
        or DEFERRED_SOURCES.intersection(sources)
        or type(attempt) is not int
        or attempt < 0
    ):
        raise ValueError("Language protocol requires distinct eligible sources and a valid attempt")
    if version == V3_VERSION and (len(sources) != 1 or sources[0] not in V3_SOURCES):
        raise ValueError("Version 3 requires one separately authorized DCLM or Pressbooks source")
    if version == V4_VERSION and (len(sources) != 1 or sources[0] not in V4_SOURCES):
        raise ValueError("Version 4 requires the separately authorized TinyStories GPT4 source")
    if version == V5_VERSION and sources != ["wikimedia-wikipedia"]:
        raise ValueError("Version 5 requires the pinned Wikimedia Wikipedia source")
    if version == V6_VERSION and sources != ["stackexchange-general-english"]:
        raise ValueError("Version 6 requires the pinned general-English Stack Exchange source")
    if version == V7_VERSION and sources != ["wikimedia-wikipedia"]:
        raise ValueError("Version 7 requires the authorized Wikimedia reevaluation")
    spec: dict[str, Any] = {
        "version": version,
        "kind": "language",
        "sources": sorted(sources),
        "attempt": attempt,
        "seed": {
            VERSION: 2026111001,
            V2_VERSION: 2026111002,
            V3_VERSION: 2026111003,
            V4_VERSION: 2026111004,
            V5_VERSION: 2026111005,
            V6_VERSION: 2026111006,
            V7_VERSION: 2026111005,
        }[version]
        + 1000 * attempt,
        "windows": 300,
        "window_tokens": 1024,
        "documents_per_source": 100,
        "maximum_full_document_tokens": None,
        "usable_window_minimum": 270,
        "usable_per_source_minimum": 90,
        "severe_window_maximum": 15,
        "severe_per_source_maximum": 2,
        "critical_maximum": 0,
        "complete_required": True,
        "early_stop": False,
        "independent_reviewers_required": 2,
        "required_checks": list(CHECKS),
        "rubric": (
            "Read every complete document. Usable means grammatical, meaningful connected English, "
            "coherent topic progression and adequate context. Assess completeness, coherence, "
            "factual/solution correctness where assessable, essential visual dependencies, notation "
            "and contamination separately. Education may be not_applicable for narratives. "
            "Notation and visuals may be not_applicable when unnecessary. Mark correctness "
            "not_applicable for fictional premises, not for unverified factual claims. Unresolved "
            "checks make an item unusable. A severe defect substantially destroys meaning, "
            "completeness or reliability. Critical means an observed protected-evaluation leak or "
            "sensitive identifier; public biographical names alone are not critical. Native BOS/EOS "
            "transitions and random-window edge crops are not extraction defects. Source style "
            "may compromise blinding. Ratings do not certify universal factual correctness."
        ),
        "disagreement_policy": "Preserve both independent ratings; conservatively combine them; unresolved concerns fail.",
        "exclusion_policy": "Exclude every previously inspected document identity and verified 64-token review passage, including development and incomplete census packets, from both sample types.",
        "representativeness_policy": "Report source document and token eligibility; block if any source loses over 5% of its training tokens or documents to inspection exclusions.",
        "replaces": "Old fixed-source panels and FineMath/FineWeb-Edu extension for this new language corpus only; historical decisions remain unchanged.",
    }

    if version in {V2_VERSION, V3_VERSION, V4_VERSION}:
        spec.update(
            development_network_maximum_bytes=1024**3,
            acquisition_maximum_bytes=16 * 1024**3,
            base_language_readiness={
                "development_completions": 100,
                "grammatical_coherent_complete_minimum": 80,
                "severe_loop_maximum": 10,
            },
            replaces=(
                "Version-1 fixed source mixture for this new language corpus only; "
                "all quality thresholds, historical decisions and launch gates remain unchanged."
            ),
        )
    if version == V3_VERSION:
        spec.update(
            source_bindings={source: dict(V3_SOURCES[source]) for source in sources},
            replaces=(
                "Version-2 Common-Corpus-only source binding for this new language corpus only; "
                "all quality thresholds, original-rights requirements, historical decisions and "
                "launch gates remain unchanged."
            ),
        )
    if version == V4_VERSION:
        spec.update(
            source_bindings={source: dict(V4_SOURCES[source]) for source in sources},
            replaces=(
                "Version-3 DCLM-or-Pressbooks-only source binding and prior15-percent "
                "TinyStories narrative-share proposal, solely for one50M-token diagnostic "
                "English-fluency continuation from the preserved500M checkpoint; all "
                "existing quality thresholds and launch gates remain unchanged."
            ),
        )
    if version == V5_VERSION:
        spec.update(
            source_bindings={
                "wikimedia-wikipedia": {
                    "repo": "common-pile/wikimedia_filtered",
                    "revision": "0641bb84bd9b7162bcddf8be7836822161a9a342",
                }
            },
            replaces="Historical source bindings and budgets; thresholds remain unchanged.",
        )
    if version == V6_VERSION:
        spec.update(
            source_bindings={
                "stackexchange-general-english": {
                    "repo": "common-pile/stackexchange_filtered",
                    "revision": "c0ac7373830c688a43fc12d1988c4b19ccd884ab",
                }
            },
            replaces="Historical source bindings and budgets; thresholds remain unchanged.",
        )
    if version == V7_VERSION:
        spec.update(
            source_bindings={
                "wikimedia-wikipedia": {
                    "repo": "common-pile/wikimedia_filtered",
                    "revision": "0641bb84bd9b7162bcddf8be7836822161a9a342",
                }
            },
            approved_rubric_sha256=GENERAL_RUBRIC_SHA256,
            original_panel_sha256=GENERAL_PANEL_SHA256,
            rating_schema="general-English-bands-1",
            required_checks=[
                "notation",
                "completeness",
                "visual_dependence",
                "factual_reliability",
                "language_coherence",
                "contamination",
            ],
            rubric=(
                "User-approved general-English rubric: U0/U1 admit coherent independently "
                "useful English, retaining genuine localized defects. R2 rejects substantial "
                "semantic damage or materially false central accounts; H holds material "
                "uncertainty; X excludes non-target scope; C retains critical exclusions. "
                "Peripheral uncertainty, incidental missing details, auxiliary media and "
                "normal native/window boundaries do not automatically fail a useful account. "
                "The separately hashed approved rubric is authoritative."
            ),
            exclusion_policy=(
                "Explicitly authorized same-sample re-adjudication of the preserved hashed "
                "100 documents and 300 windows, not unseen data. All inspected identities "
                "and verified review passages remain excluded from training and heldouts."
            ),
            replaces=(
                "Version-5 quality interpretation for this preserved Wikimedia panel only; "
                "all numerical thresholds, original judgments, rights, protected exclusions "
                "and explicit training-launch requirements remain unchanged."
            ),
        )
    return spec


def validate_specification(protocol: dict[str, Any]) -> None:
    expected = specification(
        protocol.get("sources", []),
        protocol.get("attempt", -1),
        version=protocol.get("version", ""),
    )
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Language protocol differs from its frozen specification")


def specification_hash(protocol: dict[str, Any]) -> str:
    validate_specification(protocol)
    spec = specification(protocol["sources"], protocol["attempt"], version=protocol["version"])
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def authorized_budget(record: dict[str, Any], version: str) -> bool:
    """Preserve historical slots; current Wikimedia launch is limited to exactly 500M."""
    budget = record.get("additional_target_tokens")
    if type(budget) is not int:
        return False
    if version in {V5_VERSION, V6_VERSION, V7_VERSION}:
        return (
            budget == (500_000_000 if version == V7_VERSION else 1_000_000_000)
            and record.get("cumulative_target_tokens") == 500_006_912 + budget
            and record.get("training_launch_authorized") is True
            and bool(record.get("user_decision"))
        )
    if budget == 50_000_000:
        return True
    return (
        version == V4_VERSION
        and budget == 100_000_000
        and record.get("supersedes_additional_target_tokens") == 50_000_000
        and record.get("cumulative_target_tokens") == 600_006_912
        and bool(record.get("user_decision"))
    )


def validate_authorization(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate the recorded explicit user decision, never infer approval from a flag."""
    validate_specification(protocol)
    record = json.loads(path.read_text())
    if protocol["version"] in {V5_VERSION, V6_VERSION, V7_VERSION} and (
        not protocol.get("stream_config_sha256")
        or record.get("stream_config_sha256") != protocol["stream_config_sha256"]
        or record.get("training_config_identity") != protocol.get("training_config_identity")
        or not protocol.get("training_config_identity")
    ):
        raise ValueError(
            "Expanded streaming authorization is not bound to this source and training configuration"
        )
    if (
        record.get("version") != protocol["version"]
        or record.get("status") != "authorized"
        or record.get("specification_sha256") != specification_hash(protocol)
        or record.get("sources") != protocol["sources"]
        or not authorized_budget(record, protocol["version"])
        or not record.get("user_decision")
        or not record.get("historical_roots")
        or record.get("replaces") != protocol["replaces"]
        or protocol.get("authorization_sha256") != digest(path)
        or Path(protocol.get("authorization_path", "")).resolve() != path.resolve()
    ):
        raise ValueError(
            "Explicit language-protocol authorization is missing, stale or incompatible"
        )
    return record


def general_usable(row: dict[str, Any]) -> bool:
    """Validate an explicit authorized-band judgment; localized defects remain usable."""
    band = row.get("band")
    if band not in GENERAL_DECISIONS or row.get("decision") != GENERAL_DECISIONS[band]:
        raise ValueError("Invalid general-English band or decision")
    if row.get("scope") not in {"eligible", "out_of_scope"} or (
        (band == "X") != (row["scope"] == "out_of_scope")
    ):
        raise ValueError("General-English scope and decision differ")
    if any(
        type(row.get(key)) is not bool
        for key in ("coverage_complete", "severe", "critical", "confirmed_source_defect")
    ):
        raise ValueError("Explicit general-English evidence flags required")
    if bool(row["critical"]) != (band == "C"):
        raise ValueError("Critical finding requires critical exclusion")
    usable = band in {"U0", "U1"}
    if usable and (row["severe"] or not row["coverage_complete"]):
        raise ValueError("Incomplete or severe material cannot be marked usable")
    for key in ("reason", "residual_utility"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("Item-specific general-English justification required")
    evidence = row.get("observed_evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(value, str) or not value for value in evidence)
        or not isinstance(row.get("defect_categories"), list)
        or any(not isinstance(value, str) for value in row["defect_categories"])
    ):
        raise ValueError("General-English defect annotations and evidence required")
    return usable


def adjudicate(
    mapping: list[dict[str, Any]], ratings: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    validate_specification(protocol)
    general = protocol["version"] == V7_VERSION
    by_id = {r["id"]: r for r in mapping}
    dimensions = {s: 100 for s in protocol["sources"]} | {"windows": 300}
    expected = Counter("windows" if r["kind"] == "window" else r["source"] for r in mapping)
    if len(by_id) != len(mapping) or dict(expected) != dimensions:
        raise ValueError("Language panel identities or dimensions differ from preregistration")
    documents = [r["document_id"] for r in mapping if r["kind"] == "document"]
    if len(set(documents)) != len(documents):
        raise ValueError("Duplicate language panel document")
    if any(r["kind"] not in {"document", "window"} for r in mapping):
        raise ValueError("Unknown language sample kind")
    seen: set[str] = set()
    panels: dict[str, Counter[str]] = {}
    blockers = []
    for row in ratings:
        ident = row["id"]
        if ident in seen or ident not in by_id or not str(row.get("reason", "")).strip():
            raise ValueError("Invalid, duplicate or unexplained language rating")
        flags: tuple[str, ...] = ("severe", "critical", "coverage_complete")
        if not general:
            flags += ("usable",)
        if any(type(row.get(k)) is not bool for k in flags):
            raise ValueError("Explicit language labels and full coverage required")
        sample = by_id[ident]
        if general:
            usable = general_usable(row)
            if row.get("kind") != sample["kind"]:
                raise ValueError("General-English sample kind differs")
        else:
            checks = row.get("checks", {})
            if set(checks) != set(CHECKS) or any(v not in STATUSES for v in checks.values()):
                raise ValueError("All language quality checks are required")
            usable = (
                row["usable"]
                and checks["completeness"] == "pass"
                and checks["coherence"] == "pass"
                and checks["contamination"] == "pass"
                and all(v not in {"defect", "uncertain"} for v in checks.values())
            )
        if sample["kind"] == "document" and sample.get("coverage") != "full document":
            raise ValueError("Language review requires complete source documents")
        seen.add(ident)
        panel = "windows" if sample["kind"] == "window" else sample["source"]
        counts = panels.setdefault(panel, Counter())
        counts["reviewed"] += 1
        counts["usable"] += (
            usable and row["coverage_complete"] and not row["severe"] and not row["critical"]
        )
        counts["severe"] += row["severe"]
        if row["critical"] or not row["coverage_complete"]:
            blockers.append(ident + ": critical finding or incomplete coverage")
    for panel, total in dimensions.items():
        counts = panels.get(panel, Counter())
        minimum = 270 if panel == "windows" else 90
        maximum = 15 if panel == "windows" else 2
        if counts["reviewed"] - counts["usable"] > total - minimum:
            blockers.append(panel + ": usable threshold exceeded")
        if counts["severe"] > maximum:
            blockers.append(panel + ": severe-defect threshold exceeded")
    complete = len(seen) == len(mapping)
    return {
        "status": "failed" if blockers else "passed" if complete else "incomplete",
        "training_allowed": complete and not blockers,
        "reviewed": len(seen),
        "unreviewed": len(mapping) - len(seen),
        "panels": {k: dict(v) for k, v in panels.items()},
        "blockers": blockers,
        "protocol_version": protocol["version"],
        "reviewer": "two independent source-blinded agents; not human consensus",
    }


def combine_reviews(
    inputs: list[list[dict[str, Any]]], *, version: str | None = None
) -> list[dict[str, Any]]:
    """Conservative combination of exactly two complete independent ratings."""
    if len(inputs) != 2:
        raise ValueError("Exactly two independent reviews required")
    keyed = [{row["id"]: row for row in rows} for rows in inputs]
    if set(keyed[0]) != set(keyed[1]) or any(len(k) != len(rows) for k, rows in zip(keyed, inputs)):
        raise ValueError("Independent rating identities differ or repeat")
    if version == V7_VERSION:
        combined = []
        for ident in keyed[0]:
            pair = [k[ident] for k in keyed]
            for row in pair:
                general_usable(row)
            if pair[0].get("kind") != pair[1].get("kind"):
                raise ValueError("Independent sample kinds differ")
            bands = {row["band"] for row in pair}
            if "C" in bands:
                band = "C"
            elif bands == {"X"}:
                band = "X"
            elif "X" in bands:
                band = "H"
            elif "R2" in bands:
                band = "R2"
            elif "H" in bands:
                band = "H"
            else:
                band = "U1" if "U1" in bands else "U0"
            combined.append(
                {
                    "id": ident,
                    "kind": pair[0]["kind"],
                    "band": band,
                    "decision": GENERAL_DECISIONS[band],
                    "scope": "out_of_scope" if band == "X" else "eligible",
                    "coverage_complete": all(row["coverage_complete"] for row in pair),
                    "severe": any(row["severe"] for row in pair),
                    "critical": any(row["critical"] for row in pair),
                    "confirmed_source_defect": all(row["confirmed_source_defect"] for row in pair),
                    "defect_categories": sorted(
                        set(pair[0]["defect_categories"]) | set(pair[1]["defect_categories"])
                    ),
                    "observed_evidence": pair[0]["observed_evidence"]
                    + pair[1]["observed_evidence"],
                    "reason": " | ".join(
                        f"Reviewer {i + 1}: {row['reason']}" for i, row in enumerate(pair)
                    ),
                    "residual_utility": " | ".join(row["residual_utility"] for row in pair),
                }
            )
        return combined
    order = {"not_applicable": 0, "pass": 1, "uncertain": 2, "defect": 3}
    return [
        {
            "id": ident,
            "usable": all(k[ident]["usable"] for k in keyed),
            "severe": any(k[ident]["severe"] for k in keyed),
            "critical": any(k[ident]["critical"] for k in keyed),
            "coverage_complete": all(k[ident]["coverage_complete"] for k in keyed),
            "checks": {
                check: max((k[ident]["checks"][check] for k in keyed), key=order.__getitem__)
                for check in CHECKS
            },
            "reason": " | ".join(
                f"Reviewer {i + 1}: {k[ident]['reason']}" for i, k in enumerate(keyed)
            ),
        }
        for ident in keyed[0]
    ]
